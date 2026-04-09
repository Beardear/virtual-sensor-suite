"""
Virtual Radar via Ray-Casting on 3D Gaussian Splatting.

Fires sparse radar rays into the Gaussian Splat scene and computes
range, RCS (radar cross-section), and radial velocity per detection.
Supports:
  - Configurable range, azimuth/elevation FoV, and bin count
  - RCS estimation from Gaussian opacity and scale
  - Ego-motion Doppler velocity (radial component)
  - Output as structured detections (.bin / .pcd)

Follows the same volumetric ray-casting approach as VirtualLiDAR,
but with radar-specific physics: sparser rays, RCS instead of
intensity, and Doppler radial velocity from ego motion.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .camera import GaussianSplatScene, extrinsics_to_matrix


@dataclass
class RadarConfig:
    """Configuration for a virtual radar sensor."""
    name: str
    max_range_m: float = 200.0
    azimuth_fov_deg: float = 60.0
    elevation_fov_deg: float = 10.0
    num_azimuth_bins: int = 64
    num_elevation_bins: int = 8
    rcs_model: str = "opacity_based"  # "opacity_based" or "constant"
    T_vehicle_sensor: np.ndarray = None

    def __post_init__(self):
        if self.T_vehicle_sensor is None:
            self.T_vehicle_sensor = np.eye(4, dtype=np.float64)


class VirtualRadar:
    """
    Virtual radar sensor using volumetric ray marching through Gaussians.

    For each radar ray:
      1. Generate ray direction from sensor geometry (azimuth × elevation grid)
      2. Transform ray origin/direction to world frame
      3. March along the ray, accumulating opacity from intersected Gaussians
      4. Compute expected range and RCS via volumetric rendering
      5. Compute radial velocity from ego motion (Doppler)

    Radar is much sparser than LiDAR (hundreds of rays vs tens of thousands),
    but covers longer range (200m vs 80m).
    """

    def __init__(
        self,
        config: RadarConfig,
        scene: GaussianSplatScene,
    ):
        self.config = config
        self.scene = scene
        self.device = scene.device

        # Pre-compute ray directions in sensor frame
        self._ray_dirs, self._ray_origins = self._generate_ray_pattern()

    @classmethod
    def from_config(cls, cfg: dict, scene: GaussianSplatScene) -> "VirtualRadar":
        """Instantiate from YAML config dict."""
        T_vs = extrinsics_to_matrix(cfg["extrinsics"])

        config = RadarConfig(
            name=cfg["name"],
            max_range_m=cfg.get("max_range_m", 200.0),
            azimuth_fov_deg=cfg.get("azimuth_fov_deg", 60.0),
            elevation_fov_deg=cfg.get("elevation_fov_deg", 10.0),
            num_azimuth_bins=cfg.get("num_azimuth_bins", 64),
            num_elevation_bins=cfg.get("num_elevation_bins", 8),
            rcs_model=cfg.get("rcs_model", "opacity_based"),
            T_vehicle_sensor=T_vs,
        )
        return cls(config, scene)

    def _generate_ray_pattern(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the radar ray pattern in sensor-local coordinates.

        Returns:
            ray_dirs: (N_rays, 3) unit direction vectors in sensor frame.
            ray_origins: (N_rays, 3) ray origins (all at sensor origin).
        """
        cfg = self.config
        n_rays = cfg.num_azimuth_bins * cfg.num_elevation_bins

        # Azimuth: symmetric around forward (0°)
        half_az = cfg.azimuth_fov_deg / 2.0
        azimuths = np.linspace(
            np.radians(-half_az),
            np.radians(half_az),
            cfg.num_azimuth_bins,
            endpoint=True,
        )

        # Elevation: symmetric around horizontal (0°)
        half_el = cfg.elevation_fov_deg / 2.0
        elevations = np.linspace(
            np.radians(-half_el),
            np.radians(half_el),
            cfg.num_elevation_bins,
            endpoint=True,
        )

        # Build ray directions: spherical -> Cartesian
        # Convention: x-forward, y-left, z-up
        dirs = np.zeros((cfg.num_elevation_bins, cfg.num_azimuth_bins, 3), dtype=np.float64)
        for el_idx, elev in enumerate(elevations):
            cos_e = np.cos(elev)
            sin_e = np.sin(elev)
            for az_idx, az in enumerate(azimuths):
                dirs[el_idx, az_idx, 0] = cos_e * np.cos(az)   # x (forward)
                dirs[el_idx, az_idx, 1] = cos_e * np.sin(az)   # y (left)
                dirs[el_idx, az_idx, 2] = sin_e                 # z (up)

        dirs = dirs.reshape(-1, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        ray_dirs = torch.tensor(dirs, dtype=torch.float32, device=self.device)
        ray_origins = torch.zeros(n_rays, 3, dtype=torch.float32, device=self.device)

        return ray_dirs, ray_origins

    def _ray_gaussian_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        max_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute volumetric ray-Gaussian intersection for radar rays.

        Similar to LiDAR intersection but computes RCS instead of intensity.
        RCS is estimated from Gaussian opacity and physical scale (cross-section area).

        Returns:
            depths: (N_rays,) expected range per ray.
            rcs_values: (N_rays,) estimated RCS per ray (dBsm-like).
            hit_mask: (N_rays,) bool — True if ray accumulated significant opacity.
        """
        N_rays = ray_origins.shape[0]

        all_depths = torch.zeros(N_rays, device=self.device)
        all_rcs = torch.zeros(N_rays, device=self.device)
        all_alphas = torch.zeros(N_rays, device=self.device)

        # Pre-compute Gaussian scale
        scales_exp = torch.exp(scales)  # (M, 3)

        # RCS proxy: cross-section area from Gaussian scale (product of two largest axes)
        sorted_scales, _ = scales_exp.sort(dim=1, descending=True)
        cross_section = sorted_scales[:, 0] * sorted_scales[:, 1]  # (M,)

        # Average variance for isotropic approximation
        avg_var = (scales_exp ** 2).mean(dim=1)  # (M,)

        # Process all rays (radar has few enough rays to do in one batch)
        CHUNK_SIZE = 1024
        for chunk_start in range(0, N_rays, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_rays)
            chunk_origins = ray_origins[chunk_start:chunk_end]
            chunk_dirs = ray_dirs[chunk_start:chunk_end]
            C = chunk_end - chunk_start

            # Coarse spatial culling
            chunk_center = chunk_origins.mean(dim=0)
            chunk_spread = torch.norm(chunk_origins - chunk_center, dim=1).max().item()
            cull_radius = max_range + chunk_spread + 5.0

            dist_to_center = torch.norm(means - chunk_center, dim=1)
            coarse_mask = dist_to_center < cull_radius
            # Filter out suppressed (zero-opacity) Gaussians
            coarse_mask &= opacities[:, 0] > 1e-3
            if coarse_mask.sum() == 0:
                continue

            local_means = means[coarse_mask]
            local_opac = opacities[coarse_mask]
            local_avg_var = avg_var[coarse_mask]
            local_cross_section = cross_section[coarse_mask]
            M_local = local_means.shape[0]

            # Per-ray Gaussian intersection
            delta = local_means.unsqueeze(0) - chunk_origins.unsqueeze(1)  # (C, M', 3)
            t_closest = torch.einsum("cmj,cj->cm", delta, chunk_dirs)  # (C, M')
            t_closest = t_closest.clamp(min=0.1, max=max_range)

            closest_pts = chunk_origins.unsqueeze(1) + \
                          t_closest.unsqueeze(-1) * chunk_dirs.unsqueeze(1)
            dist_vec = closest_pts - local_means.unsqueeze(0)
            dist_sq = (dist_vec ** 2).sum(dim=-1)

            gauss_weight = torch.exp(-0.5 * dist_sq / local_avg_var.clamp(min=1e-4))
            alpha_per = local_opac.squeeze(-1) * gauss_weight

            # Top-K contributing Gaussians per ray
            K = min(32, M_local)
            topk_alpha, topk_idx = alpha_per.topk(K, dim=1)

            topk_depths = t_closest.gather(1, topk_idx)

            # RCS per Gaussian: opacity × cross-section area
            rcs_per_gaussian = (local_opac.squeeze(-1) * local_cross_section)
            topk_rcs = rcs_per_gaussian.unsqueeze(0).expand(C, -1).gather(1, topk_idx)

            # Sort by depth (front-to-back)
            sort_idx = topk_depths.argsort(dim=1)
            topk_alpha = topk_alpha.gather(1, sort_idx)
            topk_depths = topk_depths.gather(1, sort_idx)
            topk_rcs = topk_rcs.gather(1, sort_idx)

            # Volumetric rendering: front-to-back compositing
            topk_alpha = topk_alpha.clamp(max=0.99)
            transmittance = torch.ones(C, device=self.device)
            depth_acc = torch.zeros(C, device=self.device)
            rcs_acc = torch.zeros(C, device=self.device)
            alpha_acc = torch.zeros(C, device=self.device)

            for ki in range(K):
                a = topk_alpha[:, ki]
                w = transmittance * a
                depth_acc += w * topk_depths[:, ki]
                rcs_acc += w * topk_rcs[:, ki]
                alpha_acc += w
                transmittance *= (1.0 - a)
                if transmittance.max() < 0.001:
                    break

            all_depths[chunk_start:chunk_end] = depth_acc
            all_rcs[chunk_start:chunk_end] = rcs_acc
            all_alphas[chunk_start:chunk_end] = alpha_acc

        # Hit mask: accumulated alpha above threshold
        hit_mask = all_alphas > 0.1

        # Normalize by accumulated alpha
        safe_alpha = all_alphas.clamp(min=1e-6)
        all_depths = all_depths / safe_alpha
        all_depths[~hit_mask] = 0.0
        all_rcs = all_rcs / safe_alpha
        all_rcs[~hit_mask] = 0.0

        return all_depths, all_rcs, hit_mask

    def render(
        self,
        ego_pose_matrix: np.ndarray,
        ego_velocity: Optional[np.ndarray] = None,
        actor_velocities: Optional[list[tuple[np.ndarray, np.ndarray, float]]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a radar scan from the given ego pose.

        Args:
            ego_pose_matrix: 4x4 T_world_vehicle transform.
            ego_velocity: (3,) ego velocity vector in world frame [m/s].
                         Used for Doppler radial velocity computation.
                         If None, radial velocity is set to zero.
            actor_velocities: Optional list of (center, velocity, radius) tuples
                            for dynamic actors. Detections near an actor center
                            use the actor's velocity for Doppler instead of zero.

        Returns:
            dict with:
                'points': (N, 3) float32 — 3D detections in world frame
                'rcs': (N,) float32 — radar cross-section per detection
                'radial_velocity': (N,) float32 — Doppler radial velocity [m/s]
                'depth': (N,) float32 — range in meters
                'n_total_rays': int — total rays fired
                'n_valid_points': int — number of valid detections
        """
        T_wv = ego_pose_matrix
        T_vs = self.config.T_vehicle_sensor
        T_ws = T_wv @ T_vs  # world <- sensor

        T_ws_t = torch.tensor(T_ws, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Transform ray origins and directions to world frame
            R_ws = T_ws_t[:3, :3]
            t_ws = T_ws_t[:3, 3]
            ray_dirs_world = (R_ws @ self._ray_dirs.T).T  # (N, 3)
            ray_origins_world = t_ws.unsqueeze(0).expand_as(ray_dirs_world)

            # Ray-Gaussian intersection
            depths, rcs_values, hit_mask = self._ray_gaussian_intersection(
                ray_origins_world,
                ray_dirs_world,
                self.scene.means,
                self.scene.scales,
                self.scene.rotations,
                self.scene.opacities,
                self.scene.sh_coeffs,
                max_range=self.config.max_range_m,
            )

            # Inject actor hits via ray-AABB intersection.
            actor_boxes = getattr(self.scene, 'actor_boxes', [])
            for center, dims in actor_boxes:
                half = torch.tensor(dims, dtype=torch.float32, device=self.device) / 2.0
                box_min = torch.tensor(center, dtype=torch.float32, device=self.device) - half
                box_max = torch.tensor(center, dtype=torch.float32, device=self.device) + half

                sign = torch.sign(ray_dirs_world)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                safe_dirs = torch.where(ray_dirs_world.abs() < 1e-8,
                                        sign * 1e-8, ray_dirs_world)
                inv_dir = 1.0 / safe_dirs
                t1 = (box_min - ray_origins_world) * inv_dir
                t2 = (box_max - ray_origins_world) * inv_dir
                t_near = torch.min(t1, t2)
                t_far = torch.max(t1, t2)
                t_enter = t_near.max(dim=1).values.clamp(min=0.5)
                t_exit = t_far.min(dim=1).values

                box_hit = (t_enter < t_exit) & (t_exit > 0.5)
                depths[box_hit] = t_enter[box_hit]
                rcs_values[box_hit] = 10.0  # Typical vehicle RCS (dBsm)
                hit_mask[box_hit] = True

            # Range filter
            range_mask = (depths > 0.5) & (depths < self.config.max_range_m)
            valid_mask = hit_mask & range_mask

            # Compute 3D points in world frame
            points_world = ray_origins_world + ray_dirs_world * depths.unsqueeze(-1)

            # Extract valid detections
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            points_out = points_world[valid_indices].cpu().numpy()
            rcs_out = rcs_values[valid_indices].clamp(0).cpu().numpy()
            depth_out = depths[valid_indices].cpu().numpy()

            # Compute radial velocity (Doppler) from ego + actor motion
            if ego_velocity is not None:
                v_ego = torch.tensor(ego_velocity, dtype=torch.float32, device=self.device)
                valid_dirs = ray_dirs_world[valid_indices]  # (N, 3)

                # For each detection, determine the object's velocity
                # Background objects are static (v_object = 0)
                # Actor objects use their own velocity
                n_pts = len(points_out)
                v_object = np.zeros((n_pts, 3), dtype=np.float32)

                if actor_velocities is not None:
                    for center, actor_vel, radius in actor_velocities:
                        dists = np.linalg.norm(points_out - center, axis=1)
                        near_mask = dists < radius
                        v_object[near_mask] = actor_vel

                v_obj_t = torch.tensor(v_object, dtype=torch.float32, device=self.device)

                # v_radial = dot(v_ego - v_object, ray_dir)
                # Relative velocity of object w.r.t. sensor, projected onto ray
                v_relative = v_ego.unsqueeze(0) - v_obj_t  # (N, 3)
                v_radial = torch.einsum("nj,nj->n", v_relative, valid_dirs)
                radial_velocity_out = v_radial.cpu().numpy()
            else:
                radial_velocity_out = np.zeros(len(points_out), dtype=np.float32)

        n_rays = self._ray_dirs.shape[0]

        return {
            "points": points_out.astype(np.float32),
            "rcs": rcs_out.astype(np.float32),
            "radial_velocity": radial_velocity_out.astype(np.float32),
            "depth": depth_out.astype(np.float32),
            "n_total_rays": n_rays,
            "n_valid_points": len(points_out),
        }

    @staticmethod
    def save_radar_bin(filepath: str, points: np.ndarray, rcs: np.ndarray,
                       radial_velocity: np.ndarray):
        """
        Save radar detections in binary format.

        Format: N x 5 float32 (x, y, z, rcs, v_radial).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([points, rcs, radial_velocity]).astype(np.float32)
        data.tofile(str(path))

    @staticmethod
    def save_radar_pcd(filepath: str, points: np.ndarray, rcs: np.ndarray,
                       radial_velocity: np.ndarray):
        """
        Save radar detections in PCD format (ASCII).

        Fields: x y z rcs radial_velocity
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        N = points.shape[0]

        with open(path, "w") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rcs radial_velocity\n")
            f.write("SIZE 4 4 4 4 4\n")
            f.write("TYPE F F F F F\n")
            f.write("COUNT 1 1 1 1 1\n")
            f.write(f"WIDTH {N}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {N}\n")
            f.write("DATA ascii\n")
            for i in range(N):
                f.write(
                    f"{points[i, 0]:.6f} {points[i, 1]:.6f} "
                    f"{points[i, 2]:.6f} {rcs[i]:.4f} {radial_velocity[i]:.4f}\n"
                )

    def __repr__(self) -> str:
        cfg = self.config
        n_rays = cfg.num_azimuth_bins * cfg.num_elevation_bins
        return (
            f"VirtualRadar('{cfg.name}', {cfg.num_azimuth_bins}az×{cfg.num_elevation_bins}el, "
            f"{n_rays} rays, range={cfg.max_range_m}m)"
        )
