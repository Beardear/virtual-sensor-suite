"""
Virtual LiDAR via Monte Carlo Ray-Casting on 3D Gaussian Splatting.

Fires virtual laser rays into the Gaussian Splat scene and accumulates
depth/intensity along each ray via volumetric rendering. Supports:
  - Configurable channel count, range, and FoV (e.g., Velodyne HDL-64E)
  - Stochastic ray drop modeling (learned ray_drop_prob from SplatAD)
  - Intensity estimation from Gaussian opacity/SH
  - Output as structured point clouds (.pcd / .bin)

This is the core novel contribution — bridging ML-learned scene representations
with physically-motivated LiDAR simulation.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .camera import GaussianSplatScene, extrinsics_to_matrix


@dataclass
class LiDARConfig:
    """Configuration for a virtual LiDAR sensor."""
    name: str
    channels: int = 64
    range_m: float = 80.0
    points_per_channel: int = 1024
    vertical_fov: tuple[float, float] = (-24.9, 2.0)   # (min_deg, max_deg)
    horizontal_fov: tuple[float, float] = (0.0, 360.0)  # (min_deg, max_deg)
    ray_drop_prob: float = 0.0                           # Global default
    intensity_model: str = "learned"                      # "learned" or "constant"
    T_vehicle_sensor: np.ndarray = None

    def __post_init__(self):
        if self.T_vehicle_sensor is None:
            self.T_vehicle_sensor = np.eye(4, dtype=np.float64)


class VirtualLiDAR:
    """
    Virtual LiDAR sensor using volumetric ray marching through Gaussians.

    For each laser ray:
      1. Generate ray direction from sensor geometry (channel elevation + azimuth)
      2. Transform ray origin/direction to world frame
      3. March along the ray, accumulating opacity from intersected Gaussians
      4. Compute expected depth and intensity via volumetric rendering
      5. Apply stochastic ray drop based on learned probability

    The ray-Gaussian intersection uses an efficient batched computation:
    for each ray, we find nearby Gaussians via spatial proximity, compute
    the ray-Gaussian distance, and evaluate the Gaussian contribution.
    """

    def __init__(
        self,
        config: LiDARConfig,
        scene: GaussianSplatScene,
    ):
        self.config = config
        self.scene = scene
        self.device = scene.device

        # Pre-compute ray directions in sensor frame
        self._ray_dirs, self._ray_origins = self._generate_ray_pattern()

    @classmethod
    def from_config(cls, cfg: dict, scene: GaussianSplatScene) -> "VirtualLiDAR":
        """Instantiate from YAML config dict."""
        T_vs = extrinsics_to_matrix(cfg["extrinsics"])
        v_fov = cfg.get("vertical_fov", [-24.9, 2.0])
        h_fov = cfg.get("horizontal_fov", [0.0, 360.0])

        config = LiDARConfig(
            name=cfg["name"],
            channels=cfg.get("channels", 64),
            range_m=cfg.get("range_m", 80.0),
            points_per_channel=cfg.get("points_per_channel", 1024),
            vertical_fov=tuple(v_fov),
            horizontal_fov=tuple(h_fov),
            ray_drop_prob=cfg.get("ray_drop_prob", 0.0),
            intensity_model=cfg.get("intensity_model", "learned"),
            T_vehicle_sensor=T_vs,
        )
        return cls(config, scene)

    def _generate_ray_pattern(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the LiDAR ray pattern in sensor-local coordinates.

        Returns:
            ray_dirs: (N_rays, 3) unit direction vectors in sensor frame.
            ray_origins: (N_rays, 3) ray origins (all at sensor origin).
        """
        cfg = self.config
        n_rays = cfg.channels * cfg.points_per_channel

        # Elevation angles: uniformly spaced across vertical FoV
        elevations = np.linspace(
            np.radians(cfg.vertical_fov[0]),
            np.radians(cfg.vertical_fov[1]),
            cfg.channels,
        )

        # Azimuth angles: uniformly spaced across horizontal FoV
        azimuths = np.linspace(
            np.radians(cfg.horizontal_fov[0]),
            np.radians(cfg.horizontal_fov[1]),
            cfg.points_per_channel,
            endpoint=False,
        )

        # Build ray directions: spherical -> Cartesian
        # Convention: x-forward, y-left, z-up
        dirs = np.zeros((cfg.channels, cfg.points_per_channel, 3), dtype=np.float64)
        for ch, elev in enumerate(elevations):
            cos_e = np.cos(elev)
            sin_e = np.sin(elev)
            for az_idx, az in enumerate(azimuths):
                dirs[ch, az_idx, 0] = cos_e * np.cos(az)   # x (forward)
                dirs[ch, az_idx, 1] = cos_e * np.sin(az)   # y (left)
                dirs[ch, az_idx, 2] = sin_e                 # z (up)

        dirs = dirs.reshape(-1, 3)
        # Normalize (should already be unit, but ensure numerical precision)
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
        Compute volumetric ray-Gaussian intersection for all rays in batch.

        For each ray, computes the expected depth and intensity by integrating
        Gaussian contributions along the ray using volumetric rendering:

            depth(r) = sum_i  T_i * alpha_i * t_i
            intensity(r) = sum_i  T_i * alpha_i * c_i

        where T_i = prod_{j<i} (1 - alpha_j) is the transmittance,
        alpha_i = opacity_i * G(d_i) is the per-Gaussian alpha,
        and d_i is the ray-to-Gaussian-center distance weighted by covariance.

        Args:
            ray_origins: (N_rays, 3) world-frame origins.
            ray_dirs: (N_rays, 3) world-frame unit directions.
            means: (M, 3) Gaussian centers.
            scales: (M, 3) log-scale parameters.
            rotations: (M, 4) quaternions.
            opacities: (M, 1) opacity values.
            sh_coeffs: (M, C, 3) SH coefficients.
            max_range: Maximum ray distance.

        Returns:
            depths: (N_rays,) expected depth per ray.
            intensities: (N_rays,) expected intensity per ray.
            hit_mask: (N_rays,) bool — True if ray accumulated significant opacity.
        """
        N_rays = ray_origins.shape[0]
        M = means.shape[0]

        # Process rays in chunks for memory efficiency
        CHUNK_SIZE = 4096
        all_depths = torch.zeros(N_rays, device=self.device)
        all_intensities = torch.zeros(N_rays, device=self.device)
        all_alphas = torch.zeros(N_rays, device=self.device)

        # Pre-compute Gaussian scale (use exp of log-scales)
        scales_exp = torch.exp(scales)  # (M, 3)
        # Effective radius for initial culling (3-sigma bound)
        max_scale = scales_exp.max(dim=1).values  # (M,)

        # Pre-compute SH DC color and grayscale intensity for all Gaussians
        C0 = 0.28209479177387814
        base_color = C0 * sh_coeffs[:, 0, :] + 0.5  # (M, 3)
        base_intensity = base_color.mean(dim=1)  # (M,) grayscale

        # Pre-compute average variance for isotropic approximation
        avg_var = (scales_exp ** 2).mean(dim=1)  # (M,)

        for chunk_start in range(0, N_rays, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_rays)
            chunk_origins = ray_origins[chunk_start:chunk_end]    # (C, 3)
            chunk_dirs = ray_dirs[chunk_start:chunk_end]          # (C, 3)
            C = chunk_end - chunk_start

            # --- Coarse spatial culling: restrict to Gaussians near this chunk ---
            # Compute chunk centroid and bounding radius
            chunk_center = chunk_origins.mean(dim=0)  # (3,)
            # The farthest a ray endpoint reaches from center
            chunk_spread = torch.norm(chunk_origins - chunk_center, dim=1).max().item()
            cull_radius = max_range + chunk_spread + 5.0  # generous margin

            # Distance from each Gaussian to chunk center
            dist_to_center = torch.norm(means - chunk_center, dim=1)  # (M,)
            coarse_mask = dist_to_center < cull_radius
            # Filter out suppressed (zero-opacity) Gaussians
            coarse_mask &= opacities[:, 0] > 1e-3
            if coarse_mask.sum() == 0:
                continue

            local_means = means[coarse_mask]         # (M', 3)
            local_opac = opacities[coarse_mask]      # (M', 1)
            local_avg_var = avg_var[coarse_mask]      # (M',)
            local_intensity = base_intensity[coarse_mask]  # (M',)
            M_local = local_means.shape[0]

            # --- Per-ray Gaussian intersection on culled set ---
            delta = local_means.unsqueeze(0) - chunk_origins.unsqueeze(1)  # (C, M', 3)

            # Parameter t along ray for closest point
            t_closest = torch.einsum("cmj,cj->cm", delta, chunk_dirs)  # (C, M')

            # Clamp to valid range
            t_closest = t_closest.clamp(min=0.1, max=max_range)

            # Closest point on ray to each Gaussian
            closest_pts = chunk_origins.unsqueeze(1) + \
                          t_closest.unsqueeze(-1) * chunk_dirs.unsqueeze(1)  # (C, M', 3)

            # Distance from closest point to Gaussian center
            dist_vec = closest_pts - local_means.unsqueeze(0)  # (C, M', 3)
            dist_sq = (dist_vec ** 2).sum(dim=-1)  # (C, M')

            # Gaussian contribution: weight by opacity and distance
            gauss_weight = torch.exp(-0.5 * dist_sq / local_avg_var.clamp(min=1e-4))  # (C, M')

            # Alpha per Gaussian per ray
            alpha_per = local_opac.squeeze(-1) * gauss_weight  # (C, M')

            # Select top-K contributing Gaussians per ray for efficiency
            K = min(64, M_local)
            topk_alpha, topk_idx = alpha_per.topk(K, dim=1)  # (C, K)

            # Gather corresponding depths and colors
            topk_depths = t_closest.gather(1, topk_idx)  # (C, K)

            topk_intensity = local_intensity.unsqueeze(0).expand(C, -1).gather(1, topk_idx)

            # Sort by depth within each ray (front-to-back)
            sort_idx = topk_depths.argsort(dim=1)
            topk_alpha = topk_alpha.gather(1, sort_idx)
            topk_depths = topk_depths.gather(1, sort_idx)
            topk_intensity = topk_intensity.gather(1, sort_idx)

            # Volumetric rendering: front-to-back compositing
            topk_alpha = topk_alpha.clamp(max=0.99)
            transmittance = torch.ones(C, device=self.device)
            depth_acc = torch.zeros(C, device=self.device)
            intensity_acc = torch.zeros(C, device=self.device)
            alpha_acc = torch.zeros(C, device=self.device)

            for ki in range(K):
                a = topk_alpha[:, ki]
                w = transmittance * a
                depth_acc += w * topk_depths[:, ki]
                intensity_acc += w * topk_intensity[:, ki]
                alpha_acc += w
                transmittance *= (1.0 - a)
                # Early termination check
                if transmittance.max() < 0.001:
                    break

            all_depths[chunk_start:chunk_end] = depth_acc
            all_intensities[chunk_start:chunk_end] = intensity_acc
            all_alphas[chunk_start:chunk_end] = alpha_acc

        # Hit mask: accumulated alpha above threshold
        hit_mask = all_alphas > 0.1

        # Normalize depth by accumulated alpha (expected value)
        safe_alpha = all_alphas.clamp(min=1e-6)
        all_depths = all_depths / safe_alpha
        all_depths[~hit_mask] = 0.0
        all_intensities = all_intensities / safe_alpha
        all_intensities[~hit_mask] = 0.0

        return all_depths, all_intensities, hit_mask

    def render(
        self,
        ego_pose_matrix: np.ndarray,
        learned_ray_drop: Optional[torch.Tensor] = None,
        stochastic: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a LiDAR scan from the given ego pose.

        Args:
            ego_pose_matrix: 4x4 T_world_vehicle transform.
            learned_ray_drop: Optional (N_rays,) per-ray drop probability
                             from the SplatAD model. Overrides config default.
            stochastic: If True, apply Monte Carlo ray dropping.
                       If False, return all rays (deterministic mode).

        Returns:
            dict with:
                'points': (N, 3) float32 — 3D points in world frame
                'intensity': (N,) float32 — per-point intensity [0, 1]
                'depth': (N,) float32 — per-point range in meters
                'ray_drop_mask': (N_total,) bool — which rays were dropped
                'ring': (N,) int32 — channel/ring index per point
                'azimuth': (N,) float32 — azimuth angle in radians
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
            depths, intensities, hit_mask = self._ray_gaussian_intersection(
                ray_origins_world,
                ray_dirs_world,
                self.scene.means,
                self.scene.scales,
                self.scene.rotations,
                self.scene.opacities,
                self.scene.sh_coeffs,
                max_range=self.config.range_m,
            )

            # Inject actor hits via ray-AABB intersection.
            # The volumetric rendering can miss actors due to transmittance
            # depletion from dense backgrounds, so we check explicitly.
            actor_boxes = getattr(self.scene, 'actor_boxes', [])
            for center, dims in actor_boxes:
                half = torch.tensor(dims, dtype=torch.float32, device=self.device) / 2.0
                box_min = torch.tensor(center, dtype=torch.float32, device=self.device) - half
                box_max = torch.tensor(center, dtype=torch.float32, device=self.device) + half

                # Slab method ray-AABB intersection
                sign = torch.sign(ray_dirs_world)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                safe_dirs = torch.where(ray_dirs_world.abs() < 1e-8,
                                        sign * 1e-8, ray_dirs_world)
                inv_dir = 1.0 / safe_dirs
                t1 = (box_min - ray_origins_world) * inv_dir
                t2 = (box_max - ray_origins_world) * inv_dir
                t_near = torch.min(t1, t2)
                t_far = torch.max(t1, t2)
                t_enter = t_near.max(dim=1).values
                t_exit = t_far.min(dim=1).values

                box_hit = (t_enter < t_exit) & (t_exit > 0.5)
                t_enter = t_enter.clamp(min=0.5)

                # Actor AABB hits override background depths — actors are
                # solid objects that should produce LiDAR returns.
                depths[box_hit] = t_enter[box_hit]
                intensities[box_hit] = 0.7
                hit_mask[box_hit] = True

            # Range filter
            range_mask = (depths > 0.5) & (depths < self.config.range_m)
            valid_mask = hit_mask & range_mask

            # --- Monte Carlo Ray Drop ---
            n_rays = self._ray_dirs.shape[0]
            ray_drop_mask = torch.zeros(n_rays, dtype=torch.bool, device=self.device)

            if stochastic:
                if learned_ray_drop is not None:
                    drop_prob = learned_ray_drop.to(self.device)
                else:
                    drop_prob = torch.full(
                        (n_rays,), self.config.ray_drop_prob, device=self.device
                    )

                # Sample Bernoulli — ray is dropped if random < prob
                drop_sample = torch.rand(n_rays, device=self.device)
                ray_drop_mask = drop_sample < drop_prob
                valid_mask &= ~ray_drop_mask

            # Compute 3D points in world frame
            points_world = ray_origins_world + ray_dirs_world * depths.unsqueeze(-1)

            # Extract valid points
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            points_out = points_world[valid_indices].cpu().numpy()
            intensity_out = intensities[valid_indices].clamp(0, 1).cpu().numpy()
            depth_out = depths[valid_indices].cpu().numpy()

            # Compute ring (channel) index and azimuth for each point
            n_per_ch = self.config.points_per_channel
            ring_out = (valid_indices // n_per_ch).int().cpu().numpy()
            azimuth_out = (valid_indices % n_per_ch).float()
            azimuth_out = (
                azimuth_out / n_per_ch *
                np.radians(self.config.horizontal_fov[1] - self.config.horizontal_fov[0])
                + np.radians(self.config.horizontal_fov[0])
            )
            azimuth_out = azimuth_out.cpu().numpy()

        return {
            "points": points_out.astype(np.float32),
            "intensity": intensity_out.astype(np.float32),
            "depth": depth_out.astype(np.float32),
            "ray_drop_mask": ray_drop_mask.cpu().numpy(),
            "ring": ring_out.astype(np.int32),
            "azimuth": azimuth_out.astype(np.float32),
            "n_total_rays": n_rays,
            "n_valid_points": len(points_out),
        }

    def render_ray_drop_heatmap(
        self,
        ego_pose_matrix: np.ndarray,
        n_samples: int = 50,
    ) -> np.ndarray:
        """
        Estimate ray drop probability per ray via Monte Carlo sampling.

        Fires the same rays N times with stochastic dropping and computes
        the empirical drop rate. Returns a (channels, points_per_channel)
        heatmap of ray drop frequency.

        Args:
            ego_pose_matrix: 4x4 ego pose.
            n_samples: Number of Monte Carlo samples.

        Returns:
            heatmap: (channels, points_per_channel) float32 in [0, 1].
        """
        cfg = self.config
        n_rays = cfg.channels * cfg.points_per_channel
        drop_counts = np.zeros(n_rays, dtype=np.float32)

        for _ in range(n_samples):
            result = self.render(ego_pose_matrix, stochastic=True)
            drop_counts += result["ray_drop_mask"].astype(np.float32)

        heatmap = (drop_counts / n_samples).reshape(cfg.channels, cfg.points_per_channel)
        return heatmap

    @staticmethod
    def save_pcd(filepath: str, points: np.ndarray, intensity: np.ndarray):
        """
        Save point cloud in PCD format (ASCII).

        Args:
            filepath: Output .pcd file path.
            points: (N, 3) xyz coordinates.
            intensity: (N,) intensity values.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        N = points.shape[0]

        with open(path, "w") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {N}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {N}\n")
            f.write("DATA ascii\n")
            for i in range(N):
                f.write(
                    f"{points[i, 0]:.6f} {points[i, 1]:.6f} "
                    f"{points[i, 2]:.6f} {intensity[i]:.4f}\n"
                )

    @staticmethod
    def save_kitti_bin(filepath: str, points: np.ndarray, intensity: np.ndarray):
        """
        Save point cloud in KITTI .bin format (binary float32).

        Format: N x 4 float32 (x, y, z, intensity).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([points, intensity]).astype(np.float32)
        data.tofile(str(path))

    def __repr__(self) -> str:
        cfg = self.config
        n_rays = cfg.channels * cfg.points_per_channel
        return (
            f"VirtualLiDAR('{cfg.name}', {cfg.channels}ch, "
            f"{n_rays} rays, range={cfg.range_m}m)"
        )
