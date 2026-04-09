"""
Virtual Camera Renderer for Novel View Synthesis.

Renders RGB + depth images from arbitrary camera poses by rasterizing
a pre-trained 3D Gaussian Splatting scene. Supports configurable intrinsics,
extrinsics, and resolution for simulating different camera hardware.

The renderer computes:
    T_world_camera = T_world_vehicle @ T_vehicle_sensor
and passes the composed transform to the 3DGS CUDA rasterizer.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float] = field(default_factory=lambda: [0.0] * 5)

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def fov_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * np.arctan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * np.arctan(self.height / (2.0 * self.fy))


def extrinsics_to_matrix(ext: dict, sensor_type: str = "lidar") -> np.ndarray:
    """
    Convert extrinsics dict {x, y, z, roll, pitch, yaw} to 4x4 SE(3) matrix.
    Angles are in degrees. Convention: T_vehicle_sensor.

    For cameras, includes a base rotation to align the camera coordinate
    convention (x-right, y-down, z-forward / OpenCV) with the vehicle
    convention (x-forward, y-left, z-up):
        camera_x = -vehicle_y  (right)
        camera_y = -vehicle_z  (down)
        camera_z = +vehicle_x  (forward)
    """
    R_user = Rotation.from_euler(
        "xyz",
        [ext["roll"], ext["pitch"], ext["yaw"]],
        degrees=True,
    ).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [ext["x"], ext["y"], ext["z"]]

    if sensor_type == "camera":
        # Base rotation: vehicle_frame -> camera_frame (OpenCV)
        # camera_x = -vehicle_y, camera_y = -vehicle_z, camera_z = vehicle_x
        # So T_vehicle_camera columns are what camera axes are in vehicle frame:
        # vehicle = T_vehicle_camera @ camera
        # col0 (camera_x in vehicle): [0, -1, 0] (camera right = -vehicle left)
        # col1 (camera_y in vehicle): [0, 0, -1] (camera down = -vehicle up)
        # col2 (camera_z in vehicle): [1, 0, 0]  (camera forward = vehicle forward)
        R_base = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=np.float64)
        T[:3, :3] = R_user @ R_base
    else:
        T[:3, :3] = R_user

    return T


class GaussianSplatScene:
    """
    Interface to a trained 3D Gaussian Splatting scene.

    Wraps the scene's Gaussian parameters (means, covariances, opacities,
    SH coefficients) and provides a rasterization method.

    In production, this loads from a trained SplatAD checkpoint.
    For standalone testing, it generates a synthetic scene.
    """

    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        sh_degree: int = 3,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.means = means.to(self.device)          # (N, 3)
        self.scales = scales.to(self.device)         # (N, 3)
        self.rotations = rotations.to(self.device)   # (N, 4) quaternions
        self.opacities = opacities.to(self.device)   # (N, 1)
        self.sh_coeffs = sh_coeffs.to(self.device)   # (N, C, 3)
        self.sh_degree = sh_degree
        self.n_gaussians = means.shape[0]
        self.n_background = means.shape[0]  # All Gaussians are background by default
        self.actor_boxes = []  # List of (center, half_extents) for actor AABBs

    @classmethod
    def create_synthetic(
        cls,
        n_gaussians: int = 50000,
        scene_extent: float = 50.0,
        device: str = "cuda",
    ) -> "GaussianSplatScene":
        """
        Generate a synthetic Gaussian Splat scene for testing.

        Creates a ground plane, scattered objects, and a few structure-like
        clusters to simulate an urban driving environment.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        # Ground plane: thin, wide Gaussians spread along the driving corridor
        n_ground = n_gaussians // 3
        ground_means = torch.zeros(n_ground, 3)
        # Ground extends forward (positive x) and to the sides
        ground_means[:, 0] = torch.rand(n_ground) * scene_extent * 2 - scene_extent * 0.5
        ground_means[:, 1] = (torch.rand(n_ground) - 0.5) * scene_extent
        ground_means[:, 2] = torch.randn(n_ground) * 0.02 - 1.7  # Ground at z ~ -1.7 (below vehicle)

        # Road surface: denser strip along x-axis
        n_road = n_gaussians // 6
        road_means = torch.zeros(n_road, 3)
        road_means[:, 0] = torch.rand(n_road) * scene_extent * 1.5 + 2.0  # In front
        road_means[:, 1] = (torch.rand(n_road) - 0.5) * 8.0  # Road width ~8m
        road_means[:, 2] = torch.randn(n_road) * 0.01 - 1.7

        # Objects: clustered along the road corridor (vehicles, buildings, poles)
        n_objects = n_gaussians - n_ground - n_road
        object_means = torch.zeros(n_objects, 3)
        n_clusters = 25
        cluster_size = n_objects // n_clusters

        for i in range(n_clusters):
            start = i * cluster_size
            end = min(start + cluster_size, n_objects)
            # Place clusters along the road, at various distances
            cx = np.random.uniform(3.0, scene_extent * 1.5)  # Forward of ego
            cy = np.random.choice([-5.0, -3.0, 3.0, 5.0]) + np.random.randn() * 1.0  # Sides of road
            cz = np.random.uniform(-0.5, 2.5)  # Ground level to building height
            center = torch.tensor([cx, cy, cz])

            # Vary cluster shape: some tall (poles/trees), some wide (buildings)
            if i % 4 == 0:  # Tall objects
                spread = torch.tensor([0.3, 0.3, 1.5])
            elif i % 4 == 1:  # Wide objects (buildings)
                spread = torch.tensor([2.0, 2.0, 0.8])
            else:  # Vehicles/medium
                spread = torch.tensor([1.2, 0.8, 0.5])

            object_means[start:end] = center + torch.randn(end - start, 3) * spread

        means = torch.cat([ground_means, road_means, object_means], dim=0)

        # Scales: ground is flat, road is flat, objects vary
        ground_scales = torch.tensor([0.4, 0.4, 0.01]).expand(n_ground, -1) + \
                        torch.randn(n_ground, 3).abs() * 0.03
        road_scales = torch.tensor([0.3, 0.3, 0.005]).expand(n_road, -1) + \
                      torch.randn(n_road, 3).abs() * 0.02
        object_scales = torch.randn(n_objects, 3).abs() * 0.12 + 0.04
        scales = torch.cat([ground_scales, road_scales, object_scales], dim=0).log()

        # Rotations: identity + noise
        rotations = torch.zeros(n_gaussians, 4)
        rotations[:, 0] = 1.0  # w component = 1 (identity)
        rotations += torch.randn_like(rotations) * 0.05
        rotations = torch.nn.functional.normalize(rotations, dim=-1)

        # Opacities: mostly opaque
        opacities = torch.sigmoid(torch.randn(n_gaussians, 1) * 0.5 + 2.0)

        # SH coefficients: DC term for base color + higher order
        n_sh = (sh_degree := 3 + 1) ** 2
        sh_coeffs = torch.zeros(n_gaussians, n_sh, 3)
        # Ground: green-grey
        sh_coeffs[:n_ground, 0, :] = torch.tensor([0.3, 0.35, 0.25]) + \
                                       torch.randn(n_ground, 3) * 0.05
        # Road: dark grey asphalt
        sh_coeffs[n_ground:n_ground+n_road, 0, :] = torch.tensor([0.25, 0.25, 0.27]) + \
                                                      torch.randn(n_road, 3) * 0.03
        # Objects: varied colors
        sh_coeffs[n_ground+n_road:, 0, :] = torch.rand(n_objects, 3) * 0.6 + 0.2
        # Higher-order SH: small random perturbations
        sh_coeffs[:, 1:, :] = torch.randn(n_gaussians, n_sh - 1, 3) * 0.02

        return cls(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            sh_degree=3,
            device=device,
        )

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = "cuda") -> "GaussianSplatScene":
        """
        Load a trained 3DGS scene from a SplatAD/neurad-studio checkpoint.

        Supports two formats:
        1. neurad-studio checkpoint: nested under pipeline._model.gauss_params.*
        2. Simple dict with keys: means, scales, rotations, opacities, sh_coeffs
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # neurad-studio / SplatAD checkpoint format
        if "pipeline" in ckpt:
            pipe = ckpt["pipeline"]
            prefix = "_model.gauss_params."
            means = pipe[prefix + "means"]
            scales = pipe[prefix + "scales"]
            rotations = pipe[prefix + "quats"]
            opacities = torch.sigmoid(pipe[prefix + "opacities"])

            # SplatAD stores features_dc (N, 3) as the base color component.
            # It uses a learned CNN decoder for final RGB, which we don't replicate.
            # Treat features_dc as SH degree-0 coefficients.
            features_dc = pipe[prefix + "features_dc"]  # (N, 3)
            sh_coeffs = features_dc.unsqueeze(1)  # (N, 1, 3)

            return cls(
                means=means,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                sh_coeffs=sh_coeffs,
                sh_degree=0,
                device=device,
            )

        # Simple dict format
        if "gaussians" in ckpt:
            g = ckpt["gaussians"]
        else:
            g = ckpt

        return cls(
            means=g["means"],
            scales=g["scales"],
            rotations=g["rotations"],
            opacities=g["opacities"],
            sh_coeffs=g["sh_coeffs"],
            sh_degree=g.get("sh_degree", 3),
            device=device,
        )

    def merge_actors(self, actor_list) -> "GaussianSplatScene":
        """
        Merge transformed actor Gaussians with this background scene.

        Returns a new GaussianSplatScene with concatenated Gaussians.
        Does NOT modify the original scene.

        Args:
            actor_list: List of TransformedActor (from actor.py).

        Returns:
            New GaussianSplatScene with background + actor Gaussians.
        """
        if not actor_list:
            return self

        # Suppress background Gaussians inside actor bounding boxes.
        # Use generous margins so that rays approaching from any angle
        # encounter a clear volume around the actor.
        bg_opacities = self.opacities.clone()
        for actor in actor_list:
            center = torch.tensor(actor.center, dtype=torch.float32, device=self.device)
            half_ext = torch.tensor(actor.dimensions, dtype=torch.float32, device=self.device) / 2.0 + 2.0
            inside = ((self.means - center).abs() < half_ext).all(dim=1)
            bg_opacities[inside] = 0.0

        all_means = [self.means]
        all_scales = [self.scales]
        all_rotations = [self.rotations]
        all_opacities = [bg_opacities]
        all_sh = [self.sh_coeffs]

        for actor in actor_list:
            all_means.append(actor.means)
            all_scales.append(actor.scales)
            all_rotations.append(actor.rotations)
            all_opacities.append(actor.opacities)
            # Pad SH coefficients to match background if needed
            actor_sh = actor.sh_coeffs
            if actor_sh.shape[1] != self.sh_coeffs.shape[1]:
                n_actor = actor_sh.shape[0]
                n_sh_bg = self.sh_coeffs.shape[1]
                padded = torch.zeros(n_actor, n_sh_bg, 3, device=self.device)
                n_copy = min(actor_sh.shape[1], n_sh_bg)
                padded[:, :n_copy, :] = actor_sh[:, :n_copy, :]
                actor_sh = padded
            all_sh.append(actor_sh)

        merged = GaussianSplatScene(
            means=torch.cat(all_means, dim=0),
            scales=torch.cat(all_scales, dim=0),
            rotations=torch.cat(all_rotations, dim=0),
            opacities=torch.cat(all_opacities, dim=0),
            sh_coeffs=torch.cat(all_sh, dim=0),
            sh_degree=self.sh_degree,
            device=str(self.device),
        )
        merged.n_background = self.n_gaussians
        merged.actor_boxes = [
            (np.array(a.center), np.array(a.dimensions))
            for a in actor_list
        ]
        return merged


class VirtualCamera:
    """
    Virtual camera renderer using 3D Gaussian Splatting.

    Given an ego pose and sensor extrinsics, computes the camera-to-world
    transform and rasterizes the Gaussian scene to produce:
        - RGB image (H, W, 3) uint8
        - Depth map (H, W) float32 in meters
        - Alpha/accumulation map (H, W) float32
    """

    def __init__(
        self,
        name: str,
        intrinsics: CameraIntrinsics,
        T_vehicle_sensor: np.ndarray,
        scene: GaussianSplatScene,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.name = name
        self.intrinsics = intrinsics
        self.T_vehicle_sensor = T_vehicle_sensor
        self.scene = scene
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=scene.device)
        self.device = scene.device

    @classmethod
    def from_config(cls, cfg: dict, scene: GaussianSplatScene) -> "VirtualCamera":
        """Instantiate from a sensor config dict (parsed from YAML)."""
        intr = cfg["intrinsics"]
        res = cfg["resolution"]
        intrinsics = CameraIntrinsics(
            fx=intr["fx"], fy=intr["fy"],
            cx=intr["cx"], cy=intr["cy"],
            width=res[0], height=res[1],
            distortion=cfg.get("distortion", [0.0] * 5),
        )
        T_vs = extrinsics_to_matrix(cfg["extrinsics"], sensor_type="camera")
        return cls(
            name=cfg["name"],
            intrinsics=intrinsics,
            T_vehicle_sensor=T_vs,
            scene=scene,
        )

    def _compute_view_matrix(self, T_world_vehicle: np.ndarray) -> torch.Tensor:
        """
        Compute the world-to-camera (view) matrix.

        T_camera_world = (T_world_vehicle @ T_vehicle_sensor)^{-1}
        """
        T_world_camera = T_world_vehicle @ self.T_vehicle_sensor
        # Invert: camera_from_world
        R = T_world_camera[:3, :3]
        t = T_world_camera[:3, 3]
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R.T
        T_cam_world[:3, 3] = -R.T @ t
        return torch.tensor(T_cam_world, dtype=torch.float32, device=self.device)

    def _compute_projection_matrix(self) -> torch.Tensor:
        """Compute OpenGL-style projection matrix from intrinsics."""
        K = self.intrinsics
        near, far = 0.1, 200.0
        w, h = K.width, K.height

        # Normalized device coordinates projection
        proj = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj[0, 0] = 2.0 * K.fx / w
        proj[1, 1] = 2.0 * K.fy / h
        proj[0, 2] = 1.0 - 2.0 * K.cx / w
        proj[1, 2] = 2.0 * K.cy / h - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0
        return proj

    def render(
        self,
        ego_pose_matrix: np.ndarray,
        return_depth: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Render the scene from a given ego pose.

        This implements a differentiable Gaussian splatting rasterizer.
        For each pixel, we:
          1. Project all Gaussian means into camera coordinates
          2. Compute 2D covariance via the Jacobian of the projection
          3. Alpha-composite front-to-back by depth order

        Args:
            ego_pose_matrix: 4x4 T_world_vehicle transform.
            return_depth: Whether to compute and return depth map.

        Returns:
            dict with keys: 'rgb' (H,W,3 uint8), 'depth' (H,W float32),
            'alpha' (H,W float32), 'viewmat' (4x4 float32).
        """
        view_matrix = self._compute_view_matrix(ego_pose_matrix)
        proj_matrix = self._compute_projection_matrix()
        W, H = self.intrinsics.width, self.intrinsics.height

        with torch.no_grad():
            # --- Step 1: Transform Gaussians to camera space ---
            means_world = self.scene.means  # (N, 3)
            R_cw = view_matrix[:3, :3]  # camera_from_world rotation
            t_cw = view_matrix[:3, 3]   # camera_from_world translation

            means_cam = (R_cw @ means_world.T).T + t_cw  # (N, 3) in camera frame

            # Frustum culling: keep Gaussians in front of camera and within range
            depths = means_cam[:, 2]
            valid_mask = (depths > 0.1) & (depths < 200.0)

            # Additional culling: check if projected center is near the image
            px = self.intrinsics.fx * means_cam[:, 0] / depths + self.intrinsics.cx
            py = self.intrinsics.fy * means_cam[:, 1] / depths + self.intrinsics.cy
            margin = 100  # pixels
            valid_mask &= (px > -margin) & (px < W + margin) & \
                          (py > -margin) & (py < H + margin)

            if valid_mask.sum() == 0:
                return {
                    "rgb": np.zeros((H, W, 3), dtype=np.uint8),
                    "depth": np.full((H, W), np.inf, dtype=np.float32),
                    "alpha": np.zeros((H, W), dtype=np.float32),
                    "viewmat": view_matrix.cpu().numpy(),
                }

            # Filter to valid Gaussians (frustum + non-zero opacity)
            valid_mask &= self.scene.opacities[:, 0] > 1e-3
            means_c = means_cam[valid_mask]  # (M, 3)
            depths_v = depths[valid_mask]
            scales_v = torch.exp(self.scene.scales[valid_mask])  # (M, 3)
            rots_v = self.scene.rotations[valid_mask]             # (M, 4) quaternion
            opac_v = self.scene.opacities[valid_mask]             # (M, 1)
            sh_v = self.scene.sh_coeffs[valid_mask]               # (M, C, 3)

            # --- Step 2: Compute per-Gaussian 3D covariance in world, project to 2D ---
            # Build rotation matrices from quaternions
            qw, qx, qy, qz = rots_v[:, 0], rots_v[:, 1], rots_v[:, 2], rots_v[:, 3]
            R_gauss = torch.stack([
                1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy),
                2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx),
                2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2),
            ], dim=-1).reshape(-1, 3, 3)  # (M, 3, 3)

            # 3D covariance: Sigma = R @ S @ S^T @ R^T
            S = torch.diag_embed(scales_v)  # (M, 3, 3)
            RS = R_gauss @ S
            cov3d = RS @ RS.transpose(-1, -2)  # (M, 3, 3)

            # Transform covariance to camera frame
            R_cw_batch = R_cw.unsqueeze(0).expand(means_c.shape[0], -1, -1)
            cov3d_cam = R_cw_batch @ cov3d @ R_cw_batch.transpose(-1, -2)

            # Jacobian of perspective projection at each Gaussian center
            fx, fy = self.intrinsics.fx, self.intrinsics.fy
            tx = means_c[:, 0]
            ty = means_c[:, 1]
            tz = means_c[:, 2].clamp(min=0.1)
            J = torch.zeros(means_c.shape[0], 2, 3, device=self.device)
            J[:, 0, 0] = fx / tz
            J[:, 0, 2] = -fx * tx / (tz ** 2)
            J[:, 1, 1] = fy / tz
            J[:, 1, 2] = -fy * ty / (tz ** 2)

            # 2D covariance: cov2d = J @ cov3d_cam @ J^T
            cov2d = J @ cov3d_cam[:, :3, :3] @ J.transpose(-1, -2)  # (M, 2, 2)

            # Add low-pass filter (anti-aliasing)
            cov2d[:, 0, 0] += 0.3
            cov2d[:, 1, 1] += 0.3

            # --- Step 3: Evaluate SH for view-dependent color ---
            # Camera position in world frame
            T_wc = ego_pose_matrix @ self.T_vehicle_sensor
            cam_pos = torch.tensor(T_wc[:3, 3], dtype=torch.float32, device=self.device)
            means_w = self.scene.means[valid_mask]
            view_dirs = means_w - cam_pos
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

            # Evaluate degree-0 SH (DC component) for base color
            # C0 = 0.28209479177387814 (1 / (2*sqrt(pi)))
            C0 = 0.28209479177387814
            colors = C0 * sh_v[:, 0, :] + 0.5  # (M, 3) — shift to [0, 1] range

            # Add degree-1 SH if available
            if sh_v.shape[1] > 1:
                C1 = 0.4886025119029199
                colors = colors + C1 * (
                    -sh_v[:, 1, :] * view_dirs[:, 1:2] +
                    sh_v[:, 2, :] * view_dirs[:, 2:3] +
                    -sh_v[:, 3, :] * view_dirs[:, 0:1]
                )
            colors = colors.clamp(0.0, 1.0)

            # --- Step 4: Per-Gaussian splatting (vectorized) ---
            # Sort by depth (front-to-back), ensuring actors are always
            # included in the rendering budget regardless of MAX_GAUSS.
            sort_idx = torch.argsort(depths_v)
            M = means_c.shape[0]
            MAX_BG = 2000  # Background Gaussian budget

            # Build original-index mask for actors vs background.
            # valid_mask was applied to scene Gaussians; map back to
            # find which valid Gaussians are actors.
            valid_indices = torch.where(valid_mask)[0]
            is_actor_valid = valid_indices >= self.scene.n_background
            is_actor_sorted = is_actor_valid[sort_idx]

            # Select: top MAX_BG background Gaussians + all actor Gaussians
            bg_sorted = torch.where(~is_actor_sorted)[0]
            actor_sorted = torch.where(is_actor_sorted)[0]
            bg_keep = bg_sorted[:MAX_BG]
            selected = torch.cat([bg_keep, actor_sorted])
            # Re-sort the selection by depth
            selected = selected[torch.argsort(depths_v[sort_idx[selected]])]
            sort_idx = sort_idx[selected]

            means_c = means_c[sort_idx]
            depths_v = depths_v[sort_idx]
            cov2d = cov2d[sort_idx]
            opac_v = opac_v[sort_idx]
            colors = colors[sort_idx]

            # Project centers to pixel coordinates
            px_sorted = fx * means_c[:, 0] / means_c[:, 2] + self.intrinsics.cx
            py_sorted = fy * means_c[:, 1] / means_c[:, 2] + self.intrinsics.cy

            # Compute eigenvalues for bounding radius
            det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
            trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
            lambda_max = 0.5 * (trace + torch.sqrt((trace ** 2 - 4 * det).clamp(min=0.0)))
            radius = torch.ceil(3.0 * torch.sqrt(lambda_max.clamp(min=0.01)))

            # Compute inverse of 2D covariance for Gaussian evaluation
            inv_det = 1.0 / det.clamp(min=1e-6)
            cov2d_inv_00 = cov2d[:, 1, 1] * inv_det
            cov2d_inv_01 = -cov2d[:, 0, 1] * inv_det
            cov2d_inv_11 = cov2d[:, 0, 0] * inv_det

            # Initialize output buffers
            rgb_buffer = self.bg_color.unsqueeze(0).unsqueeze(0).expand(H, W, -1).clone()
            depth_buffer = torch.zeros(H, W, device=self.device)
            T_buffer = torch.ones(H, W, device=self.device)  # Transmittance

            # Per-Gaussian splatting: iterate over sorted Gaussians
            MAX_GAUSS = means_c.shape[0]

            for gi in range(MAX_GAUSS):
                cx_g = px_sorted[gi]
                cy_g = py_sorted[gi]
                r = int(radius[gi].item())
                if r < 1:
                    continue

                # Bounding box for this Gaussian's footprint
                x0 = max(0, int(cx_g.item()) - r)
                x1 = min(W, int(cx_g.item()) + r + 1)
                y0 = max(0, int(cy_g.item()) - r)
                y1 = min(H, int(cy_g.item()) + r + 1)
                if x0 >= x1 or y0 >= y1:
                    continue

                # Early termination: if max transmittance in region is tiny, skip
                T_region = T_buffer[y0:y1, x0:x1]
                if T_region.max() < 0.005:
                    continue

                # Pixel grid for this Gaussian's footprint
                pix_y = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
                pix_x = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
                gy, gx = torch.meshgrid(pix_y, pix_x, indexing="ij")

                # Displacement from Gaussian center
                dx = gx - cx_g
                dy = gy - cy_g

                # Mahalanobis distance using inverse covariance components
                maha = (cov2d_inv_00[gi] * dx * dx +
                        2.0 * cov2d_inv_01[gi] * dx * dy +
                        cov2d_inv_11[gi] * dy * dy)

                # Gaussian weight * opacity
                gauss_w = torch.exp(-0.5 * maha)
                alpha_g = (opac_v[gi, 0] * gauss_w).clamp(max=0.99)

                # Weight by current transmittance
                weight = alpha_g * T_region

                # Accumulate color and depth
                rgb_buffer[y0:y1, x0:x1] += weight.unsqueeze(-1) * colors[gi]
                depth_buffer[y0:y1, x0:x1] += weight * depths_v[gi]

                # Update transmittance
                T_buffer[y0:y1, x0:x1] = T_region * (1.0 - alpha_g)

            # Add background
            alpha_buffer = 1.0 - T_buffer
            rgb_buffer += T_buffer.unsqueeze(-1) * self.bg_color

        # Convert to output format
        rgb_np = (rgb_buffer.clamp(0, 1) * 255).byte().cpu().numpy()
        result = {
            "rgb": rgb_np,
            "alpha": alpha_buffer.cpu().numpy(),
            "viewmat": view_matrix.cpu().numpy(),
        }
        if return_depth:
            depth_np = depth_buffer.cpu().numpy()
            depth_np[alpha_buffer.cpu().numpy() < 0.01] = np.inf
            result["depth"] = depth_np

        return result

    def get_world_transform(self, ego_pose: np.ndarray) -> np.ndarray:
        """Get the full camera-to-world transform for a given ego pose."""
        return ego_pose @ self.T_vehicle_sensor

    def __repr__(self) -> str:
        K = self.intrinsics
        return (
            f"VirtualCamera('{self.name}', {K.width}x{K.height}, "
            f"fx={K.fx:.1f}, fy={K.fy:.1f})"
        )
