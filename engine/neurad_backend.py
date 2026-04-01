"""
NeuRAD/SplatAD backend for high-fidelity rendering.

Uses the trained neurad-studio model with CUDA-accelerated gsplat rasterization
and learned CNN/MLP decoders for camera RGB and LiDAR rendering.

This replaces the pure-Python splatting renderer when a real trained checkpoint
is available, providing:
  - Real-time CUDA rasterization (~10ms per frame on RTX 4090)
  - Learned RGB decoder (CNN) for view-dependent appearance
  - Learned LiDAR decoder (MLP) for intensity and ray-drop probability
"""

import sys
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Optional

# OpenCV <-> nerfstudio convention transform
OPENCV_TO_NERFSTUDIO = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
NERFSTUDIO_TO_OPENCV = OPENCV_TO_NERFSTUDIO  # self-inverse


class NeuradBackend:
    """
    Wraps a trained neurad-studio SplatAD model for rendering.

    Handles coordinate convention differences between our engine (OpenCV)
    and nerfstudio's internal convention.
    """

    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = device
        self._load_model(config_path)

    def _load_model(self, config_path: str):
        """Load the trained model and dataparser outputs."""
        # Add neurad-studio to path
        neurad_path = str(Path(__file__).parent.parent.parent / "SplatAD" / "neurad-studio")
        if neurad_path not in sys.path:
            sys.path.insert(0, neurad_path)

        from nerfstudio.utils.eval_utils import eval_setup

        config_path = Path(config_path)
        _, self.pipeline, _, _ = eval_setup(
            config_path, eval_num_rays_per_chunk=None, test_mode="test"
        )
        self.model = self.pipeline.model
        self.model.eval()

        # Cache the dataparser cameras for coordinate reference
        self._eval_cameras = self.pipeline.datamanager.eval_dataset.cameras
        self._train_cameras = self.pipeline.datamanager.train_dataset.cameras

        # Get dataparser transform for converting external poses to scene frame
        dp_outputs = self.pipeline.datamanager.train_dataparser_outputs
        self.dataparser_transform = dp_outputs.dataparser_transform  # (3, 4)
        self.dataparser_scale = dp_outputs.dataparser_scale

        # Build 4x4 world-to-scene transform
        self._w2s = torch.eye(4, dtype=torch.float32)
        self._w2s[:3, :] = self.dataparser_transform
        self._w2s = self._w2s.numpy()

        # Cache LiDAR dataset if available
        dm = self.pipeline.datamanager
        self._has_lidar = hasattr(dm, 'train_lidar_dataset')
        if self._has_lidar:
            self._train_lidar_dataset = dm.train_lidar_dataset
            self._train_lidars = deepcopy(self._train_lidar_dataset.lidars)
            self._cached_lidar_train = dm.cached_lidar_train

    @property
    def n_gaussians(self):
        return self.model.num_points

    def render_camera(
        self,
        camera_to_world_opencv: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        timestamp: float = 0.0,
    ) -> dict:
        """
        Render RGB + depth from an arbitrary camera pose.

        Args:
            camera_to_world_opencv: 4x4 camera-to-world in OpenCV convention
                                    (x-right, y-down, z-forward) in the
                                    ORIGINAL world frame (pre-dataparser-transform).
            fx, fy, cx, cy: Camera intrinsics.
            width, height: Image resolution.
            timestamp: Frame timestamp (for dynamic objects).

        Returns:
            dict with 'rgb' (H,W,3 uint8), 'depth' (H,W float32),
            'alpha' (H,W float32).
        """
        from nerfstudio.cameras.cameras import Cameras, CameraType

        # Convert OpenCV cam2world to nerfstudio convention
        T_cv = np.array(camera_to_world_opencv, dtype=np.float64)
        T_ns = T_cv.copy()
        T_ns[:3, :3] = T_cv[:3, :3] @ OPENCV_TO_NERFSTUDIO

        # Apply dataparser transform (world -> scene)
        T_scene = self._w2s @ T_ns
        T_scene_ns = T_scene.copy()
        # The rotation was already in nerfstudio convention

        c2w = torch.tensor(T_scene_ns[:3, :4], dtype=torch.float32).unsqueeze(0)

        camera = Cameras(
            camera_to_worlds=c2w,
            fx=torch.tensor([fx], dtype=torch.float32),
            fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([cx], dtype=torch.float32),
            cy=torch.tensor([cy], dtype=torch.float32),
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor([timestamp], dtype=torch.float64),
            metadata={"sensor_idxs": torch.tensor([[0]], dtype=torch.int32)},
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        depth = outputs["depth"].cpu().numpy().squeeze()
        alpha = outputs["accumulation"].cpu().numpy().squeeze()

        return {"rgb": rgb, "depth": depth, "alpha": alpha}

    def render_train_camera(self, frame_idx: int) -> dict:
        """Render using the exact training camera pose (for V&V comparison)."""
        camera = self._train_cameras[frame_idx : frame_idx + 1].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        depth = outputs["depth"].cpu().numpy().squeeze()
        alpha = outputs["accumulation"].cpu().numpy().squeeze()

        return {"rgb": rgb, "depth": depth, "alpha": alpha}

    def render_eval_camera(self, frame_idx: int) -> dict:
        """Render using an eval camera pose."""
        camera = self._eval_cameras[frame_idx : frame_idx + 1].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        depth = outputs["depth"].cpu().numpy().squeeze()
        alpha = outputs["accumulation"].cpu().numpy().squeeze()

        return {"rgb": rgb, "depth": depth, "alpha": alpha}

    def render_train_lidar(self, lidar_idx: int) -> dict:
        """
        Render LiDAR using the trained model's MLP decoder.

        Uses the exact training LiDAR pose and point cloud layout (raster_pts)
        to produce depth, intensity, and ray-drop predictions via gsplat's
        lidar_rasterization + learned MLP decoder.

        Args:
            lidar_idx: Index into the training LiDAR dataset (0 to N-1).

        Returns:
            dict with:
                'points' (M,3) float32 — 3D points in LiDAR-local frame
                'intensity' (M,) float32 — predicted reflectance [0,1]
                'depth' (H,W) float32 — depth image in azimuth-elevation space
                'ray_drop_prob' (H,W) float32 — probability each ray didn't return
                'points_world' (M,3) float32 — 3D points in scene world frame
        """
        assert self._has_lidar, "No LiDAR dataset available in this checkpoint"

        dm = self.pipeline.datamanager
        lidar = self._train_lidars[lidar_idx : lidar_idx + 1].to(self.device)
        if lidar.metadata is None:
            lidar.metadata = {}
        lidar.metadata["lidar_idx"] = lidar_idx

        data = self._cached_lidar_train[lidar_idx].copy()
        dm._add_metadata(lidar, data, len(dm.train_dataset))

        with torch.no_grad():
            outputs = self.model.get_lidar_outputs(lidar)

        depth = outputs["depth"].cpu().numpy().squeeze()
        ray_drop_prob = outputs["ray_drop_prob"].cpu().numpy().squeeze()
        intensity = outputs["intensity"].cpu().numpy().squeeze()
        accumulation = outputs["accumulation"].cpu().numpy().squeeze()

        # Reconstruct 3D points from the raster_pts (azimuth, elevation) + predicted depth
        raster_pts = lidar.metadata["raster_pts"]  # (1, H, W, 4+) — azim, elev, depth, time
        azim_deg = raster_pts[0, :, :, 0].cpu().numpy()
        elev_deg = raster_pts[0, :, :, 1].cpu().numpy()
        azim_rad = np.deg2rad(azim_deg)
        elev_rad = np.deg2rad(elev_deg)

        pred_depth = outputs["depth"][0, :, :, 0].cpu().numpy()
        pred_drop = outputs["ray_drop_prob"][0].cpu().numpy().squeeze()

        # Spherical to Cartesian (LiDAR-local frame)
        x = pred_depth * np.cos(azim_rad) * np.cos(elev_rad)
        y = pred_depth * np.sin(azim_rad) * np.cos(elev_rad)
        z = pred_depth * np.sin(elev_rad)

        # Filter: keep rays that returned (drop prob < 0.5) and have valid depth
        valid = (pred_drop < 0.5) & (pred_depth > 0.2) & (accumulation.squeeze() > 0.1)
        points_local = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
        intensities = intensity.squeeze()[valid].astype(np.float32)

        # Transform to world frame
        l2w = lidar.lidar_to_worlds[0].cpu().numpy()  # (3, 4)
        R, t = l2w[:3, :3], l2w[:3, 3]
        points_world = (points_local @ R.T + t).astype(np.float32)

        return {
            "points": points_local,
            "points_world": points_world,
            "intensity": intensities,
            "depth": depth,
            "ray_drop_prob": ray_drop_prob,
        }

    def get_train_lidar_count(self) -> int:
        """Number of training LiDAR scans."""
        if not self._has_lidar:
            return 0
        return len(self._train_lidar_dataset)

    def get_train_camera_count(self) -> int:
        """Number of training camera frames (all sensors combined)."""
        return len(self._train_cameras)

    def get_eval_camera_count(self) -> int:
        """Number of eval camera frames."""
        return len(self._eval_cameras)
