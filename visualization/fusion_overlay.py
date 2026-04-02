"""
Sensor Fusion Visualization: LiDAR-Camera Overlay.

Projects synthetic 3D LiDAR points onto the synthetic 2D camera image,
colored by depth. This is the canonical "Hello World" of sensor fusion
calibration — proving spatial alignment between virtual sensors.

Also generates:
  - Ray drop probability heatmaps
  - Depth error maps
  - Bird's eye view (BEV) point cloud visualizations
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional


class FusionOverlay:
    """
    Project LiDAR points onto camera images for sensor fusion visualization.

    Validates that the virtual camera and LiDAR share consistent spatial
    calibration by overlaying projected LiDAR depth onto the rendered image.
    """

    def __init__(
        self,
        colormap: str = "turbo",
        point_size: float = 2.0,
        depth_range: tuple[float, float] = (0.5, 80.0),
        dpi: int = 150,
    ):
        self.colormap = colormap
        self.point_size = point_size
        self.depth_range = depth_range
        self.dpi = dpi

    def overlay_lidar_on_image(
        self,
        rgb_image: np.ndarray,
        lidar_points: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_lidar: np.ndarray,
        lidar_depths: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Project LiDAR points onto a camera image, colored by depth.

        Args:
            rgb_image: (H, W, 3) uint8 camera image.
            lidar_points: (N, 3) LiDAR points in world frame.
            intrinsics: (3, 3) camera intrinsic matrix K.
            T_cam_lidar: (4, 4) transform: camera <- world.
            lidar_depths: Optional (N,) depth values for coloring.
                         If None, computed from camera z-coordinate.
            output_path: If set, save the overlay image.

        Returns:
            (H, W, 3) uint8 overlay image.
        """
        H, W = rgb_image.shape[:2]

        # Transform LiDAR points to camera frame
        pts_hom = np.column_stack([lidar_points, np.ones(len(lidar_points))])
        pts_cam = (T_cam_lidar @ pts_hom.T).T[:, :3]

        # Filter: keep only points in front of camera
        mask = pts_cam[:, 2] > self.depth_range[0]
        pts_cam = pts_cam[mask]

        if lidar_depths is not None:
            depths = lidar_depths[mask]
        else:
            depths = pts_cam[:, 2]

        if len(pts_cam) == 0:
            return rgb_image.copy()

        # Project to pixel coordinates
        px = intrinsics[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + intrinsics[1, 2]

        # Filter to image bounds
        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[in_bounds]
        py = py[in_bounds]
        depths = depths[in_bounds]

        # Depth range filtering
        depth_mask = (depths > self.depth_range[0]) & (depths < self.depth_range[1])
        px = px[depth_mask]
        py = py[depth_mask]
        depths = depths[depth_mask]

        if len(depths) == 0:
            return rgb_image.copy()

        # Color by depth using colormap
        norm = Normalize(vmin=self.depth_range[0], vmax=self.depth_range[1])
        cmap = cm.get_cmap(self.colormap)
        colors = cmap(norm(depths))[:, :3]  # (N, 3) RGB in [0, 1]

        # Create overlay
        fig, ax = plt.subplots(1, 1, figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi)
        ax.imshow(rgb_image)
        scatter = ax.scatter(
            px, py,
            c=depths,
            cmap=self.colormap,
            s=self.point_size,
            vmin=self.depth_range[0],
            vmax=self.depth_range[1],
            alpha=0.8,
            edgecolors="none",
        )
        plt.colorbar(scatter, ax=ax, label="Depth (m)", shrink=0.7)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        ax.set_title("LiDAR → Camera Fusion Overlay", fontsize=10)
        fig.tight_layout(pad=0.5)

        # Render to numpy array
        fig.canvas.draw()
        overlay = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(overlay).save(output_path)

        return overlay

    def overlay_radar_on_image(
        self,
        rgb_image: np.ndarray,
        radar_points: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_radar: np.ndarray,
        radar_rcs: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Project radar detections onto a camera image, colored by RCS.

        Uses distinct diamond markers (larger than LiDAR dots) to
        differentiate radar from LiDAR in combined visualizations.

        Args:
            rgb_image: (H, W, 3) uint8 camera image.
            radar_points: (N, 3) radar points in world frame.
            intrinsics: (3, 3) camera intrinsic matrix K.
            T_cam_radar: (4, 4) transform: camera <- world.
            radar_rcs: Optional (N,) RCS values for coloring.
            output_path: If set, save the overlay image.

        Returns:
            (H, W, 3) uint8 overlay image.
        """
        H, W = rgb_image.shape[:2]

        if len(radar_points) == 0:
            return rgb_image.copy()

        # Transform radar points to camera frame
        pts_hom = np.column_stack([radar_points, np.ones(len(radar_points))])
        pts_cam = (T_cam_radar @ pts_hom.T).T[:, :3]

        # Filter: keep only points in front of camera
        mask = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[mask]

        if radar_rcs is not None:
            rcs = radar_rcs[mask]
        else:
            rcs = np.ones(len(pts_cam))

        if len(pts_cam) == 0:
            return rgb_image.copy()

        # Project to pixel coordinates
        px = intrinsics[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + intrinsics[1, 2]

        # Filter to image bounds
        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[in_bounds]
        py = py[in_bounds]
        rcs = rcs[in_bounds]

        if len(rcs) == 0:
            return rgb_image.copy()

        # Create overlay
        fig, ax = plt.subplots(1, 1, figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi)
        ax.imshow(rgb_image)
        scatter = ax.scatter(
            px, py,
            c=rcs,
            cmap="plasma",
            s=self.point_size * 20,
            marker="D",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="RCS", shrink=0.7)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        ax.set_title("Radar → Camera Fusion Overlay", fontsize=10)
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        overlay = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(overlay).save(output_path)

        return overlay

    def overlay_radar_velocity_on_image(
        self,
        rgb_image: np.ndarray,
        radar_points: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_world: np.ndarray,
        radial_velocity: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Project radar detections onto a camera image, colored by radial velocity.

        Uses a diverging colormap (RdBu_r): red = approaching, blue = receding,
        white = near-zero relative velocity.

        Args:
            rgb_image: (H, W, 3) uint8 camera image.
            radar_points: (N, 3) radar points in world frame.
            intrinsics: (3, 3) camera intrinsic matrix K.
            T_cam_world: (4, 4) transform: camera <- world.
            radial_velocity: (N,) radial velocity values [m/s].
            output_path: If set, save the overlay image.

        Returns:
            (H, W, 3) uint8 overlay image.
        """
        H, W = rgb_image.shape[:2]

        if len(radar_points) == 0:
            return rgb_image.copy()

        pts_hom = np.column_stack([radar_points, np.ones(len(radar_points))])
        pts_cam = (T_cam_world @ pts_hom.T).T[:, :3]

        mask = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[mask]
        vel = radial_velocity[mask]

        if len(pts_cam) == 0:
            return rgb_image.copy()

        px = intrinsics[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + intrinsics[1, 2]

        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[in_bounds]
        py = py[in_bounds]
        vel = vel[in_bounds]

        if len(vel) == 0:
            return rgb_image.copy()

        # Symmetric velocity range for diverging colormap
        v_max = max(np.abs(vel).max(), 1.0)

        fig, ax = plt.subplots(1, 1, figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi)
        ax.imshow(rgb_image)
        scatter = ax.scatter(
            px, py,
            c=vel,
            cmap="RdBu_r",
            vmin=-v_max, vmax=v_max,
            s=self.point_size * 20,
            marker="D",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Radial Velocity (m/s)", shrink=0.7)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        ax.set_title("Radar Doppler → Camera Overlay", fontsize=10)
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        overlay = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(overlay).save(output_path)

        return overlay

    def render_bev_velocity(
        self,
        lidar_points: np.ndarray,
        lidar_intensity: Optional[np.ndarray] = None,
        radar_points: Optional[np.ndarray] = None,
        radial_velocity: Optional[np.ndarray] = None,
        x_range: tuple[float, float] = (-40, 40),
        y_range: tuple[float, float] = (-40, 40),
        ego_position: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        BEV with radar detections colored by radial velocity.

        LiDAR shown as faint background, radar as diamonds colored by
        Doppler velocity (RdBu_r: red = approaching, blue = receding).

        Args:
            lidar_points: (N, 3) LiDAR points.
            lidar_intensity: Optional (N,) LiDAR intensity.
            radar_points: (N, 3) radar detections.
            radial_velocity: (N,) radial velocity [m/s].
            x_range: BEV x-axis range.
            y_range: BEV y-axis range.
            ego_position: (3,) ego position to center view.
            output_path: Optional save path.

        Returns:
            Rendered BEV image as numpy array.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=self.dpi)

        # LiDAR background
        pts = lidar_points.copy()
        if ego_position is not None:
            pts = pts - ego_position
        l_mask = (
            (pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) &
            (pts[:, 1] > y_range[0]) & (pts[:, 1] < y_range[1])
        )
        l_colors = lidar_intensity[l_mask] if lidar_intensity is not None else pts[l_mask, 2]
        ax.scatter(
            pts[l_mask, 0], pts[l_mask, 1],
            c=l_colors, cmap="gray", s=0.3, alpha=0.3, edgecolors="none",
        )

        # Ego vehicle
        ego_rect = plt.Rectangle((-2, -1), 4, 2, fill=True,
                                  facecolor="red", alpha=0.5, edgecolor="red")
        ax.add_patch(ego_rect)
        ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)

        # Radar colored by velocity
        if radar_points is not None and len(radar_points) > 0 and radial_velocity is not None:
            r_pts = radar_points.copy()
            if ego_position is not None:
                r_pts = r_pts - ego_position
            r_mask = (
                (np.abs(r_pts[:, 0]) < x_range[1]) &
                (np.abs(r_pts[:, 1]) < y_range[1])
            )
            if r_mask.sum() > 0:
                vel = radial_velocity[r_mask]
                v_max = max(np.abs(vel).max(), 1.0)
                sc = ax.scatter(
                    r_pts[r_mask, 0], r_pts[r_mask, 1],
                    c=vel, cmap="RdBu_r", vmin=-v_max, vmax=v_max,
                    s=30, marker="D", alpha=0.9,
                    edgecolors="white", linewidths=0.5,
                )
                plt.colorbar(sc, ax=ax, label="Radial Velocity (m/s)", shrink=0.7)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X (forward) [m]")
        ax.set_ylabel("Y (left) [m]")
        ax.set_title("BEV - Radar Doppler Velocity", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(result).save(output_path)

        return result

    def render_depth_comparison(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Side-by-side depth map comparison with error visualization.

        Args:
            pred_depth: (H, W) predicted depth map.
            gt_depth: (H, W) ground-truth depth map.
            output_path: Optional save path.

        Returns:
            Rendered comparison image as numpy array.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=self.dpi)

        vmin, vmax = self.depth_range

        # Predicted depth
        im0 = axes[0].imshow(pred_depth, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[0].set_title("Predicted Depth", fontsize=10)
        plt.colorbar(im0, ax=axes[0], shrink=0.7)

        # Ground truth depth
        valid_gt = gt_depth.copy()
        valid_gt[~np.isfinite(valid_gt)] = 0
        im1 = axes[1].imshow(valid_gt, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[1].set_title("Ground Truth Depth", fontsize=10)
        plt.colorbar(im1, ax=axes[1], shrink=0.7)

        # Absolute error
        error = np.abs(pred_depth - gt_depth)
        error[~np.isfinite(error)] = 0
        im2 = axes[2].imshow(error, cmap="hot", vmin=0, vmax=5.0)
        axes[2].set_title("Absolute Error (m)", fontsize=10)
        plt.colorbar(im2, ax=axes[2], shrink=0.7)

        for ax in axes:
            ax.set_axis_off()

        fig.suptitle("Depth Map V&V Comparison", fontsize=12, fontweight="bold")
        fig.tight_layout()

        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(result).save(output_path)

        return result

    def render_ray_drop_heatmap(
        self,
        heatmap: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize ray drop probability as a polar/rectangular heatmap.

        Args:
            heatmap: (channels, points_per_channel) ray drop probability.
            output_path: Optional save path.

        Returns:
            Rendered heatmap as numpy array.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=self.dpi)

        # Rectangular heatmap
        im0 = axes[0].imshow(
            heatmap, cmap="hot", vmin=0, vmax=1.0,
            aspect="auto", interpolation="bilinear",
        )
        axes[0].set_xlabel("Azimuth Index")
        axes[0].set_ylabel("Channel (Elevation)")
        axes[0].set_title("Ray Drop Probability (Rect.)")
        plt.colorbar(im0, ax=axes[0], shrink=0.7, label="P(drop)")

        # Polar projection
        channels, pts_per_ch = heatmap.shape
        az = np.linspace(0, 2 * np.pi, pts_per_ch, endpoint=False)
        el = np.linspace(-24.9, 2.0, channels)
        AZ, EL = np.meshgrid(az, el)
        R = 90.0 - EL  # Map elevation to radius

        im1 = axes[1].pcolormesh(
            AZ * 180 / np.pi, R, heatmap,
            cmap="hot", vmin=0, vmax=1.0, shading="auto",
        )
        axes[1].set_xlabel("Azimuth (deg)")
        axes[1].set_ylabel("90 - Elevation (deg)")
        axes[1].set_title("Ray Drop Probability (Polar Proj.)")
        plt.colorbar(im1, ax=axes[1], shrink=0.7, label="P(drop)")

        fig.suptitle(
            "Monte Carlo Ray Drop Analysis",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()

        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(result).save(output_path)

        return result

    def render_bev(
        self,
        points: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        x_range: tuple[float, float] = (-40, 40),
        y_range: tuple[float, float] = (-40, 40),
        resolution: float = 0.1,
        ego_position: Optional[np.ndarray] = None,
        radar_points: Optional[np.ndarray] = None,
        radar_rcs: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render a bird's-eye view (BEV) of the LiDAR point cloud.

        Args:
            points: (N, 3) 3D points.
            intensity: Optional (N,) intensity for coloring.
            x_range: (min, max) x-axis range in meters.
            y_range: (min, max) y-axis range in meters.
            resolution: Grid cell size in meters.
            ego_position: Optional (3,) ego position to center the view.
            radar_points: Optional (N, 3) radar detections to overlay.
            radar_rcs: Optional (N,) RCS values for radar coloring.
            output_path: Optional save path.

        Returns:
            Rendered BEV image as numpy array.
        """
        if ego_position is not None:
            points = points - ego_position

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=self.dpi)

        # Filter to range
        mask = (
            (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) &
            (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
        )
        pts = points[mask]

        if intensity is not None:
            colors = intensity[mask]
        else:
            colors = pts[:, 2]  # Color by height

        scatter = ax.scatter(
            pts[:, 0], pts[:, 1],
            c=colors, cmap=self.colormap, s=0.5,
            alpha=0.6, edgecolors="none",
        )
        plt.colorbar(scatter, ax=ax, shrink=0.7,
                      label="Intensity" if intensity is not None else "Height (m)")

        # Draw ego vehicle
        ego_rect = plt.Rectangle((-2, -1), 4, 2, fill=True,
                                  facecolor="red", alpha=0.5, edgecolor="red")
        ax.add_patch(ego_rect)
        ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)

        # Overlay radar detections
        if radar_points is not None and len(radar_points) > 0:
            r_pts = radar_points
            if ego_position is not None:
                r_pts = r_pts - ego_position
            r_mask = (
                (np.abs(r_pts[:, 0]) < x_range[1]) &
                (np.abs(r_pts[:, 1]) < y_range[1])
            )
            r_colors = radar_rcs[r_mask] if radar_rcs is not None else r_pts[r_mask, 2]
            ax.scatter(
                r_pts[r_mask, 0], r_pts[r_mask, 1],
                c=r_colors, cmap="plasma", s=25,
                marker="D", alpha=0.9, edgecolors="white", linewidths=0.5,
                label="Radar",
            )
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X (forward) [m]")
        ax.set_ylabel("Y (left) [m]")
        ax.set_title("Bird's Eye View - LiDAR + Radar", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(result).save(output_path)

        return result

    def render_multi_panel(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        lidar_points: np.ndarray,
        lidar_intensity: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_world: np.ndarray,
        ego_position: np.ndarray,
        radar_points: Optional[np.ndarray] = None,
        radar_rcs: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a comprehensive 4-panel visualization:
          1. Camera RGB
          2. Depth map
          3. LiDAR + Radar → Camera overlay
          4. BEV point cloud (LiDAR + Radar)

        Args:
            rgb_image: (H, W, 3) camera image.
            depth_map: (H, W) depth in meters.
            lidar_points: (N, 3) world-frame LiDAR points.
            lidar_intensity: (N,) intensity values.
            intrinsics: (3, 3) camera K matrix.
            T_cam_world: (4, 4) camera-from-world transform.
            ego_position: (3,) ego position.
            radar_points: Optional (N, 3) radar detections in world frame.
            radar_rcs: Optional (N,) RCS values.
            output_path: Optional save path.

        Returns:
            Multi-panel image as numpy array.
        """
        H, W = rgb_image.shape[:2]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=self.dpi)

        # Panel 1: Camera RGB
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title("Virtual Camera (RGB)", fontsize=10)
        axes[0, 0].set_axis_off()

        # Panel 2: Depth map
        dm = depth_map.copy()
        dm[~np.isfinite(dm)] = 0
        im_depth = axes[0, 1].imshow(
            dm, cmap=self.colormap,
            vmin=self.depth_range[0], vmax=self.depth_range[1],
        )
        axes[0, 1].set_title("Depth Map", fontsize=10)
        axes[0, 1].set_axis_off()
        plt.colorbar(im_depth, ax=axes[0, 1], shrink=0.7, label="Depth (m)")

        # Panel 3: Fusion overlay
        pts_hom = np.column_stack([lidar_points, np.ones(len(lidar_points))])
        pts_cam = (T_cam_world @ pts_hom.T).T[:, :3]
        front_mask = pts_cam[:, 2] > 0.5
        pts_front = pts_cam[front_mask]
        depths_front = pts_front[:, 2]

        px = intrinsics[0, 0] * pts_front[:, 0] / pts_front[:, 2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * pts_front[:, 1] / pts_front[:, 2] + intrinsics[1, 2]
        in_img = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        axes[1, 0].imshow(rgb_image)
        if in_img.sum() > 0:
            sc = axes[1, 0].scatter(
                px[in_img], py[in_img],
                c=depths_front[in_img],
                cmap=self.colormap, s=self.point_size,
                vmin=self.depth_range[0], vmax=self.depth_range[1],
                alpha=0.8, edgecolors="none",
            )
            plt.colorbar(sc, ax=axes[1, 0], shrink=0.7, label="Depth (m)")

        # Overlay radar detections on fusion panel
        if radar_points is not None and len(radar_points) > 0:
            r_hom = np.column_stack([radar_points, np.ones(len(radar_points))])
            r_cam = (T_cam_world @ r_hom.T).T[:, :3]
            r_front = r_cam[:, 2] > 0.5
            if r_front.sum() > 0:
                r_pts = r_cam[r_front]
                r_px = intrinsics[0, 0] * r_pts[:, 0] / r_pts[:, 2] + intrinsics[0, 2]
                r_py = intrinsics[1, 1] * r_pts[:, 1] / r_pts[:, 2] + intrinsics[1, 2]
                r_in = (r_px >= 0) & (r_px < W) & (r_py >= 0) & (r_py < H)
                if r_in.sum() > 0:
                    r_c = radar_rcs[r_front][r_in] if radar_rcs is not None else None
                    axes[1, 0].scatter(
                        r_px[r_in], r_py[r_in],
                        c=r_c, cmap="plasma",
                        s=self.point_size * 20, marker="D",
                        alpha=0.9, edgecolors="white", linewidths=0.5,
                    )

        axes[1, 0].set_title("LiDAR + Radar → Camera Fusion", fontsize=10)
        axes[1, 0].set_axis_off()

        # Panel 4: BEV
        pts_local = lidar_points - ego_position
        bev_range = 40.0
        bev_mask = (
            (np.abs(pts_local[:, 0]) < bev_range) &
            (np.abs(pts_local[:, 1]) < bev_range)
        )
        axes[1, 1].scatter(
            pts_local[bev_mask, 0], pts_local[bev_mask, 1],
            c=lidar_intensity[bev_mask] if lidar_intensity is not None else pts_local[bev_mask, 2],
            cmap=self.colormap, s=0.3, alpha=0.5, edgecolors="none",
        )
        ego_rect = plt.Rectangle((-2, -1), 4, 2, fill=True,
                                  facecolor="red", alpha=0.5, edgecolor="red")
        axes[1, 1].add_patch(ego_rect)

        # Overlay radar on BEV
        if radar_points is not None and len(radar_points) > 0:
            r_local = radar_points - ego_position
            r_bev_mask = (
                (np.abs(r_local[:, 0]) < bev_range) &
                (np.abs(r_local[:, 1]) < bev_range)
            )
            if r_bev_mask.sum() > 0:
                r_c = radar_rcs[r_bev_mask] if radar_rcs is not None else r_local[r_bev_mask, 2]
                axes[1, 1].scatter(
                    r_local[r_bev_mask, 0], r_local[r_bev_mask, 1],
                    c=r_c, cmap="plasma", s=25,
                    marker="D", alpha=0.9, edgecolors="white", linewidths=0.5,
                    label="Radar",
                )
                axes[1, 1].legend(loc="upper right", fontsize=7)

        axes[1, 1].set_xlim(-bev_range, bev_range)
        axes[1, 1].set_ylim(-bev_range, bev_range)
        axes[1, 1].set_aspect("equal")
        axes[1, 1].set_title("BEV - LiDAR + Radar", fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            "Virtual Sensor Suite — Multi-Modal Output",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        fig.canvas.draw()
        result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(result).save(output_path)

        return result
