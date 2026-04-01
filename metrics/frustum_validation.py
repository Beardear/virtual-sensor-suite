"""
Frustum-Culled Validation for spatially meaningful V&V.

Crops ground-truth LiDAR points to the virtual camera's exact field of view
before computing depth errors. This ensures comparisons are spatially consistent —
we only evaluate fidelity within the region the virtual camera actually observes.

This is a critical distinction from naive whole-scene comparison: a global metric
can be dominated by regions far from the camera, while frustum-culled metrics
reflect what the perception stack actually sees.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .depth_error import DepthErrorMetric, DepthMetrics


@dataclass
class FrustumParams:
    """Camera frustum parameters for 3D culling."""
    near: float = 0.5
    far: float = 80.0
    fov_x: float = 1.57  # radians (~90 deg)
    fov_y: float = 0.58  # radians (~33 deg)

    @classmethod
    def from_intrinsics(
        cls,
        fx: float, fy: float,
        cx: float, cy: float,
        width: int, height: int,
        near: float = 0.5,
        far: float = 80.0,
    ) -> "FrustumParams":
        """Compute frustum from camera intrinsics."""
        fov_x = 2.0 * np.arctan(width / (2.0 * fx))
        fov_y = 2.0 * np.arctan(height / (2.0 * fy))
        return cls(near=near, far=far, fov_x=fov_x, fov_y=fov_y)


class FrustumValidator:
    """
    Frustum-culled V&V between synthetic and ground-truth LiDAR.

    Pipeline:
      1. Transform GT LiDAR points into the virtual camera frame
      2. Cull points outside the camera frustum (near/far + FoV)
      3. Compute depth metrics only on the culled subset

    This provides a spatially meaningful comparison that answers:
    "How accurate is the simulation within the camera's actual view?"
    """

    def __init__(
        self,
        depth_metric: Optional[DepthErrorMetric] = None,
    ):
        self.depth_metric = depth_metric or DepthErrorMetric()

    def cull_to_frustum(
        self,
        points: np.ndarray,
        T_cam_world: np.ndarray,
        frustum: FrustumParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cull 3D points to a camera frustum.

        Args:
            points: (N, 3) points in world frame.
            T_cam_world: (4, 4) world-to-camera transform.
            frustum: Camera frustum parameters.

        Returns:
            culled_points: (M, 3) points in world frame that fall within frustum.
            culled_mask: (N,) boolean mask.
        """
        # Transform to camera frame
        pts_hom = np.column_stack([points, np.ones(len(points))])
        pts_cam = (T_cam_world @ pts_hom.T).T[:, :3]

        # Depth (z) culling
        z = pts_cam[:, 2]
        mask = (z > frustum.near) & (z < frustum.far)

        # Horizontal FoV culling
        half_fov_x = frustum.fov_x / 2.0
        angle_x = np.abs(np.arctan2(pts_cam[:, 0], z))
        mask &= angle_x < half_fov_x

        # Vertical FoV culling
        half_fov_y = frustum.fov_y / 2.0
        angle_y = np.abs(np.arctan2(pts_cam[:, 1], z))
        mask &= angle_y < half_fov_y

        return points[mask], mask

    def validate(
        self,
        synthetic_points: np.ndarray,
        synthetic_depths: np.ndarray,
        gt_points: np.ndarray,
        gt_depths: np.ndarray,
        T_cam_world: np.ndarray,
        frustum: FrustumParams,
    ) -> dict:
        """
        Run frustum-culled validation.

        Computes depth metrics for:
          1. Full scene (all points)
          2. Frustum-only (points within camera FOV)

        Args:
            synthetic_points: (N, 3) synthetic LiDAR points (world frame).
            synthetic_depths: (N,) synthetic range values.
            gt_points: (M, 3) ground-truth LiDAR points (world frame).
            gt_depths: (M,) ground-truth range values.
            T_cam_world: (4, 4) camera-from-world transform.
            frustum: Camera frustum parameters.

        Returns:
            dict with 'full_scene' and 'frustum_culled' DepthMetrics,
            plus culling statistics.
        """
        # Full-scene metrics
        full_metrics = self.depth_metric.evaluate_pointclouds(
            synthetic_points, synthetic_depths,
            gt_points, gt_depths,
        )

        # Frustum-cull both point clouds
        synth_culled, synth_mask = self.cull_to_frustum(
            synthetic_points, T_cam_world, frustum
        )
        gt_culled, gt_mask = self.cull_to_frustum(
            gt_points, T_cam_world, frustum
        )

        synth_depths_culled = synthetic_depths[synth_mask]
        gt_depths_culled = gt_depths[gt_mask]

        # Frustum-culled metrics
        frustum_metrics = self.depth_metric.evaluate_pointclouds(
            synth_culled, synth_depths_culled,
            gt_culled, gt_depths_culled,
        )

        return {
            "full_scene": full_metrics,
            "frustum_culled": frustum_metrics,
            "stats": {
                "total_synthetic": len(synthetic_points),
                "total_gt": len(gt_points),
                "frustum_synthetic": len(synth_culled),
                "frustum_gt": len(gt_culled),
                "synth_cull_ratio": len(synth_culled) / max(len(synthetic_points), 1),
                "gt_cull_ratio": len(gt_culled) / max(len(gt_points), 1),
            },
        }

    def validate_depth_map_with_frustum(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        intrinsics: np.ndarray,
        frustum: Optional[FrustumParams] = None,
    ) -> dict:
        """
        Validate depth maps with optional frustum-based masking.

        Creates a valid-pixel mask based on the frustum's depth range,
        then computes depth metrics only within that region.

        Args:
            pred_depth: (H, W) predicted depth map.
            gt_depth: (H, W) ground-truth depth map.
            intrinsics: (3, 3) camera intrinsic matrix.
            frustum: Optional frustum params for depth range filtering.

        Returns:
            dict with 'full' and 'frustum_filtered' DepthMetrics.
        """
        # Full depth map comparison
        full_metrics = self.depth_metric.evaluate_depth_maps(pred_depth, gt_depth)

        if frustum is None:
            return {"full": full_metrics}

        # Apply frustum depth range as mask
        mask = (gt_depth > frustum.near) & (gt_depth < frustum.far)
        frustum_metrics = self.depth_metric.evaluate_depth_maps(
            pred_depth, gt_depth, valid_mask=mask
        )

        n_total = np.isfinite(gt_depth).sum()
        n_frustum = mask.sum()

        return {
            "full": full_metrics,
            "frustum_filtered": frustum_metrics,
            "stats": {
                "total_pixels": int(n_total),
                "frustum_pixels": int(n_frustum),
                "cull_ratio": float(n_frustum / max(n_total, 1)),
            },
        }

    @staticmethod
    def generate_report(results: dict) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 60,
            "Frustum-Culled V&V Report",
            "=" * 60,
        ]

        if "full_scene" in results:
            lines.append("\n--- Full Scene ---")
            m = results["full_scene"]
            lines.append(f"  MAE:    {m.mae:.4f} m")
            lines.append(f"  RMSE:   {m.rmse:.4f} m")
            lines.append(f"  AbsRel: {m.abs_rel:.4f}")
            lines.append(f"  d<1.25: {m.delta_1:.3f}")
            lines.append(f"  Points: {m.n_valid}")

        if "frustum_culled" in results:
            lines.append("\n--- Frustum Culled ---")
            m = results["frustum_culled"]
            lines.append(f"  MAE:    {m.mae:.4f} m")
            lines.append(f"  RMSE:   {m.rmse:.4f} m")
            lines.append(f"  AbsRel: {m.abs_rel:.4f}")
            lines.append(f"  d<1.25: {m.delta_1:.3f}")
            lines.append(f"  Points: {m.n_valid}")

        if "stats" in results:
            lines.append("\n--- Culling Statistics ---")
            for k, v in results["stats"].items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}")
                else:
                    lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)
