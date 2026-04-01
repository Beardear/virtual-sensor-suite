"""
Depth Error Metrics for V&V (Verification & Validation).

Compares synthetic LiDAR/depth data against ground-truth measurements
to quantify sensor simulation fidelity. Implements standard depth
evaluation metrics used in autonomous driving research.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DepthMetrics:
    """Container for depth evaluation results."""
    mae: float            # Mean Absolute Error (L1)
    rmse: float           # Root Mean Squared Error
    median_ae: float      # Median Absolute Error (robust to outliers)
    abs_rel: float        # Absolute Relative Error: mean(|d_pred - d_gt| / d_gt)
    sq_rel: float         # Squared Relative Error: mean((d_pred - d_gt)^2 / d_gt)
    delta_1: float        # % of points with max(d_pred/d_gt, d_gt/d_pred) < 1.25
    delta_2: float        # < 1.25^2
    delta_3: float        # < 1.25^3
    n_valid: int          # Number of valid comparison points
    mean_gt_depth: float  # Mean ground-truth depth (for context)

    def __repr__(self) -> str:
        return (
            f"DepthMetrics(\n"
            f"  MAE={self.mae:.4f}m, RMSE={self.rmse:.4f}m, "
            f"MedianAE={self.median_ae:.4f}m\n"
            f"  AbsRel={self.abs_rel:.4f}, SqRel={self.sq_rel:.4f}\n"
            f"  delta<1.25: {self.delta_1:.3f}, "
            f"delta<1.25^2: {self.delta_2:.3f}, "
            f"delta<1.25^3: {self.delta_3:.3f}\n"
            f"  n_valid={self.n_valid}, mean_gt_depth={self.mean_gt_depth:.1f}m\n)"
        )

    def to_dict(self) -> dict:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "median_ae": self.median_ae,
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "n_valid": self.n_valid,
            "mean_gt_depth": self.mean_gt_depth,
        }


class DepthErrorMetric:
    """
    Compute depth error metrics between synthetic and ground-truth depth data.

    Supports comparison between:
      - Depth maps (2D, e.g., from camera rendering)
      - Point clouds (3D, e.g., from LiDAR simulation)
      - Sparse-to-dense matching (LiDAR vs depth map)

    For LiDAR-to-LiDAR comparison, points are matched using nearest-neighbor
    in 3D space with a configurable distance threshold.
    """

    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
        match_threshold: float = 0.5,
    ):
        """
        Args:
            min_depth: Minimum valid depth in meters.
            max_depth: Maximum valid depth in meters.
            match_threshold: Max distance (m) for point matching in 3D comparisons.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.match_threshold = match_threshold

    def evaluate_depth_maps(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> DepthMetrics:
        """
        Compare predicted vs ground-truth depth maps.

        Args:
            pred_depth: (H, W) predicted depth in meters.
            gt_depth: (H, W) ground-truth depth in meters.
            valid_mask: Optional (H, W) boolean mask for valid pixels.

        Returns:
            DepthMetrics with all standard evaluation metrics.
        """
        assert pred_depth.shape == gt_depth.shape, \
            f"Shape mismatch: {pred_depth.shape} vs {gt_depth.shape}"

        # Build validity mask
        mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
        mask &= (pred_depth > self.min_depth) & (pred_depth < self.max_depth)
        mask &= np.isfinite(pred_depth) & np.isfinite(gt_depth)
        if valid_mask is not None:
            mask &= valid_mask

        if mask.sum() == 0:
            return DepthMetrics(
                mae=float("inf"), rmse=float("inf"), median_ae=float("inf"),
                abs_rel=float("inf"), sq_rel=float("inf"),
                delta_1=0.0, delta_2=0.0, delta_3=0.0,
                n_valid=0, mean_gt_depth=0.0,
            )

        pred = pred_depth[mask]
        gt = gt_depth[mask]

        return self._compute_metrics(pred, gt)

    def evaluate_pointclouds(
        self,
        pred_points: np.ndarray,
        pred_depths: np.ndarray,
        gt_points: np.ndarray,
        gt_depths: np.ndarray,
    ) -> DepthMetrics:
        """
        Compare predicted vs ground-truth LiDAR point clouds.

        Points are matched using nearest-neighbor search in 3D.
        Only matched pairs within the distance threshold are evaluated.

        Args:
            pred_points: (N, 3) predicted 3D points.
            pred_depths: (N,) predicted range values.
            gt_points: (M, 3) ground-truth 3D points.
            gt_depths: (M,) ground-truth range values.

        Returns:
            DepthMetrics for matched point pairs.
        """
        from scipy.spatial import cKDTree

        if len(pred_points) == 0 or len(gt_points) == 0:
            return DepthMetrics(
                mae=float("inf"), rmse=float("inf"), median_ae=float("inf"),
                abs_rel=float("inf"), sq_rel=float("inf"),
                delta_1=0.0, delta_2=0.0, delta_3=0.0,
                n_valid=0, mean_gt_depth=0.0,
            )

        # Build KD-tree on ground truth
        tree = cKDTree(gt_points)
        distances, indices = tree.query(pred_points, k=1)

        # Filter by distance threshold
        valid = distances < self.match_threshold
        if valid.sum() == 0:
            return DepthMetrics(
                mae=float("inf"), rmse=float("inf"), median_ae=float("inf"),
                abs_rel=float("inf"), sq_rel=float("inf"),
                delta_1=0.0, delta_2=0.0, delta_3=0.0,
                n_valid=0, mean_gt_depth=0.0,
            )

        matched_pred_depth = pred_depths[valid]
        matched_gt_depth = gt_depths[indices[valid]]

        # Range filter
        depth_valid = (
            (matched_gt_depth > self.min_depth) &
            (matched_gt_depth < self.max_depth) &
            (matched_pred_depth > self.min_depth) &
            (matched_pred_depth < self.max_depth)
        )
        matched_pred_depth = matched_pred_depth[depth_valid]
        matched_gt_depth = matched_gt_depth[depth_valid]

        return self._compute_metrics(matched_pred_depth, matched_gt_depth)

    def evaluate_lidar_vs_depthmap(
        self,
        lidar_points: np.ndarray,
        lidar_depths: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_lidar: np.ndarray,
    ) -> DepthMetrics:
        """
        Compare LiDAR point cloud against a camera depth map.

        Projects LiDAR points into the camera frame, looks up corresponding
        depth map values, and computes error metrics.

        Args:
            lidar_points: (N, 3) LiDAR points in LiDAR frame.
            lidar_depths: (N,) LiDAR range values.
            depth_map: (H, W) depth map in meters.
            intrinsics: (3, 3) camera intrinsic matrix K.
            T_cam_lidar: (4, 4) transform from LiDAR to camera frame.

        Returns:
            DepthMetrics for projected point comparisons.
        """
        H, W = depth_map.shape

        # Transform LiDAR points to camera frame
        pts_hom = np.column_stack([lidar_points, np.ones(len(lidar_points))])
        pts_cam = (T_cam_lidar @ pts_hom.T).T[:, :3]

        # Filter points behind camera
        in_front = pts_cam[:, 2] > self.min_depth
        pts_cam = pts_cam[in_front]
        lidar_d = lidar_depths[in_front]

        # Project to pixel coordinates
        px = intrinsics[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + intrinsics[0, 2]
        py = intrinsics[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + intrinsics[1, 2]

        # Round to nearest pixel and filter within image bounds
        px_int = np.round(px).astype(int)
        py_int = np.round(py).astype(int)
        in_bounds = (px_int >= 0) & (px_int < W) & (py_int >= 0) & (py_int < H)

        px_int = px_int[in_bounds]
        py_int = py_int[in_bounds]
        lidar_d = lidar_d[in_bounds]
        cam_d = pts_cam[in_bounds, 2]  # Depth along camera z-axis

        # Look up depth map values
        dm_d = depth_map[py_int, px_int]

        # Filter valid depth map values
        valid = (dm_d > self.min_depth) & (dm_d < self.max_depth) & np.isfinite(dm_d)
        pred = dm_d[valid]
        gt = cam_d[valid]

        if len(gt) == 0:
            return DepthMetrics(
                mae=float("inf"), rmse=float("inf"), median_ae=float("inf"),
                abs_rel=float("inf"), sq_rel=float("inf"),
                delta_1=0.0, delta_2=0.0, delta_3=0.0,
                n_valid=0, mean_gt_depth=0.0,
            )

        return self._compute_metrics(pred, gt)

    def _compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> DepthMetrics:
        """Compute all depth metrics from matched prediction/ground-truth arrays."""
        assert len(pred) == len(gt) and len(pred) > 0

        err = np.abs(pred - gt)

        # L1 / MAE
        mae = float(err.mean())

        # RMSE
        rmse = float(np.sqrt(((pred - gt) ** 2).mean()))

        # Median Absolute Error
        median_ae = float(np.median(err))

        # Absolute Relative Error
        abs_rel = float((err / gt).mean())

        # Squared Relative Error
        sq_rel = float((((pred - gt) ** 2) / gt).mean())

        # Delta thresholds
        ratio = np.maximum(pred / gt, gt / pred)
        delta_1 = float((ratio < 1.25).mean())
        delta_2 = float((ratio < 1.25 ** 2).mean())
        delta_3 = float((ratio < 1.25 ** 3).mean())

        return DepthMetrics(
            mae=mae,
            rmse=rmse,
            median_ae=median_ae,
            abs_rel=abs_rel,
            sq_rel=sq_rel,
            delta_1=delta_1,
            delta_2=delta_2,
            delta_3=delta_3,
            n_valid=len(pred),
            mean_gt_depth=float(gt.mean()),
        )
