"""
Trajectory loader for timestamped ego poses.

Supports KITTI odometry format (12-element row-major 3x4 transformation matrices)
and generic CSV/JSON formats. Provides interpolation for sub-frame querying.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


class Pose:
    """Rigid-body 6-DoF pose (SE(3)) with convenience accessors."""

    def __init__(self, T: np.ndarray, timestamp: float = 0.0):
        """
        Args:
            T: 4x4 homogeneous transformation matrix (world <- body).
            timestamp: Timestamp in seconds.
        """
        assert T.shape == (4, 4), f"Expected 4x4 matrix, got {T.shape}"
        self.T = T.astype(np.float64)
        self.timestamp = timestamp

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix."""
        return self.T[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """3-element translation vector."""
        return self.T[:3, 3]

    @property
    def position(self) -> np.ndarray:
        """Alias for translation (world-frame position of body origin)."""
        return self.translation

    @property
    def quaternion(self) -> np.ndarray:
        """Unit quaternion [x, y, z, w] (scipy convention)."""
        return Rotation.from_matrix(self.rotation).as_quat()

    @property
    def euler_deg(self) -> np.ndarray:
        """Euler angles [roll, pitch, yaw] in degrees."""
        return Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)

    def inverse(self) -> "Pose":
        """Return the inverse pose (body <- world)."""
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = self.rotation.T
        T_inv[:3, 3] = -self.rotation.T @ self.translation
        return Pose(T_inv, self.timestamp)

    def compose(self, other: "Pose") -> "Pose":
        """Compose transforms: self * other (chain right)."""
        return Pose(self.T @ other.T, self.timestamp)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform Nx3 points from body frame to world frame.

        Args:
            points: (N, 3) array in body frame.
        Returns:
            (N, 3) array in world frame.
        """
        assert points.ndim == 2 and points.shape[1] == 3
        return (self.rotation @ points.T).T + self.translation

    def __repr__(self) -> str:
        pos = self.translation
        rpy = self.euler_deg
        return (
            f"Pose(t={self.timestamp:.3f}s, "
            f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
            f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg)"
        )


class Trajectory:
    """
    Sequence of timestamped ego poses with interpolation support.

    Designed for batch simulation: iterate over poses to generate
    synchronized sensor data at each timestep.
    """

    def __init__(self, poses: list[Pose]):
        assert len(poses) > 0, "Trajectory must contain at least one pose"
        self.poses = sorted(poses, key=lambda p: p.timestamp)
        self._timestamps = np.array([p.timestamp for p in self.poses])

        # Pre-compute interpolation components for sub-frame queries
        if len(poses) > 1:
            self._translations = np.array([p.translation for p in self.poses])
            self._rotations = Rotation.from_matrix(
                np.array([p.rotation for p in self.poses])
            )
            self._trans_interp = interp1d(
                self._timestamps,
                self._translations,
                axis=0,
                kind="cubic" if len(poses) >= 4 else "linear",
                fill_value="extrapolate",
            )
            self._rot_interp = Slerp(self._timestamps, self._rotations)

    @classmethod
    def from_kitti(
        cls,
        poses_file: str,
        timestamps_file: Optional[str] = None,
    ) -> "Trajectory":
        """
        Load trajectory from KITTI odometry format.

        KITTI poses: each line is 12 floats (row-major 3x4 matrix).
        Timestamps: each line is a float (seconds). If not provided,
        frames are assigned 0.1s intervals (10 Hz).

        Args:
            poses_file: Path to poses.txt (or .txt with 3x4 matrices).
            timestamps_file: Optional path to timestamps.txt.
        """
        poses_path = Path(poses_file)
        assert poses_path.exists(), f"Poses file not found: {poses_file}"

        raw = np.loadtxt(poses_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        assert raw.shape[1] == 12, f"Expected 12 columns, got {raw.shape[1]}"

        n_frames = raw.shape[0]

        # Load or generate timestamps
        if timestamps_file and Path(timestamps_file).exists():
            timestamps = np.loadtxt(timestamps_file)
            assert len(timestamps) >= n_frames, "Fewer timestamps than poses"
            timestamps = timestamps[:n_frames]
        else:
            timestamps = np.arange(n_frames) * 0.1  # 10 Hz default

        poses = []
        for i in range(n_frames):
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = raw[i].reshape(3, 4)
            poses.append(Pose(T, timestamp=timestamps[i]))

        return cls(poses)

    @classmethod
    def from_transforms(
        cls,
        transforms: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> "Trajectory":
        """
        Create trajectory from Nx4x4 transformation matrices.

        Args:
            transforms: (N, 4, 4) array of homogeneous transforms.
            timestamps: (N,) array of timestamps. Defaults to 10 Hz.
        """
        n = transforms.shape[0]
        if timestamps is None:
            timestamps = np.arange(n) * 0.1
        poses = [Pose(transforms[i], timestamps[i]) for i in range(n)]
        return cls(poses)

    @classmethod
    def generate_synthetic(
        cls,
        n_frames: int = 100,
        speed_mps: float = 10.0,
        dt: float = 0.1,
        curve_radius: Optional[float] = None,
    ) -> "Trajectory":
        """
        Generate a synthetic trajectory for testing.

        Args:
            n_frames: Number of frames.
            speed_mps: Forward speed in m/s.
            dt: Time step between frames.
            curve_radius: If set, drive in a circle of this radius.
                         If None, drive straight forward.
        """
        poses = []
        for i in range(n_frames):
            t = i * dt
            T = np.eye(4, dtype=np.float64)

            if curve_radius is not None:
                # Circular trajectory
                angle = (speed_mps * t) / curve_radius
                T[0, 3] = curve_radius * np.sin(angle)
                T[1, 3] = curve_radius * (1 - np.cos(angle))
                T[2, 3] = 0.0
                # Heading follows tangent
                R = Rotation.from_euler("z", angle).as_matrix()
                T[:3, :3] = R
            else:
                # Straight-line trajectory
                T[0, 3] = speed_mps * t
                T[1, 3] = 0.0
                T[2, 3] = 0.0

            poses.append(Pose(T, timestamp=t))
        return cls(poses)

    def interpolate(self, timestamp: float) -> Pose:
        """
        Interpolate pose at an arbitrary timestamp.

        Uses cubic spline for translation and SLERP for rotation.
        """
        if len(self.poses) == 1:
            return self.poses[0]

        t_clamped = np.clip(timestamp, self._timestamps[0], self._timestamps[-1])
        trans = self._trans_interp(t_clamped)
        rot = self._rot_interp(t_clamped)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = trans
        return Pose(T, timestamp=timestamp)

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        return self._timestamps[-1] - self._timestamps[0]

    @property
    def start_time(self) -> float:
        return self._timestamps[0]

    @property
    def end_time(self) -> float:
        return self._timestamps[-1]

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx) -> Pose:
        return self.poses[idx]

    def __iter__(self):
        return iter(self.poses)

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_frames={len(self)}, "
            f"duration={self.duration:.1f}s, "
            f"t=[{self.start_time:.2f}, {self.end_time:.2f}])"
        )
