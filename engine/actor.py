"""
Dynamic Actor Injection for 3D Gaussian Splatting scenes.

Implements object-centric Gaussians for rigid bodies (vehicles, pedestrians).
Each actor is a small set of Gaussians in a local coordinate frame, transformed
via SE(3) per timestep, and merged with the background scene at render time.

Key classes:
  - ActorGaussians: Local-frame Gaussian parameters for one actor
  - ActorTrajectory: Timestamped SE(3) trajectory with velocity for Doppler
  - ScenarioManager: Loads/generates actors and queries their state per frame
"""

import torch
import numpy as np
import yaml
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from scipy.spatial.transform import Rotation

from .trajectory import Trajectory, Pose


@dataclass
class TransformedActor:
    """Actor Gaussians transformed to world frame at a specific timestep."""
    name: str
    means: torch.Tensor       # (N, 3) world frame
    scales: torch.Tensor       # (N, 3) log-scales (unchanged)
    rotations: torch.Tensor    # (N, 4) quaternions (composed with actor pose)
    opacities: torch.Tensor    # (N, 1)
    sh_coeffs: torch.Tensor    # (N, C, 3)
    velocity: np.ndarray       # (3,) world-frame velocity
    center: np.ndarray         # (3,) world-frame center position
    radius: float              # bounding radius for Doppler assignment
    dimensions: np.ndarray     # (3,) [length, width, height] for AABB suppression


class ActorGaussians:
    """
    Gaussian parameters for a single actor in local coordinates.

    The actor's origin (0,0,0) is its center. Gaussians are distributed
    relative to this center to form the actor's shape.
    """

    def __init__(
        self,
        name: str,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        dimensions: tuple[float, float, float] = (4.5, 1.8, 1.5),
        device: str = "cuda",
    ):
        self.name = name
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.means = means.to(dev)
        self.scales = scales.to(dev)
        self.rotations = rotations.to(dev)
        self.opacities = opacities.to(dev)
        self.sh_coeffs = sh_coeffs.to(dev)
        self.dimensions = dimensions
        self.device = dev
        self.radius = max(dimensions) / 2.0 + 1.0  # bounding radius with margin

    @classmethod
    def create_synthetic_vehicle(
        cls,
        name: str = "vehicle",
        length: float = 4.5,
        width: float = 1.8,
        height: float = 1.5,
        color: tuple[float, float, float] = (0.4, 0.4, 0.5),
        n_gaussians: int = 300,
        device: str = "cuda",
    ) -> "ActorGaussians":
        """
        Generate a car-shaped Gaussian cluster in local coordinates.

        The car is centered at origin, aligned with x-forward, y-left, z-up.
        Body sits with bottom at z=0 (ground contact), top at z=height.
        """
        torch.manual_seed(hash(name) % 2**32)

        # Body shell: dense surface Gaussians
        n_body = int(n_gaussians * 0.7)
        n_roof = int(n_gaussians * 0.2)
        n_wheels = n_gaussians - n_body - n_roof

        # Body: box-like distribution
        body_means = torch.zeros(n_body, 3)
        body_means[:, 0] = (torch.rand(n_body) - 0.5) * length
        body_means[:, 1] = (torch.rand(n_body) - 0.5) * width
        body_means[:, 2] = torch.rand(n_body) * height * 0.6 + height * 0.1

        # Roof: upper portion, slightly narrower
        roof_means = torch.zeros(n_roof, 3)
        roof_means[:, 0] = (torch.rand(n_roof) - 0.5) * length * 0.6
        roof_means[:, 1] = (torch.rand(n_roof) - 0.5) * width * 0.9
        roof_means[:, 2] = torch.rand(n_roof) * height * 0.35 + height * 0.65

        # Wheels: four clusters at corners, low z
        n_per_wheel = n_wheels // 4
        wheel_positions = [
            (length * 0.35, width * 0.45, height * 0.08),
            (length * 0.35, -width * 0.45, height * 0.08),
            (-length * 0.35, width * 0.45, height * 0.08),
            (-length * 0.35, -width * 0.45, height * 0.08),
        ]
        wheel_means_list = []
        for wx, wy, wz in wheel_positions:
            wm = torch.zeros(n_per_wheel, 3)
            wm[:, 0] = torch.randn(n_per_wheel) * 0.15 + wx
            wm[:, 1] = torch.randn(n_per_wheel) * 0.1 + wy
            wm[:, 2] = torch.randn(n_per_wheel) * 0.1 + wz
            wheel_means_list.append(wm)
        wheel_means = torch.cat(wheel_means_list, dim=0)

        means = torch.cat([body_means, roof_means, wheel_means], dim=0)
        n_total = means.shape[0]

        # Scales: body is flat-ish, roof thinner, wheels small.
        # Scales must be large enough for LiDAR ray-casting to detect
        # (rays need to pass within ~3*scale of a Gaussian center).
        body_scales = torch.tensor([0.35, 0.20, 0.15]).expand(n_body, -1) + \
                      torch.randn(n_body, 3).abs() * 0.03
        roof_scales = torch.tensor([0.30, 0.18, 0.10]).expand(n_roof, -1) + \
                      torch.randn(n_roof, 3).abs() * 0.03
        wheel_scales = torch.tensor([0.15, 0.15, 0.15]).expand(wheel_means.shape[0], -1) + \
                       torch.randn(wheel_means.shape[0], 3).abs() * 0.02
        scales = torch.cat([body_scales, roof_scales, wheel_scales], dim=0).log()

        # Rotations: identity + small noise
        rotations = torch.zeros(n_total, 4)
        rotations[:, 0] = 1.0
        rotations += torch.randn_like(rotations) * 0.03
        rotations = torch.nn.functional.normalize(rotations, dim=-1)

        # Opacities: high
        opacities = torch.sigmoid(torch.randn(n_total, 1) * 0.3 + 3.0)

        # SH coefficients: body color + wheel dark
        n_sh = 16  # degree 3
        sh_coeffs = torch.zeros(n_total, n_sh, 3)
        color_t = torch.tensor(color)
        sh_coeffs[:n_body + n_roof, 0, :] = color_t + torch.randn(n_body + n_roof, 3) * 0.05
        sh_coeffs[n_body + n_roof:, 0, :] = torch.tensor([0.1, 0.1, 0.1]) + \
                                              torch.randn(wheel_means.shape[0], 3) * 0.02

        return cls(
            name=name, means=means, scales=scales, rotations=rotations,
            opacities=opacities, sh_coeffs=sh_coeffs,
            dimensions=(length, width, height), device=device,
        )

    @classmethod
    def create_synthetic_pedestrian(
        cls,
        name: str = "pedestrian",
        height: float = 1.7,
        width: float = 0.5,
        color: tuple[float, float, float] = (0.8, 0.6, 0.4),
        n_gaussians: int = 100,
        device: str = "cuda",
    ) -> "ActorGaussians":
        """Generate an upright pedestrian-shaped Gaussian cluster."""
        torch.manual_seed(hash(name) % 2**32)

        means = torch.zeros(n_gaussians, 3)
        means[:, 0] = torch.randn(n_gaussians) * width * 0.3
        means[:, 1] = torch.randn(n_gaussians) * width * 0.3
        means[:, 2] = torch.rand(n_gaussians) * height

        scales = torch.tensor([0.12, 0.12, 0.18]).expand(n_gaussians, -1) + \
                 torch.randn(n_gaussians, 3).abs() * 0.02
        scales = scales.log()

        rotations = torch.zeros(n_gaussians, 4)
        rotations[:, 0] = 1.0
        rotations = torch.nn.functional.normalize(
            rotations + torch.randn_like(rotations) * 0.02, dim=-1
        )

        opacities = torch.sigmoid(torch.randn(n_gaussians, 1) * 0.3 + 3.0)

        n_sh = 16
        sh_coeffs = torch.zeros(n_gaussians, n_sh, 3)
        sh_coeffs[:, 0, :] = torch.tensor(color) + torch.randn(n_gaussians, 3) * 0.05

        return cls(
            name=name, means=means, scales=scales, rotations=rotations,
            opacities=opacities, sh_coeffs=sh_coeffs,
            dimensions=(width, width, height), device=device,
        )

    def transform(self, T_world_actor: np.ndarray) -> TransformedActor:
        """
        Apply SE(3) transform to get world-frame Gaussians.

        Args:
            T_world_actor: 4x4 homogeneous transform (world <- actor local).

        Returns:
            TransformedActor with world-frame tensors.
        """
        T = torch.tensor(T_world_actor, dtype=torch.float32, device=self.device)
        R = T[:3, :3]
        t = T[:3, 3]

        # Transform means: world_pos = R @ local_pos + t
        means_world = (R @ self.means.T).T + t

        # Compose rotations: R_world_gaussian = R_world_actor @ R_actor_gaussian
        # Convert actor rotation matrix to quaternion
        R_np = T_world_actor[:3, :3]
        q_actor = Rotation.from_matrix(R_np).as_quat()  # [x, y, z, w] scipy
        q_actor_torch = torch.tensor(
            [q_actor[3], q_actor[0], q_actor[1], q_actor[2]],  # [w, x, y, z]
            dtype=torch.float32, device=self.device,
        )

        # Quaternion multiplication: q_world = q_actor * q_local
        rots_world = _quat_multiply(
            q_actor_torch.unsqueeze(0).expand(self.rotations.shape[0], -1),
            self.rotations,
        )

        center = T_world_actor[:3, 3].copy()

        return TransformedActor(
            name=self.name,
            means=means_world,
            scales=self.scales.clone(),
            rotations=rots_world,
            opacities=self.opacities.clone(),
            sh_coeffs=self.sh_coeffs.clone(),
            velocity=np.zeros(3),  # filled by ActorTrajectory
            center=center,
            radius=self.radius,
            dimensions=np.array(self.dimensions, dtype=np.float64),
        )


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion batches [w, x, y, z]."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class ActorTrajectory:
    """Trajectory for a single actor with velocity computation."""

    def __init__(self, name: str, trajectory: Trajectory):
        self.name = name
        self.trajectory = trajectory

    @classmethod
    def from_linear(
        cls,
        name: str,
        start: np.ndarray,
        velocity: np.ndarray,
        yaw: Optional[float] = None,
        duration: float = 10.0,
        dt: float = 0.1,
    ) -> "ActorTrajectory":
        """Create a constant-velocity linear trajectory."""
        n_frames = int(duration / dt) + 1
        if yaw is None:
            # Derive yaw from velocity direction
            yaw = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > 0.01 else 0.0

        poses = []
        for i in range(n_frames):
            t = i * dt
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = start + velocity * t
            T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            poses.append(Pose(T, timestamp=t))

        return cls(name, Trajectory(poses))

    @classmethod
    def from_static(
        cls,
        name: str,
        position: np.ndarray,
        yaw: float = 0.0,
        duration: float = 10.0,
    ) -> "ActorTrajectory":
        """Create a stationary actor."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = position
        T[:3, :3] = Rotation.from_euler("z", np.radians(yaw)).as_matrix()
        poses = [Pose(T, timestamp=0.0), Pose(T.copy(), timestamp=duration)]
        return cls(name, Trajectory(poses))

    def pose_at(self, timestamp: float) -> np.ndarray:
        """Get 4x4 SE(3) pose at timestamp."""
        return self.trajectory.interpolate(timestamp).T

    def velocity_at(self, timestamp: float) -> np.ndarray:
        """Compute world-frame velocity via finite difference."""
        dt = 0.01
        t0 = max(timestamp - dt, self.trajectory.start_time)
        t1 = min(timestamp + dt, self.trajectory.end_time)
        if t1 - t0 < 1e-6:
            return np.zeros(3)
        p0 = self.trajectory.interpolate(t0).translation
        p1 = self.trajectory.interpolate(t1).translation
        return (p1 - p0) / (t1 - t0)


class ScenarioManager:
    """
    Manages dynamic actors in a scene.

    Loads actor definitions and trajectories from YAML config or
    generates synthetic traffic procedurally.
    """

    def __init__(
        self,
        actors: list[tuple[ActorGaussians, ActorTrajectory]],
    ):
        self.actors = actors

    @classmethod
    def from_config(
        cls,
        scenario_path: str,
        device: str = "cuda",
    ) -> "ScenarioManager":
        """Load actors from a scenario YAML file."""
        path = Path(scenario_path)
        assert path.exists(), f"Scenario file not found: {scenario_path}"

        with open(path) as f:
            config = yaml.safe_load(f)

        actors = []
        for actor_cfg in config.get("actors", []):
            name = actor_cfg["name"]
            atype = actor_cfg.get("type", "vehicle")
            dims = actor_cfg.get("dimensions", [4.5, 1.8, 1.5])
            color = tuple(actor_cfg.get("color", [0.4, 0.4, 0.5]))

            # Create Gaussians
            if atype == "vehicle":
                gaussians = ActorGaussians.create_synthetic_vehicle(
                    name=name, length=dims[0], width=dims[1], height=dims[2],
                    color=color, device=device,
                )
            elif atype == "pedestrian":
                gaussians = ActorGaussians.create_synthetic_pedestrian(
                    name=name, height=dims[2], width=dims[0],
                    color=color, device=device,
                )
            else:
                raise ValueError(f"Unknown actor type: {atype}")

            # Create trajectory
            traj_cfg = actor_cfg.get("trajectory", {})
            traj_type = traj_cfg.get("type", "static")

            if traj_type == "linear":
                start = np.array(traj_cfg["start"], dtype=np.float64)
                vel = np.array(traj_cfg["velocity"], dtype=np.float64)
                traj = ActorTrajectory.from_linear(name, start, vel)
            elif traj_type == "static":
                pos = np.array(traj_cfg["position"], dtype=np.float64)
                yaw = traj_cfg.get("yaw", 0.0)
                traj = ActorTrajectory.from_static(name, pos, yaw)
            else:
                raise ValueError(f"Unknown trajectory type: {traj_type}")

            actors.append((gaussians, traj))

        # Auto-traffic
        auto_cfg = config.get("auto_traffic", {})
        if auto_cfg.get("enabled", False):
            n_vehicles = auto_cfg.get("n_vehicles", 5)
            speed_range = auto_cfg.get("speed_range", [5.0, 20.0])
            auto_actors = cls._generate_auto_traffic(
                n_vehicles, speed_range, device=device,
            )
            actors.extend(auto_actors)

        return cls(actors)

    @classmethod
    def generate_synthetic_traffic(
        cls,
        n_vehicles: int = 5,
        speed_range: tuple[float, float] = (5.0, 20.0),
        device: str = "cuda",
    ) -> "ScenarioManager":
        """Generate random traffic for quick testing."""
        actors = cls._generate_auto_traffic(n_vehicles, speed_range, device=device)
        return cls(actors)

    @staticmethod
    def _generate_auto_traffic(
        n_vehicles: int,
        speed_range: tuple[float, float],
        device: str = "cuda",
    ) -> list[tuple[ActorGaussians, ActorTrajectory]]:
        """Generate procedural vehicles with random trajectories."""
        np.random.seed(123)
        actors = []

        for i in range(n_vehicles):
            name = f"auto_vehicle_{i}"

            # Random car properties
            length = np.random.uniform(3.5, 5.5)
            width = np.random.uniform(1.6, 2.0)
            height = np.random.uniform(1.3, 1.8)
            color = tuple(np.random.rand(3) * 0.6 + 0.2)

            gaussians = ActorGaussians.create_synthetic_vehicle(
                name=name, length=length, width=width, height=height,
                color=color, device=device,
            )

            # Random trajectory: place along road, some same direction, some oncoming
            x_start = np.random.uniform(15.0, 60.0)
            lane = np.random.choice([-3.0, 3.0, -6.0, 6.0])
            speed = np.random.uniform(speed_range[0], speed_range[1])

            # Same direction or oncoming
            if lane > 0:  # left lanes: oncoming
                direction = -1.0
            else:  # right lanes: same direction
                direction = 1.0

            start = np.array([x_start, lane, 0.0])
            velocity = np.array([direction * speed, 0.0, 0.0])

            traj = ActorTrajectory.from_linear(name, start, velocity)
            actors.append((gaussians, traj))

        return actors

    def get_actors_at(self, timestamp: float) -> list[TransformedActor]:
        """
        Get all actors transformed to world frame at the given timestamp.

        Returns:
            List of TransformedActor with world-frame Gaussians and velocity.
        """
        result = []
        for gaussians, trajectory in self.actors:
            T_world_actor = trajectory.pose_at(timestamp)
            transformed = gaussians.transform(T_world_actor)
            transformed.velocity = trajectory.velocity_at(timestamp)
            transformed.center = T_world_actor[:3, 3].copy()
            result.append(transformed)
        return result

    @property
    def n_actors(self) -> int:
        return len(self.actors)

    @property
    def actor_names(self) -> list[str]:
        return [g.name for g, _ in self.actors]

    def __repr__(self) -> str:
        return f"ScenarioManager({self.n_actors} actors: {self.actor_names})"
