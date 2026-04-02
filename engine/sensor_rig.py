"""
VirtualSensorRig: Config-driven multi-sensor simulation orchestrator.

Reads a YAML config specifying the full sensor layout (cameras, LiDAR, radar)
and manages synchronized rendering across all sensors for each ego pose.
This is the primary API surface consumed by downstream teams.

Usage:
    scene = GaussianSplatScene.create_synthetic()
    rig = VirtualSensorRig.from_config("configs/kitti_rig.yaml", scene=scene)
    trajectory = Trajectory.generate_synthetic(n_frames=50)
    sensor_log = rig.simulate(trajectory, output_dir="./output/")
"""

import yaml
import time
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from .camera import VirtualCamera, GaussianSplatScene
from .lidar import VirtualLiDAR
from .radar import VirtualRadar
from .trajectory import Trajectory, Pose
from .actor import ScenarioManager


@dataclass
class SensorOutput:
    """Container for a single sensor's output at one timestep."""
    sensor_name: str
    sensor_type: str
    timestamp: float
    data: dict
    render_time_ms: float = 0.0


@dataclass
class FrameOutput:
    """Container for all sensor outputs at one timestep."""
    frame_idx: int
    timestamp: float
    ego_pose: np.ndarray
    sensor_outputs: dict[str, SensorOutput] = field(default_factory=dict)

    @property
    def total_render_time_ms(self) -> float:
        return sum(s.render_time_ms for s in self.sensor_outputs.values())


class VirtualSensorRig:
    """
    Multi-sensor simulation rig driven by YAML configuration.

    Manages a collection of virtual sensors (cameras, LiDAR, radar)
    with known extrinsics relative to the vehicle frame. Given an ego pose,
    all sensors fire synchronously, producing a coherent multi-modal snapshot.
    """

    def __init__(
        self,
        cameras: list[VirtualCamera],
        lidars: list[VirtualLiDAR],
        radars: list[VirtualRadar],
        scene: GaussianSplatScene,
        config: dict,
        scenario: Optional[ScenarioManager] = None,
    ):
        self.cameras = {cam.name: cam for cam in cameras}
        self.lidars = {lid.config.name: lid for lid in lidars}
        self.radars = {rad.config.name: rad for rad in radars}
        self.scene = scene
        self.config = config
        self.scenario = scenario

    @classmethod
    def from_config(
        cls,
        config_path: str,
        scene: GaussianSplatScene,
    ) -> "VirtualSensorRig":
        """
        Build the sensor rig from a YAML config file.

        Args:
            config_path: Path to the rig YAML config.
            scene: Pre-trained 3DGS scene to render from.

        Returns:
            Configured VirtualSensorRig instance.
        """
        config_path = Path(config_path)
        assert config_path.exists(), f"Config not found: {config_path}"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cameras = []
        lidars = []
        radars = []

        for sensor_cfg in config.get("sensors", []):
            stype = sensor_cfg["type"]
            if stype == "camera":
                cameras.append(VirtualCamera.from_config(sensor_cfg, scene))
            elif stype == "lidar":
                lidars.append(VirtualLiDAR.from_config(sensor_cfg, scene))
            elif stype == "radar":
                radars.append(VirtualRadar.from_config(sensor_cfg, scene))
            else:
                raise ValueError(f"Unknown sensor type: {stype}")

        return cls(cameras=cameras, lidars=lidars, radars=radars,
                   scene=scene, config=config)

    def render_frame(
        self,
        ego_pose: Pose,
        frame_idx: int = 0,
        render_cameras: bool = True,
        render_lidars: bool = True,
        render_radars: bool = True,
        lidar_stochastic: bool = True,
        ego_velocity: Optional[np.ndarray] = None,
    ) -> FrameOutput:
        """
        Render all sensors for a single ego pose.

        Args:
            ego_pose: The vehicle's world-frame pose.
            frame_idx: Frame sequence index.
            render_cameras: Whether to render camera sensors.
            render_lidars: Whether to render LiDAR sensors.
            render_radars: Whether to render radar sensors.
            lidar_stochastic: Whether to apply Monte Carlo ray dropping.
            ego_velocity: (3,) ego velocity in world frame for Doppler.

        Returns:
            FrameOutput containing all sensor data.
        """
        frame = FrameOutput(
            frame_idx=frame_idx,
            timestamp=ego_pose.timestamp,
            ego_pose=ego_pose.T,
        )

        T_wv = ego_pose.T

        # Inject dynamic actors into scene if scenario is present
        actor_infos = []  # [(center, velocity, radius), ...] for radar Doppler
        if self.scenario is not None:
            transformed_actors = self.scenario.get_actors_at(ego_pose.timestamp)
            if transformed_actors:
                merged_scene = self.scene.merge_actors(transformed_actors)
                actor_infos = [
                    (a.center, a.velocity, a.radius) for a in transformed_actors
                ]
            else:
                merged_scene = self.scene
        else:
            merged_scene = self.scene

        # Temporarily swap scene for all sensors
        original_scenes = {}
        if merged_scene is not self.scene:
            for name, cam in self.cameras.items():
                original_scenes[('cam', name)] = cam.scene
                cam.scene = merged_scene
            for name, lidar in self.lidars.items():
                original_scenes[('lidar', name)] = lidar.scene
                lidar.scene = merged_scene
            for name, radar in self.radars.items():
                original_scenes[('radar', name)] = radar.scene
                radar.scene = merged_scene

        try:
            if render_cameras:
                for name, cam in self.cameras.items():
                    t0 = time.perf_counter()
                    data = cam.render(T_wv, return_depth=True)
                    dt = (time.perf_counter() - t0) * 1000

                    frame.sensor_outputs[name] = SensorOutput(
                        sensor_name=name,
                        sensor_type="camera",
                        timestamp=ego_pose.timestamp,
                        data=data,
                        render_time_ms=dt,
                    )

            if render_lidars:
                for name, lidar in self.lidars.items():
                    t0 = time.perf_counter()
                    data = lidar.render(T_wv, stochastic=lidar_stochastic)
                    dt = (time.perf_counter() - t0) * 1000

                    frame.sensor_outputs[name] = SensorOutput(
                        sensor_name=name,
                        sensor_type="lidar",
                        timestamp=ego_pose.timestamp,
                        data=data,
                        render_time_ms=dt,
                    )

            if render_radars:
                for name, radar in self.radars.items():
                    t0 = time.perf_counter()
                    data = radar.render(
                        T_wv,
                        ego_velocity=ego_velocity,
                        actor_velocities=actor_infos if actor_infos else None,
                    )
                    dt = (time.perf_counter() - t0) * 1000

                    frame.sensor_outputs[name] = SensorOutput(
                        sensor_name=name,
                        sensor_type="radar",
                        timestamp=ego_pose.timestamp,
                        data=data,
                        render_time_ms=dt,
                    )
        finally:
            # Restore original scene references
            for (stype, name), orig_scene in original_scenes.items():
                if stype == 'cam':
                    self.cameras[name].scene = orig_scene
                elif stype == 'lidar':
                    self.lidars[name].scene = orig_scene
                elif stype == 'radar':
                    self.radars[name].scene = orig_scene

        return frame

    def simulate(
        self,
        trajectory: Trajectory,
        output_dir: Optional[str] = None,
        render_cameras: bool = True,
        render_lidars: bool = True,
        render_radars: bool = True,
        lidar_stochastic: bool = True,
        save_format: str = "kitti",
        verbose: bool = True,
    ) -> list[FrameOutput]:
        """
        Run batch simulation over an entire trajectory.

        Generates synchronized sensor data for every pose in the trajectory,
        optionally writing results to disk in a structured output directory.

        Args:
            trajectory: Sequence of timestamped ego poses.
            output_dir: If set, save outputs to this directory.
            render_cameras: Render camera sensors.
            render_lidars: Render LiDAR sensors.
            render_radars: Render radar sensors.
            lidar_stochastic: Apply stochastic ray dropping.
            save_format: Output format ("kitti" or "raw").
            verbose: Show progress bar.

        Returns:
            List of FrameOutput objects.

        Output directory structure (KITTI format):
            output_dir/
            ├── image_00/         # front_camera RGB
            │   ├── 000000.png
            │   └── ...
            ├── depth_00/         # front_camera depth
            │   └── ...
            ├── velodyne/         # LiDAR point clouds
            │   ├── 000000.bin
            │   └── ...
            ├── radar_00/         # Radar detections (x,y,z,rcs,v_radial)
            │   ├── 000000.bin
            │   └── ...
            ├── poses.txt         # Ego poses (KITTI format)
            └── timestamps.txt    # Frame timestamps
        """
        frames = []
        out_path = Path(output_dir) if output_dir else None

        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        iterator = enumerate(trajectory)
        if verbose:
            iterator = tqdm(
                list(iterator),
                desc="Simulating",
                unit="frame",
            )

        timing_stats = {"camera_ms": [], "lidar_ms": [], "radar_ms": []}

        # Pre-compute ego velocities from trajectory (finite difference)
        poses_list = list(trajectory)
        ego_velocities = []
        for i in range(len(poses_list)):
            if i == 0 and len(poses_list) > 1:
                dt = poses_list[1].timestamp - poses_list[0].timestamp
                if dt > 0:
                    dp = poses_list[1].T[:3, 3] - poses_list[0].T[:3, 3]
                    ego_velocities.append(dp / dt)
                else:
                    ego_velocities.append(np.zeros(3))
            elif i > 0:
                dt = poses_list[i].timestamp - poses_list[i-1].timestamp
                if dt > 0:
                    dp = poses_list[i].T[:3, 3] - poses_list[i-1].T[:3, 3]
                    ego_velocities.append(dp / dt)
                else:
                    ego_velocities.append(np.zeros(3))
            else:
                ego_velocities.append(np.zeros(3))

        for idx, pose in iterator:
            ego_vel = ego_velocities[idx] if idx < len(ego_velocities) else np.zeros(3)

            frame = self.render_frame(
                ego_pose=pose,
                frame_idx=idx,
                render_cameras=render_cameras,
                render_lidars=render_lidars,
                render_radars=render_radars,
                lidar_stochastic=lidar_stochastic,
                ego_velocity=ego_vel,
            )
            frames.append(frame)

            # Collect timing stats
            for so in frame.sensor_outputs.values():
                if so.sensor_type == "camera":
                    timing_stats["camera_ms"].append(so.render_time_ms)
                elif so.sensor_type == "lidar":
                    timing_stats["lidar_ms"].append(so.render_time_ms)
                elif so.sensor_type == "radar":
                    timing_stats["radar_ms"].append(so.render_time_ms)

            # Save to disk
            if out_path:
                self._save_frame(out_path, frame, save_format)

        # Save trajectory metadata
        if out_path:
            self._save_metadata(out_path, trajectory, timing_stats)

        if verbose:
            self._print_summary(frames, timing_stats)

        return frames

    def _save_frame(self, out_path: Path, frame: FrameOutput, fmt: str):
        """Save a single frame's sensor outputs to disk."""
        frame_id = f"{frame.frame_idx:06d}"

        for cam_idx, (name, cam) in enumerate(self.cameras.items()):
            if name not in frame.sensor_outputs:
                continue
            data = frame.sensor_outputs[name].data

            # Save RGB image
            img_dir = out_path / f"image_{cam_idx:02d}"
            img_dir.mkdir(parents=True, exist_ok=True)
            self._save_image(img_dir / f"{frame_id}.png", data["rgb"])

            # Save depth map
            if "depth" in data:
                depth_dir = out_path / f"depth_{cam_idx:02d}"
                depth_dir.mkdir(parents=True, exist_ok=True)
                np.save(depth_dir / f"{frame_id}.npy", data["depth"])

        for name, lidar in self.lidars.items():
            if name not in frame.sensor_outputs:
                continue
            data = frame.sensor_outputs[name].data

            vel_dir = out_path / "velodyne"
            vel_dir.mkdir(parents=True, exist_ok=True)

            if data["points"].shape[0] > 0:
                VirtualLiDAR.save_kitti_bin(
                    str(vel_dir / f"{frame_id}.bin"),
                    data["points"],
                    data["intensity"],
                )

                # Also save PCD for visualization
                pcd_dir = out_path / "pointclouds"
                pcd_dir.mkdir(parents=True, exist_ok=True)
                VirtualLiDAR.save_pcd(
                    str(pcd_dir / f"{frame_id}.pcd"),
                    data["points"],
                    data["intensity"],
                )

        for radar_idx, (name, radar) in enumerate(self.radars.items()):
            if name not in frame.sensor_outputs:
                continue
            data = frame.sensor_outputs[name].data

            if data["n_valid_points"] > 0:
                # Save binary (x, y, z, rcs, v_radial)
                radar_dir = out_path / f"radar_{radar_idx:02d}"
                radar_dir.mkdir(parents=True, exist_ok=True)
                VirtualRadar.save_radar_bin(
                    str(radar_dir / f"{frame_id}.bin"),
                    data["points"],
                    data["rcs"],
                    data["radial_velocity"],
                )

                # Also save PCD
                radar_pcd_dir = out_path / f"radar_pcd_{radar_idx:02d}"
                radar_pcd_dir.mkdir(parents=True, exist_ok=True)
                VirtualRadar.save_radar_pcd(
                    str(radar_pcd_dir / f"{frame_id}.pcd"),
                    data["points"],
                    data["rcs"],
                    data["radial_velocity"],
                )

    def _save_metadata(self, out_path: Path, trajectory: Trajectory, timing: dict):
        """Save trajectory poses and timestamps."""
        # Poses in KITTI format (3x4 row-major)
        poses_lines = []
        timestamps_lines = []
        for pose in trajectory:
            row = pose.T[:3, :].flatten()
            poses_lines.append(" ".join(f"{v:.6e}" for v in row))
            timestamps_lines.append(f"{pose.timestamp:.6f}")

        (out_path / "poses.txt").write_text("\n".join(poses_lines) + "\n")
        (out_path / "timestamps.txt").write_text("\n".join(timestamps_lines) + "\n")

        # Timing report
        report_lines = ["# Rendering Performance Report", ""]
        for key, times in timing.items():
            if times:
                arr = np.array(times)
                report_lines.extend([
                    f"## {key}",
                    f"  Mean: {arr.mean():.1f} ms",
                    f"  Std:  {arr.std():.1f} ms",
                    f"  Min:  {arr.min():.1f} ms",
                    f"  Max:  {arr.max():.1f} ms",
                    f"  FPS:  {1000.0 / arr.mean():.1f}",
                    "",
                ])
        (out_path / "performance.txt").write_text("\n".join(report_lines))

    @staticmethod
    def _save_image(filepath: Path, rgb: np.ndarray):
        """Save RGB image as PNG."""
        from PIL import Image
        img = Image.fromarray(rgb)
        img.save(str(filepath))

    @staticmethod
    def _print_summary(frames: list[FrameOutput], timing: dict):
        """Print simulation summary to console."""
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        print(f"  Frames rendered: {len(frames)}")

        for key, times in timing.items():
            if times:
                arr = np.array(times)
                print(f"\n  {key}:")
                print(f"    Mean:  {arr.mean():.1f} ms")
                print(f"    Std:   {arr.std():.1f} ms")
                print(f"    FPS:   {1000.0 / arr.mean():.1f}")

        total_time = sum(f.total_render_time_ms for f in frames)
        print(f"\n  Total render time: {total_time / 1000:.2f}s")
        print(f"  Overall FPS: {len(frames) / (total_time / 1000):.1f}")
        print("=" * 60)

    @property
    def sensor_names(self) -> list[str]:
        """List all sensor names in the rig."""
        names = list(self.cameras.keys()) + list(self.lidars.keys())
        names += list(self.radars.keys())
        return names

    def __repr__(self) -> str:
        return (
            f"VirtualSensorRig("
            f"cameras={list(self.cameras.keys())}, "
            f"lidars={list(self.lidars.keys())}, "
            f"radars={list(self.radars.keys())})"
        )
