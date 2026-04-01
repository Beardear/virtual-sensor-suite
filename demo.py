#!/usr/bin/env python3
"""
Virtual Sensor Suite — Interactive Demo

Demonstrates the full pipeline:
  1. Load/create a 3DGS scene
  2. Configure a multi-sensor rig from YAML
  3. Generate or load a driving trajectory
  4. Run synchronized batch simulation (camera + LiDAR)
  5. Compute V&V metrics (depth error + frustum validation)
  6. Generate sensor fusion visualizations
  7. Benchmark rendering performance

Usage:
    # Full demo with synthetic scene
    python demo.py --synthetic --n-frames 20 --output ./output

    # With trained model
    python demo.py --model-path /path/to/checkpoint.pth --config configs/kitti_rig.yaml

    # Benchmark only
    python demo.py --synthetic --benchmark --n-frames 50

    # Reduced LiDAR for faster testing
    python demo.py --synthetic --n-frames 5 --lidar-channels 16 --lidar-ppc 256
"""

import argparse
import time
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.camera import GaussianSplatScene, VirtualCamera, CameraIntrinsics
from engine.lidar import VirtualLiDAR
from engine.sensor_rig import VirtualSensorRig
from engine.trajectory import Trajectory, Pose
from metrics.depth_error import DepthErrorMetric
from metrics.frustum_validation import FrustumValidator, FrustumParams
from visualization.fusion_overlay import FusionOverlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual Sensor Suite Demo — 3DGS-based AV sensor simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scene options
    scene_group = parser.add_argument_group("Scene")
    scene_group.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained 3DGS checkpoint (.pth)",
    )
    scene_group.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic Gaussian Splat scene for testing",
    )
    scene_group.add_argument(
        "--n-gaussians", type=int, default=50000,
        help="Number of Gaussians in synthetic scene (default: 50000)",
    )

    # Rig config
    rig_group = parser.add_argument_group("Sensor Rig")
    rig_group.add_argument(
        "--config", type=str, default="configs/kitti_rig.yaml",
        help="Path to sensor rig YAML config",
    )
    rig_group.add_argument(
        "--lidar-channels", type=int, default=None,
        help="Override LiDAR channel count (for faster testing)",
    )
    rig_group.add_argument(
        "--lidar-ppc", type=int, default=None,
        help="Override LiDAR points-per-channel (for faster testing)",
    )

    # Trajectory
    traj_group = parser.add_argument_group("Trajectory")
    traj_group.add_argument(
        "--poses-file", type=str, default=None,
        help="Path to KITTI poses.txt for real trajectory",
    )
    traj_group.add_argument(
        "--n-frames", type=int, default=10,
        help="Number of frames for synthetic trajectory (default: 10)",
    )
    traj_group.add_argument(
        "--speed", type=float, default=10.0,
        help="Ego speed in m/s for synthetic trajectory (default: 10.0)",
    )
    traj_group.add_argument(
        "--curve-radius", type=float, default=None,
        help="Curve radius for circular trajectory (default: straight line)",
    )

    # Simulation
    sim_group = parser.add_argument_group("Simulation")
    sim_group.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)",
    )
    sim_group.add_argument(
        "--no-camera", action="store_true",
        help="Skip camera rendering",
    )
    sim_group.add_argument(
        "--no-lidar", action="store_true",
        help="Skip LiDAR rendering",
    )
    sim_group.add_argument(
        "--deterministic", action="store_true",
        help="Disable stochastic ray dropping (deterministic LiDAR)",
    )
    sim_group.add_argument(
        "--ray-drop-prob", type=float, default=0.15,
        help="Ray drop probability for Monte Carlo LiDAR (default: 0.15)",
    )

    # Benchmark
    bench_group = parser.add_argument_group("Benchmark")
    bench_group.add_argument(
        "--benchmark", action="store_true",
        help="Run performance benchmark",
    )
    bench_group.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup frames before benchmarking (default: 3)",
    )

    # Visualization
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--no-vis", action="store_true",
        help="Skip visualization generation",
    )
    vis_group.add_argument(
        "--ray-drop-heatmap", action="store_true",
        help="Generate ray drop probability heatmap (slow: Monte Carlo sampling)",
    )

    # System
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    sys_group.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def print_header():
    print("=" * 64)
    print("  Virtual Sensor Suite — 3DGS-based AV Sensor Simulation")
    print("  Probabilistic Multi-Modal Rendering Engine")
    print("=" * 64)
    print()


def load_scene(args) -> GaussianSplatScene:
    """Load or create the 3DGS scene."""
    if args.model_path:
        print(f"Loading trained model from: {args.model_path}")
        return GaussianSplatScene.load_checkpoint(args.model_path, device=args.device)
    elif args.synthetic:
        print(f"Creating synthetic scene ({args.n_gaussians} Gaussians)...")
        return GaussianSplatScene.create_synthetic(
            n_gaussians=args.n_gaussians,
            device=args.device,
        )
    else:
        print("ERROR: Specify --model-path or --synthetic")
        sys.exit(1)


def load_trajectory(args) -> Trajectory:
    """Load or generate trajectory."""
    if args.poses_file:
        print(f"Loading trajectory from: {args.poses_file}")
        return Trajectory.from_kitti(args.poses_file)
    else:
        print(f"Generating synthetic trajectory: {args.n_frames} frames, "
              f"{args.speed} m/s, "
              f"{'curve r=' + str(args.curve_radius) if args.curve_radius else 'straight'}")
        return Trajectory.generate_synthetic(
            n_frames=args.n_frames,
            speed_mps=args.speed,
            curve_radius=args.curve_radius,
        )


def apply_config_overrides(config: dict, args) -> dict:
    """Apply CLI overrides to the YAML config."""
    for sensor in config.get("sensors", []):
        if sensor["type"] == "lidar":
            if args.lidar_channels is not None:
                sensor["channels"] = args.lidar_channels
            if args.lidar_ppc is not None:
                sensor["points_per_channel"] = args.lidar_ppc
            sensor["ray_drop_prob"] = args.ray_drop_prob
    return config


def run_benchmark(rig: VirtualSensorRig, trajectory: Trajectory, args):
    """Run isolated performance benchmarks for each sensor type."""
    print("\n" + "=" * 64)
    print("  Performance Benchmark")
    print("=" * 64)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pose = trajectory[0]

    # Camera benchmark
    if not args.no_camera and rig.cameras:
        cam = list(rig.cameras.values())[0]
        print(f"\n  Camera: {cam.name} ({cam.intrinsics.width}x{cam.intrinsics.height})")

        # Warmup
        for _ in range(args.warmup):
            cam.render(pose.T)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        n_bench = max(args.n_frames, 10)
        for i in range(n_bench):
            t_pose = trajectory[i % len(trajectory)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            cam.render(t_pose.T)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"    Mean:   {times.mean():.1f} ms")
        print(f"    Std:    {times.std():.1f} ms")
        print(f"    Min:    {times.min():.1f} ms")
        print(f"    Max:    {times.max():.1f} ms")
        print(f"    FPS:    {1000.0 / times.mean():.1f}")

    # LiDAR benchmark
    if not args.no_lidar and rig.lidars:
        lidar = list(rig.lidars.values())[0]
        print(f"\n  LiDAR: {lidar.config.name} ({lidar.config.channels}ch, "
              f"{lidar.config.channels * lidar.config.points_per_channel} rays)")

        # Warmup
        for _ in range(args.warmup):
            lidar.render(pose.T, stochastic=not args.deterministic)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_det = []
        times_stoch = []
        n_bench = max(args.n_frames, 10)
        for i in range(n_bench):
            t_pose = trajectory[i % len(trajectory)]
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Deterministic
            t0 = time.perf_counter()
            lidar.render(t_pose.T, stochastic=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_det.append((time.perf_counter() - t0) * 1000)

            # Stochastic
            t0 = time.perf_counter()
            lidar.render(t_pose.T, stochastic=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_stoch.append((time.perf_counter() - t0) * 1000)

        times_det = np.array(times_det)
        times_stoch = np.array(times_stoch)
        print(f"\n    Deterministic:")
        print(f"      Mean:  {times_det.mean():.1f} ms  |  FPS: {1000.0 / times_det.mean():.1f}")
        print(f"    Stochastic (p_drop={args.ray_drop_prob}):")
        print(f"      Mean:  {times_stoch.mean():.1f} ms  |  FPS: {1000.0 / times_stoch.mean():.1f}")

    print("\n" + "=" * 64)


def run_vv_metrics(frames, rig, args):
    """Compute V&V metrics using self-consistency checks."""
    print("\n" + "=" * 64)
    print("  V&V Metrics — Self-Consistency Analysis")
    print("=" * 64)

    depth_metric = DepthErrorMetric(min_depth=0.5, max_depth=80.0)
    frustum_validator = FrustumValidator(depth_metric=depth_metric)

    # Compare deterministic vs stochastic LiDAR
    if not args.no_lidar and rig.lidars:
        lidar = list(rig.lidars.values())[0]
        pose = frames[0].ego_pose

        print("\n  Deterministic vs Stochastic LiDAR comparison:")
        det_result = lidar.render(pose, stochastic=False)
        stoch_result = lidar.render(pose, stochastic=True)

        if det_result["n_valid_points"] > 0 and stoch_result["n_valid_points"] > 0:
            metrics = depth_metric.evaluate_pointclouds(
                stoch_result["points"], stoch_result["depth"],
                det_result["points"], det_result["depth"],
            )
            print(f"    Deterministic points: {det_result['n_valid_points']}")
            print(f"    Stochastic points:   {stoch_result['n_valid_points']}")
            print(f"    Ray drop rate:       {1.0 - stoch_result['n_valid_points'] / max(det_result['n_valid_points'], 1):.3f}")
            print(f"    MAE (stoch vs det):  {metrics.mae:.4f} m")
            print(f"    delta < 1.25:        {metrics.delta_1:.3f}")

    # Frustum-culled validation
    if not args.no_camera and not args.no_lidar and rig.cameras and rig.lidars:
        cam_name = list(rig.cameras.keys())[0]
        cam = rig.cameras[cam_name]

        print(f"\n  Frustum-culled validation (camera: {cam_name}):")

        frustum = FrustumParams.from_intrinsics(
            fx=cam.intrinsics.fx, fy=cam.intrinsics.fy,
            cx=cam.intrinsics.cx, cy=cam.intrinsics.cy,
            width=cam.intrinsics.width, height=cam.intrinsics.height,
        )

        if cam_name in frames[0].sensor_outputs:
            cam_output = frames[0].sensor_outputs[cam_name]
            depth_map = cam_output.data.get("depth")
            viewmat = cam_output.data.get("viewmat")

            lidar_name = list(rig.lidars.keys())[0]
            if lidar_name in frames[0].sensor_outputs:
                lidar_output = frames[0].sensor_outputs[lidar_name].data

                if depth_map is not None and lidar_output["n_valid_points"] > 0:
                    T_cam_world = viewmat
                    results = frustum_validator.validate(
                        synthetic_points=lidar_output["points"],
                        synthetic_depths=lidar_output["depth"],
                        gt_points=lidar_output["points"],  # Self-consistency
                        gt_depths=lidar_output["depth"],
                        T_cam_world=T_cam_world,
                        frustum=frustum,
                    )
                    print(FrustumValidator.generate_report(results))

    # Multi-frame depth consistency
    if len(frames) >= 2 and not args.no_camera:
        cam_name = list(rig.cameras.keys())[0]
        print(f"\n  Multi-frame depth consistency ({cam_name}):")
        depth_diffs = []
        for i in range(1, min(len(frames), 5)):
            if cam_name in frames[i-1].sensor_outputs and cam_name in frames[i].sensor_outputs:
                d1 = frames[i-1].sensor_outputs[cam_name].data.get("depth")
                d2 = frames[i].sensor_outputs[cam_name].data.get("depth")
                if d1 is not None and d2 is not None:
                    valid = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0.5) & (d2 > 0.5)
                    if valid.sum() > 0:
                        diff = np.abs(d1[valid] - d2[valid]).mean()
                        depth_diffs.append(diff)
                        print(f"    Frame {i-1} -> {i}: mean depth diff = {diff:.4f} m")

        if depth_diffs:
            print(f"    Average inter-frame depth change: {np.mean(depth_diffs):.4f} m")

    print("=" * 64)


def run_visualization(frames, rig, args):
    """Generate visualization outputs."""
    print("\n  Generating visualizations...")
    vis = FusionOverlay(depth_range=(0.5, 80.0))
    out_dir = Path(args.output) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_name = list(rig.cameras.keys())[0] if rig.cameras else None
    lidar_name = list(rig.lidars.keys())[0] if rig.lidars else None

    for idx, frame in enumerate(frames[:3]):  # Visualize first 3 frames
        prefix = f"frame_{idx:04d}"

        # Multi-panel visualization
        if cam_name and lidar_name:
            cam_out = frame.sensor_outputs.get(cam_name)
            lidar_out = frame.sensor_outputs.get(lidar_name)

            if cam_out and lidar_out and lidar_out.data["n_valid_points"] > 0:
                cam = rig.cameras[cam_name]
                rgb = cam_out.data["rgb"]
                depth = cam_out.data.get("depth", np.zeros(rgb.shape[:2]))
                points = lidar_out.data["points"]
                intensity = lidar_out.data["intensity"]
                viewmat = cam_out.data["viewmat"]

                vis.render_multi_panel(
                    rgb_image=rgb,
                    depth_map=depth,
                    lidar_points=points,
                    lidar_intensity=intensity,
                    intrinsics=cam.intrinsics.K,
                    T_cam_world=viewmat,
                    ego_position=frame.ego_pose[:3, 3],
                    output_path=str(out_dir / f"{prefix}_multi_panel.png"),
                )
                print(f"    Saved: {prefix}_multi_panel.png")

                # Fusion overlay
                vis.overlay_lidar_on_image(
                    rgb_image=rgb,
                    lidar_points=points,
                    intrinsics=cam.intrinsics.K,
                    T_cam_lidar=viewmat,
                    output_path=str(out_dir / f"{prefix}_fusion.png"),
                )
                print(f"    Saved: {prefix}_fusion.png")

                # BEV
                vis.render_bev(
                    points=points,
                    intensity=intensity,
                    ego_position=frame.ego_pose[:3, 3],
                    output_path=str(out_dir / f"{prefix}_bev.png"),
                )
                print(f"    Saved: {prefix}_bev.png")

        elif cam_name:
            cam_out = frame.sensor_outputs.get(cam_name)
            if cam_out:
                from PIL import Image
                Image.fromarray(cam_out.data["rgb"]).save(
                    str(out_dir / f"{prefix}_camera.png")
                )
                print(f"    Saved: {prefix}_camera.png")

    # Ray drop heatmap (optional — expensive)
    if args.ray_drop_heatmap and lidar_name and rig.lidars:
        print("\n  Computing ray drop heatmap (Monte Carlo, 30 samples)...")
        lidar = list(rig.lidars.values())[0]
        pose = frames[0].ego_pose
        heatmap = lidar.render_ray_drop_heatmap(pose, n_samples=30)
        vis.render_ray_drop_heatmap(
            heatmap,
            output_path=str(out_dir / "ray_drop_heatmap.png"),
        )
        print(f"    Saved: ray_drop_heatmap.png")

    # Depth comparison (deterministic vs stochastic)
    if lidar_name and cam_name and not args.no_camera:
        cam = rig.cameras[cam_name]
        cam_out = frames[0].sensor_outputs.get(cam_name)
        if cam_out and cam_out.data.get("depth") is not None:
            # Render a second depth map with a slightly different pose for comparison
            if len(frames) > 1:
                cam_out2 = frames[1].sensor_outputs.get(cam_name)
                if cam_out2 and cam_out2.data.get("depth") is not None:
                    vis.render_depth_comparison(
                        pred_depth=cam_out.data["depth"],
                        gt_depth=cam_out2.data["depth"],
                        output_path=str(out_dir / "depth_comparison.png"),
                    )
                    print(f"    Saved: depth_comparison.png")


def main():
    args = parse_args()

    if not args.quiet:
        print_header()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    if not args.quiet:
        device = torch.device(args.device)
        if device.type == "cuda":
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  Device: CPU")
        print()

    # --- Phase 1: Load scene ---
    scene = load_scene(args)
    print(f"  Scene loaded: {scene.n_gaussians} Gaussians on {scene.device}")

    # --- Phase 2: Build sensor rig ---
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = apply_config_overrides(config, args)

    # Write modified config back to temp for rig loading
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf)
        tmp_config = tf.name

    rig = VirtualSensorRig.from_config(tmp_config, scene=scene)
    os.unlink(tmp_config)

    print(f"  Sensor rig: {rig}")
    print()

    # --- Phase 3: Load trajectory ---
    trajectory = load_trajectory(args)
    print(f"  Trajectory: {trajectory}")
    print()

    # --- Phase 4: Run simulation ---
    print("Starting batch simulation...")
    out_dir = Path(args.output)

    frames = rig.simulate(
        trajectory,
        output_dir=str(out_dir),
        render_cameras=not args.no_camera,
        render_lidars=not args.no_lidar,
        lidar_stochastic=not args.deterministic,
        verbose=not args.quiet,
    )

    # --- Phase 5: V&V Metrics ---
    run_vv_metrics(frames, rig, args)

    # --- Phase 6: Visualizations ---
    if not args.no_vis:
        run_visualization(frames, rig, args)

    # --- Phase 7: Benchmark ---
    if args.benchmark:
        run_benchmark(rig, trajectory, args)

    # Summary
    print(f"\n  Output saved to: {out_dir.resolve()}")
    print(f"  Frames: {len(frames)}")

    total_cam = sum(
        1 for f in frames
        for s in f.sensor_outputs.values()
        if s.sensor_type == "camera"
    )
    total_lidar = sum(
        1 for f in frames
        for s in f.sensor_outputs.values()
        if s.sensor_type == "lidar"
    )
    print(f"  Camera renders: {total_cam}")
    print(f"  LiDAR renders:  {total_lidar}")
    print("\nDone.")


if __name__ == "__main__":
    main()
