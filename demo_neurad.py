#!/usr/bin/env python3
"""
Virtual Sensor Suite — NeuRAD/SplatAD Demo

Renders a full driving sequence using the trained SplatAD model with
CUDA-accelerated rasterization, then runs V&V metrics and visualizations.

Usage:
    python demo_neurad.py --config-path /path/to/config.yml --output ./output_kitti
    python demo_neurad.py --config-path /path/to/config.yml --n-frames 20 --output ./output_kitti
"""

import argparse
import time
import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from engine.neurad_backend import NeuradBackend
from metrics.depth_error import DepthErrorMetric
from metrics.frustum_validation import FrustumValidator, FrustumParams
from visualization.fusion_overlay import FusionOverlay


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Sensor Suite — NeuRAD Demo")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to neurad-studio config.yml")
    parser.add_argument("--output", type=str, default="./output_kitti",
                        help="Output directory")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Number of frames to render (default: all)")
    parser.add_argument("--skip-vis", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--camera-idx", type=int, default=0,
                        help="Camera sensor index (0=image_02, 1=image_03)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Virtual Sensor Suite — NeuRAD/SplatAD Rendering Engine")
    print("  CUDA-Accelerated 3DGS with Learned Decoders")
    print("=" * 64)
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {gpu_name}")
        print(f"  VRAM:   {gpu_mem:.1f} GB")
    print()

    # Load model
    print(f"Loading SplatAD model from: {args.config_path}")
    backend = NeuradBackend(args.config_path)
    print(f"  Gaussians: {backend.n_gaussians:,}")
    print(f"  Train cameras: {backend.get_train_camera_count()}")
    print()

    # Determine frames to render
    total_train = backend.get_train_camera_count()
    # Train cameras include both image_02 and image_03 interleaved
    # Sensor idx 0 = image_02, sensor idx 1 = image_03
    # Total per camera = total_train / 2
    frames_per_cam = total_train // 2
    n_frames = min(args.n_frames or frames_per_cam, frames_per_cam)

    # Camera indices: sensor 0 is first half, sensor 1 is second half
    # (or interleaved depending on dataparser). Let's figure it out.
    cam_indices = []
    sensor_idxs = backend._train_cameras.metadata["sensor_idxs"].squeeze(-1)
    for i in range(total_train):
        if sensor_idxs[i] == args.camera_idx:
            cam_indices.append(i)
    cam_indices = cam_indices[:n_frames]

    print(f"Rendering {n_frames} frames from camera {args.camera_idx}...")
    print()

    # Create output directories
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "depth").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    # Render loop
    rgb_frames = []
    depth_frames = []
    render_times = []

    for fi, cam_idx in enumerate(cam_indices):
        t0 = time.time()
        result = backend.render_train_camera(cam_idx)
        dt = time.time() - t0
        render_times.append(dt)

        rgb = result["rgb"]
        depth = result["depth"]
        alpha = result["alpha"]

        # Save outputs
        Image.fromarray(rgb).save(output_dir / "images" / f"{fi:06d}.png")
        np.save(output_dir / "depth" / f"{fi:06d}.npy", depth)

        rgb_frames.append(rgb)
        depth_frames.append(depth)

        if fi % 20 == 0 or fi == n_frames - 1:
            fps = 1.0 / dt if dt > 0 else 0
            print(f"  [{fi+1}/{n_frames}] {dt*1000:.0f}ms ({fps:.1f} FPS)")

    render_times = np.array(render_times)
    # Skip first frame (warmup/JIT compilation)
    if len(render_times) > 1:
        steady_times = render_times[1:]
    else:
        steady_times = render_times

    print()
    print("=" * 60)
    print("  Rendering Performance")
    print("=" * 60)
    print(f"  Frames rendered: {n_frames}")
    print(f"  Mean render time: {steady_times.mean()*1000:.1f} ms")
    print(f"  Std:              {steady_times.std()*1000:.1f} ms")
    print(f"  FPS:              {1.0/steady_times.mean():.1f}")
    print(f"  Min:              {steady_times.min()*1000:.1f} ms")
    print(f"  Max:              {steady_times.max()*1000:.1f} ms")
    print("=" * 60)

    # Save performance report
    with open(output_dir / "performance.txt", "w") as f:
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Mean render time: {steady_times.mean()*1000:.1f} ms\n")
        f.write(f"FPS: {1.0/steady_times.mean():.1f}\n")
        f.write(f"Std: {steady_times.std()*1000:.1f} ms\n")
        f.write(f"Min: {steady_times.min()*1000:.1f} ms\n")
        f.write(f"Max: {steady_times.max()*1000:.1f} ms\n")
        f.write(f"Gaussians: {backend.n_gaussians}\n")

    # Visualizations
    if not args.skip_vis and n_frames >= 3:
        print()
        print("Generating visualizations...")
        overlay = FusionOverlay()

        # Multi-frame depth consistency
        print()
        print("  Multi-frame depth consistency:")
        for i in range(min(n_frames - 1, 5)):
            d1 = depth_frames[i]
            d2 = depth_frames[i + 1]
            valid = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0) & (d2 > 0)
            if valid.sum() > 0:
                diff = np.abs(d1[valid] - d2[valid]).mean()
                print(f"    Frame {i} -> {i+1}: mean depth diff = {diff:.4f} m")

        # Save sample visualizations
        for fi in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
            if fi < n_frames:
                rgb = rgb_frames[fi]
                depth = depth_frames[fi]

                # Depth visualization
                valid_depth = depth[np.isfinite(depth) & (depth > 0)]
                if len(valid_depth) > 0:
                    d_min, d_max = np.percentile(valid_depth, [2, 98])
                    depth_vis = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
                    depth_vis[~np.isfinite(depth)] = 0
                    depth_colored = (plt_colormap(depth_vis) * 255).astype(np.uint8)

                    # Side-by-side: RGB | Depth
                    combined = np.concatenate([rgb, depth_colored[:, :, :3]], axis=0)
                    Image.fromarray(combined).save(
                        output_dir / "visualizations" / f"frame_{fi:04d}_rgb_depth.png"
                    )
                    print(f"    Saved: frame_{fi:04d}_rgb_depth.png")

    print()
    print(f"  Output saved to: {output_dir}")
    print(f"  Frames: {n_frames}")
    print()
    print("Done.")


def plt_colormap(values: np.ndarray) -> np.ndarray:
    """Apply a turbo-like colormap to a 0-1 normalized array."""
    # Simple turbo approximation
    r = np.clip(1.0 - 2.0 * np.abs(values - 0.75), 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(values - 0.5), 0, 1)
    b = np.clip(1.0 - 2.0 * np.abs(values - 0.25), 0, 1)
    return np.stack([r, g, b, np.ones_like(values)], axis=-1)


if __name__ == "__main__":
    main()
