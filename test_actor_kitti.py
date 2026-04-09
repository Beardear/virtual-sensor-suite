#!/usr/bin/env python3
"""
Test dynamic actor injection on KITTI data via NeuRAD/SplatAD backend.

Renders training frame WITH and WITHOUT injected actors, then compares.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.neurad_backend import NeuradBackend
from engine.actor import ScenarioManager, ActorGaussians, ActorTrajectory

CONFIG_PATH = "/workspace/SplatAD/outputs/unnamed/splatad/2026-04-01_020926/config.yml"

print("=" * 64)
print("  Actor Injection Test — KITTI / SplatAD")
print("=" * 64)

# Load model
print("\nLoading SplatAD model...")
backend = NeuradBackend(CONFIG_PATH)
print(f"  Gaussians: {backend.n_gaussians:,}")

# Pick a training frame (mid-sequence)
total = backend.get_train_camera_count()
sensor_idxs = backend._train_cameras.metadata["sensor_idxs"].squeeze(-1)
cam0_indices = [i for i in range(total) if sensor_idxs[i] == 0]
frame_idx = cam0_indices[len(cam0_indices) // 2]  # mid-sequence
print(f"  Test frame: {frame_idx} (of {total})")

# 1. Render WITHOUT actors
print("\nRendering WITHOUT actors...")
result_no = backend.render_train_camera(frame_idx)
print(f"  RGB shape: {result_no['rgb'].shape}")
print(f"  Depth range: [{result_no['depth'][result_no['depth'] > 0].min():.2f}, "
      f"{result_no['depth'].max():.2f}]")

# 2. Create actors in SCENE frame
# We need to figure out where to place actors in scene coordinates.
# Let's check the mean position of the scene Gaussians.
model = backend.model
means = model.gauss_params["means"].data
print(f"\n  Scene Gaussian stats:")
print(f"    means range: x=[{means[:,0].min():.1f}, {means[:,0].max():.1f}], "
      f"y=[{means[:,1].min():.1f}, {means[:,1].max():.1f}], "
      f"z=[{means[:,2].min():.1f}, {means[:,2].max():.1f}]")
print(f"    means center: [{means[:,0].mean():.1f}, {means[:,1].mean():.1f}, {means[:,2].mean():.1f}]")

# Get the camera pose for this frame to know where the ego is
cam = backend._train_cameras[frame_idx : frame_idx + 1]
c2w = cam.camera_to_worlds[0].cpu().numpy()  # (3, 4)
cam_pos = c2w[:3, 3]
cam_forward = c2w[:3, 2]  # z-axis in nerfstudio = forward
print(f"\n  Camera position (scene frame): {cam_pos}")
print(f"  Camera forward (scene frame): {cam_forward}")

# Place actors relative to camera position, in the camera's forward direction
# Actor 1: A vehicle 10m ahead
vehicle_pos = cam_pos + cam_forward * 10.0
# Actor 2: A vehicle 20m ahead, slightly to the right
vehicle2_pos = cam_pos + cam_forward * 20.0 + np.cross(cam_forward, [0, 0, 1]) * 3.0

print(f"\n  Vehicle 1 position: {vehicle_pos}")
print(f"  Vehicle 2 position: {vehicle2_pos}")

# Create synthetic actors
device = "cuda"
actor1 = ActorGaussians.create_synthetic_vehicle(
    name="test_vehicle_1", color=(0.8, 0.1, 0.1),  # bright red
    n_gaussians=500, device=device,
)
actor2 = ActorGaussians.create_synthetic_vehicle(
    name="test_vehicle_2", color=(0.1, 0.1, 0.8),  # bright blue
    n_gaussians=500, device=device,
)

# Create transforms to place them in scene frame
T1 = np.eye(4, dtype=np.float64)
T1[:3, 3] = vehicle_pos
T2 = np.eye(4, dtype=np.float64)
T2[:3, 3] = vehicle2_pos

transformed1 = actor1.transform(T1)
transformed2 = actor2.transform(T2)

print(f"\n  Actor 1: {transformed1.means.shape[0]} Gaussians at {transformed1.center}")
print(f"  Actor 2: {transformed2.means.shape[0]} Gaussians at {transformed2.center}")

# 3. Render WITH actors
print("\nRendering WITH actors...")
result_with = backend.render_train_camera_with_actors(
    frame_idx, [transformed1, transformed2]
)
print(f"  Model restored: {backend.model.num_points == backend.n_gaussians}")

# 4. Compare
rgb_no = result_no['rgb'].astype(np.float32)
rgb_with = result_with['rgb'].astype(np.float32)
rgb_diff = np.abs(rgb_with - rgb_no)
mean_diff = rgb_diff.mean()
max_diff = rgb_diff.max()
changed_pixels = (rgb_diff.sum(axis=-1) > 10).sum()
total_pixels = rgb_diff.shape[0] * rgb_diff.shape[1]

print(f"\n=== Comparison ===")
print(f"  RGB mean diff:    {mean_diff:.2f}")
print(f"  RGB max diff:     {max_diff:.0f}")
print(f"  Changed pixels:   {changed_pixels}/{total_pixels} ({100*changed_pixels/total_pixels:.1f}%)")

# Depth comparison
d_no = result_no['depth']
d_with = result_with['depth']
valid = np.isfinite(d_no) & np.isfinite(d_with) & (d_no > 0) & (d_with > 0)
if valid.sum() > 0:
    depth_diff = np.abs(d_with[valid] - d_no[valid])
    print(f"  Depth mean diff:  {depth_diff.mean():.4f} m")
    print(f"  Depth max diff:   {depth_diff.max():.4f} m")
    # How many pixels have closer depth (actor occluding background)?
    closer = (d_with[valid] < d_no[valid] - 0.5).sum()
    print(f"  Pixels with closer depth: {closer}")

# 5. Save comparison image
try:
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    fig.suptitle("KITTI Actor Injection — A/B Comparison", fontsize=14)

    axes[0, 0].imshow(result_no['rgb'])
    axes[0, 0].set_title("WITHOUT actors")

    axes[0, 1].imshow(result_with['rgb'])
    axes[0, 1].set_title("WITH actors")

    diff_vis = np.clip(rgb_diff * 3, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(diff_vis)
    axes[0, 2].set_title(f"Difference (x3) — {changed_pixels} px")

    axes[1, 0].imshow(d_no, cmap='viridis', vmin=0, vmax=np.percentile(d_no[d_no > 0], 95))
    axes[1, 0].set_title("Depth WITHOUT")

    axes[1, 1].imshow(d_with, cmap='viridis', vmin=0, vmax=np.percentile(d_no[d_no > 0], 95))
    axes[1, 1].set_title("Depth WITH")

    depth_diff_img = np.abs(d_with - d_no)
    depth_diff_img[~valid] = 0
    axes[1, 2].imshow(depth_diff_img, cmap='hot')
    axes[1, 2].set_title("Depth Difference")

    for ax in axes.flat:
        ax.axis('off')

    out = "output_actor_test/kitti_actor_AB_comparison.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {out}")

except Exception as e:
    print(f"\n  Visualization error: {e}")
    import traceback
    traceback.print_exc()

# Verdict
print("\n=== VERDICT ===")
if changed_pixels > 100 and mean_diff > 1.0:
    print("  PASS: Actors are affecting the KITTI render.")
else:
    print("  FAIL: Actors are NOT visible in the KITTI render.")
print("\nDone.")
