#!/usr/bin/env python3
"""
Quick A/B test: render one frame WITH and WITHOUT dynamic actors,
then compare RGB, depth, LiDAR point counts, and BEV to verify
actors are actually appearing in the scene.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.camera import GaussianSplatScene, VirtualCamera, CameraIntrinsics
from engine.lidar import VirtualLiDAR, LiDARConfig
from engine.sensor_rig import VirtualSensorRig
from engine.actor import ScenarioManager, ActorGaussians, ActorTrajectory
from engine.trajectory import Trajectory, Pose

device = "cpu"

# 1. Create a simple synthetic scene
print("=== Actor Injection A/B Test ===\n")
scene = GaussianSplatScene.create_synthetic(n_gaussians=30000, device=device)
print(f"Background scene: {scene.n_gaussians} Gaussians")

# 2. Create scenario with actors
scenario = ScenarioManager.from_config("configs/scenario_demo.yaml", device=device)
print(f"Scenario: {scenario}\n")

# 3. Set up camera + LiDAR using proper YAML config (correct extrinsics)
import yaml, tempfile, os
rig = VirtualSensorRig.from_config("configs/kitti_rig.yaml", scene=scene)

# Override LiDAR to smaller for speed
for name, lidar in rig.lidars.items():
    lidar.config.channels = 16
    lidar.config.points_per_channel = 256
    lidar._ray_dirs, lidar._ray_origins = lidar._generate_ray_pattern()

cam_name = list(rig.cameras.keys())[0]
lidar_name = list(rig.lidars.keys())[0]
cam = rig.cameras[cam_name]
lidar = rig.lidars[lidar_name]
print(f"Camera: {cam_name}, LiDAR: {lidar_name}")

# 4. Define ego pose (identity = origin, looking forward)
T_ego = np.eye(4, dtype=np.float64)
timestamp = 0.5  # mid-scenario

# 5. Render WITHOUT actors
print("--- Rendering WITHOUT actors ---")
cam_result_no = cam.render(T_ego, return_depth=True)
lidar_result_no = lidar.render(T_ego, stochastic=False)
print(f"  Camera depth range: [{cam_result_no['depth'][cam_result_no['depth'] > 0].min():.2f}, "
      f"{cam_result_no['depth'].max():.2f}] m")
print(f"  LiDAR valid points: {lidar_result_no['n_valid_points']}")

# 6. Merge actors into scene and render WITH actors
print("\n--- Rendering WITH actors ---")
transformed_actors = scenario.get_actors_at(timestamp)
merged_scene = scene.merge_actors(transformed_actors)
print(f"  Merged scene: {merged_scene.n_gaussians} Gaussians "
      f"(+{merged_scene.n_gaussians - scene.n_gaussians} from actors)")

# Swap scene temporarily
cam.scene = merged_scene
lidar.scene = merged_scene

cam_result_with = cam.render(T_ego, return_depth=True)
lidar_result_with = lidar.render(T_ego, stochastic=False)
print(f"  Camera depth range: [{cam_result_with['depth'][cam_result_with['depth'] > 0].min():.2f}, "
      f"{cam_result_with['depth'].max():.2f}] m")
print(f"  LiDAR valid points: {lidar_result_with['n_valid_points']}")

# Restore
cam.scene = scene
lidar.scene = scene

# 7. Compare
print("\n=== Comparison ===")

# RGB difference
rgb_no = cam_result_no['rgb'].astype(np.float32)
rgb_with = cam_result_with['rgb'].astype(np.float32)
rgb_diff = np.abs(rgb_with - rgb_no)
mean_rgb_diff = rgb_diff.mean()
max_rgb_diff = rgb_diff.max()
changed_pixels = (rgb_diff.sum(axis=-1) > 5).sum()
total_pixels = rgb_diff.shape[0] * rgb_diff.shape[1]
print(f"  RGB mean diff:    {mean_rgb_diff:.2f}")
print(f"  RGB max diff:     {max_rgb_diff:.0f}")
print(f"  Changed pixels:   {changed_pixels}/{total_pixels} ({100*changed_pixels/total_pixels:.1f}%)")

# Depth difference
depth_no = cam_result_no['depth']
depth_with = cam_result_with['depth']
valid = (depth_no > 0) & (depth_with > 0)
depth_diff = np.abs(depth_with - depth_no)
print(f"  Depth mean diff:  {depth_diff[valid].mean():.3f} m")
print(f"  Depth max diff:   {depth_diff[valid].max():.3f} m")

# LiDAR point count difference
lidar_diff = lidar_result_with['n_valid_points'] - lidar_result_no['n_valid_points']
print(f"  LiDAR point diff: {lidar_diff:+d}")

# Check where actors should be
print("\n=== Actor Positions at t={:.1f}s ===".format(timestamp))
for actor in transformed_actors:
    print(f"  {actor.name}: center={actor.center}, "
          f"velocity={actor.velocity}, "
          f"n_gaussians={actor.means.shape[0]}")

# 8. Save side-by-side comparison
try:
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Dynamic Actor Injection — A/B Comparison", fontsize=14)

    axes[0, 0].imshow(cam_result_no['rgb'])
    axes[0, 0].set_title("Camera WITHOUT actors")

    axes[0, 1].imshow(cam_result_with['rgb'])
    axes[0, 1].set_title("Camera WITH actors")

    diff_vis = np.clip(rgb_diff * 3, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(diff_vis)
    axes[0, 2].set_title(f"Difference (x3) — {changed_pixels} changed px")

    axes[1, 0].imshow(depth_no, cmap='viridis', vmin=0, vmax=80)
    axes[1, 0].set_title("Depth WITHOUT actors")

    axes[1, 1].imshow(depth_with, cmap='viridis', vmin=0, vmax=80)
    axes[1, 1].set_title("Depth WITH actors")

    depth_diff_vis = depth_diff.copy()
    depth_diff_vis[~valid] = 0
    axes[1, 2].imshow(depth_diff_vis, cmap='hot', vmin=0, vmax=depth_diff_vis.max())
    axes[1, 2].set_title("Depth Difference")

    for ax in axes.flat:
        ax.axis('off')

    out_path = "output_actor_test/actor_AB_comparison.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved comparison: {out_path}")

    # BEV comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("BEV — Actor Injection Comparison", fontsize=14)

    pts_no = lidar_result_no['points']
    pts_with = lidar_result_with['points']

    axes2[0].scatter(pts_no[:, 0], pts_no[:, 1], s=1, c='blue', alpha=0.5)
    axes2[0].set_title(f"WITHOUT actors ({lidar_result_no['n_valid_points']} pts)")
    axes2[0].set_xlim(-40, 40); axes2[0].set_ylim(-40, 40)
    axes2[0].set_aspect('equal')
    axes2[0].set_xlabel("X (forward)"); axes2[0].set_ylabel("Y (left)")

    axes2[1].scatter(pts_with[:, 0], pts_with[:, 1], s=1, c='blue', alpha=0.5)
    # Mark actor positions
    for actor in transformed_actors:
        axes2[1].plot(actor.center[0], actor.center[1], 'r*', markersize=12)
        axes2[1].annotate(actor.name, actor.center[:2], fontsize=7, color='red')
    axes2[1].set_title(f"WITH actors ({lidar_result_with['n_valid_points']} pts)")
    axes2[1].set_xlim(-40, 40); axes2[1].set_ylim(-40, 40)
    axes2[1].set_aspect('equal')
    axes2[1].set_xlabel("X (forward)")

    # Overlay both
    axes2[2].scatter(pts_no[:, 0], pts_no[:, 1], s=1, c='blue', alpha=0.3, label='without')
    axes2[2].scatter(pts_with[:, 0], pts_with[:, 1], s=1, c='red', alpha=0.3, label='with')
    for actor in transformed_actors:
        axes2[2].plot(actor.center[0], actor.center[1], 'g*', markersize=15)
        axes2[2].annotate(actor.name, actor.center[:2], fontsize=7, color='green')
    axes2[2].set_title("Overlay (blue=without, red=with)")
    axes2[2].set_xlim(-40, 40); axes2[2].set_ylim(-40, 40)
    axes2[2].set_aspect('equal')
    axes2[2].legend(fontsize=8)
    axes2[2].set_xlabel("X (forward)")

    plt.tight_layout()
    plt.savefig("output_actor_test/actor_BEV_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved BEV comparison: output_actor_test/actor_BEV_comparison.png")

except Exception as e:
    print(f"  Visualization failed: {e}")

# 9. Diagnostic: render LiDAR with actors ONLY (no background)
print("\n=== Diagnostic: Actor-Only LiDAR ===")
actor_only_scene = GaussianSplatScene(
    means=torch.cat([a.means for a in transformed_actors]),
    scales=torch.cat([a.scales for a in transformed_actors]),
    rotations=torch.cat([a.rotations for a in transformed_actors]),
    opacities=torch.cat([a.opacities for a in transformed_actors]),
    sh_coeffs=torch.cat([a.sh_coeffs for a in transformed_actors]),
    sh_degree=scene.sh_degree,
    device=device,
)
print(f"  Actor-only scene: {actor_only_scene.n_gaussians} Gaussians")
lidar.scene = actor_only_scene
lidar_result_actor_only = lidar.render(T_ego, stochastic=False)
print(f"  LiDAR valid points (actors only): {lidar_result_actor_only['n_valid_points']}")
if lidar_result_actor_only['n_valid_points'] > 0:
    pts = lidar_result_actor_only['points']
    print(f"  Point range: x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], "
          f"y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}], "
          f"z=[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")
    print(f"  Depth range: [{lidar_result_actor_only['depth'].min():.1f}, "
          f"{lidar_result_actor_only['depth'].max():.1f}] m")
lidar.scene = scene  # restore

# Check actor Gaussian properties
print("\n=== Actor Gaussian Properties ===")
for actor in transformed_actors:
    scales_exp = torch.exp(actor.scales)
    avg_var = (scales_exp ** 2).mean(dim=1)
    print(f"  {actor.name}:")
    print(f"    n_gaussians: {actor.means.shape[0]}")
    print(f"    center: {actor.center}")
    print(f"    mean scale (exp): {scales_exp.mean(dim=0).tolist()}")
    print(f"    avg_var: mean={avg_var.mean():.4f}, min={avg_var.min():.4f}, max={avg_var.max():.4f}")
    print(f"    opacity: mean={actor.opacities.mean():.3f}, min={actor.opacities.min():.3f}")

print("\n=== VERDICT ===")
if changed_pixels > 100 and mean_rgb_diff > 0.5:
    print("  PASS: Actors are clearly affecting the rendered image.")
else:
    print("  WARNING: Actors may not be visible in camera renders.")

if abs(lidar_diff) > 10:
    print(f"  PASS: LiDAR detects actor presence ({lidar_diff:+d} points).")
else:
    print("  WARNING: LiDAR point count barely changed — actors may not be detectable.")

print("\nDone.")
