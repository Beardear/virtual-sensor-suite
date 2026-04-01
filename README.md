# Virtual Sensor Suite

**Probabilistic, config-driven sensor simulation engine built on 3D Gaussian Splatting.**

Renders synchronized Camera and LiDAR streams from arbitrary ego poses through a pre-trained 3DGS scene, with Monte Carlo ray-drop modeling, automated V&V metrics, and sensor fusion visualization.

---

## Architecture

```
virtual-sensor-suite/
├── configs/
│   └── kitti_rig.yaml              # Sensor layout (camera, LiDAR, radar placeholder)
├── engine/
│   ├── sensor_rig.py               # VirtualSensorRig: config-driven multi-sensor orchestrator
│   ├── camera.py                   # 3DGS camera renderer (novel view synthesis)
│   ├── lidar.py                    # Monte Carlo ray-casting LiDAR with ray_drop_prob
│   └── trajectory.py               # Trajectory loader (timestamped SE(3) poses)
├── metrics/
│   ├── depth_error.py              # L1 / MAE / RMSE / delta depth metrics
│   └── frustum_validation.py       # Frustum-culled V&V comparison
├── visualization/
│   └── fusion_overlay.py           # LiDAR→Camera overlay, BEV, depth comparison, ray drop heatmap
└── demo.py                         # CLI entry point with benchmarking
```

## Quick Start

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml scipy matplotlib open3d tqdm pillow

# Run with synthetic scene (no training/data needed)
python demo.py --synthetic --n-frames 20 --output ./output

# With trained SplatAD checkpoint
python demo.py --model-path /path/to/checkpoint.pth --config configs/kitti_rig.yaml

# Performance benchmark
python demo.py --synthetic --benchmark --n-frames 50

# Reduced LiDAR for faster iteration
python demo.py --synthetic --n-frames 10 --lidar-channels 32 --lidar-ppc 512 --curve-radius 50
```

## Key Features

### Config-Driven Sensor Rig
Define your full sensor layout in YAML — camera intrinsics/extrinsics, LiDAR channels/range/FoV, radar placeholder. One ego pose in, all sensors fire synchronously.

```python
scene = GaussianSplatScene.create_synthetic()
rig = VirtualSensorRig.from_config("configs/kitti_rig.yaml", scene=scene)
trajectory = Trajectory.generate_synthetic(n_frames=100, speed_mps=10.0)
sensor_log = rig.simulate(trajectory, output_dir="./output/")
```

### Monte Carlo LiDAR Ray-Casting
Virtual laser rays march through the Gaussian scene via volumetric rendering. Stochastic ray dropping models beam penetration through soft matter (vegetation, rain, dust) using learned `ray_drop_prob`.

### Novel View Synthesis
Renders RGB + depth from arbitrary camera poses using per-Gaussian splatting with:
- Full 3D covariance → 2D projection via Jacobian
- Spherical harmonics for view-dependent color
- Front-to-back alpha compositing

### V&V Metrics
- **Depth Error:** L1/MAE/RMSE, absolute/squared relative error, delta thresholds (1.25, 1.25^2, 1.25^3)
- **Frustum-Culled Validation:** Crops ground-truth LiDAR to the virtual camera's exact FoV before computing error

### Sensor Fusion Visualization
- LiDAR → Camera projection overlay (colored by depth)
- Bird's Eye View (BEV) point cloud
- Depth map comparison with error heatmap
- Ray drop probability heatmap

## Output Structure (KITTI format)

```
output/
├── image_00/         # Front camera RGB (.png)
├── image_01/         # Rear camera RGB (.png)
├── depth_00/         # Front camera depth (.npy)
├── velodyne/         # LiDAR point clouds (.bin, KITTI format)
├── pointclouds/      # LiDAR point clouds (.pcd)
├── visualizations/   # Multi-panel, fusion, BEV, depth comparison
├── poses.txt         # Ego poses (KITTI 3x4 format)
├── timestamps.txt    # Frame timestamps
└── performance.txt   # Rendering FPS report
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--synthetic` | Use procedural 3DGS scene (no data needed) |
| `--model-path` | Path to trained SplatAD checkpoint |
| `--config` | Sensor rig YAML config |
| `--n-frames` | Number of trajectory frames |
| `--speed` | Ego speed (m/s) |
| `--curve-radius` | Circular trajectory radius (None = straight) |
| `--lidar-channels` | Override LiDAR channels |
| `--lidar-ppc` | Override points-per-channel |
| `--ray-drop-prob` | Monte Carlo ray drop probability |
| `--deterministic` | Disable stochastic ray dropping |
| `--benchmark` | Run isolated FPS benchmarks |
| `--ray-drop-heatmap` | Generate ray drop heatmap (Monte Carlo) |

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NumPy, SciPy, PyYAML, matplotlib, Pillow, tqdm, open3d
