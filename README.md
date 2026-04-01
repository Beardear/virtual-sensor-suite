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
│   ├── neurad_backend.py           # NeuRAD/SplatAD CUDA rendering backend (camera + LiDAR)
│   └── trajectory.py               # Trajectory loader (timestamped SE(3) poses)
├── metrics/
│   ├── depth_error.py              # L1 / MAE / RMSE / delta depth metrics
│   └── frustum_validation.py       # Frustum-culled V&V comparison
├── visualization/
│   └── fusion_overlay.py           # LiDAR→Camera overlay, BEV, depth comparison, ray drop heatmap
├── demo.py                         # CLI entry point (synthetic scenes)
└── demo_neurad.py                  # CLI entry point (trained SplatAD model)
```

---

## How It Works

This project has two rendering paths:

1. **Pure Python renderer** (`demo.py`) — A from-scratch 3DGS renderer with Monte Carlo LiDAR ray-casting. Works on synthetic Gaussian scenes with no external dependencies. Good for demonstrating the math (splatting, volumetric rendering, ray-drop modeling).

2. **NeuRAD/SplatAD backend** (`demo_neurad.py`) — Wraps a trained [SplatAD](https://github.com/carlinds/splatad) model for photorealistic rendering of real driving scenes. Uses CUDA-accelerated gsplat rasterization + learned CNN decoder (camera) and MLP decoder (LiDAR). This is the production-grade path.

### Rendering Pipeline

```
KITTI Driving Sequence
        │
        ▼
   SplatAD Training (neurad-studio)
        │
        ▼
   Trained 3DGS Scene (5M Gaussians)
        │
        ├─── Camera Renderer ──► RGB + Depth  (58 FPS, CUDA gsplat + CNN decoder)
        │
        └─── LiDAR Renderer ──► 3D Point Cloud + Intensity + Ray Drop
                                 (gsplat lidar_rasterization + MLP decoder)
        │
        ▼
   V&V Metrics (vs ground-truth KITTI LiDAR)
        │
        ▼
   Sensor Fusion Visualization (LiDAR→Camera overlay, BEV)
```

---

## External Dependencies

This engine is designed to sit **on top of** a trained 3DGS reconstruction. The following external repositories are used:

### [SplatAD](https://github.com/carlinds/splatad) (CVPR 2025)
- **What it is:** CUDA-accelerated 3D Gaussian Splatting for autonomous driving, with joint camera + LiDAR rendering
- **How we use it:** The gsplat CUDA rasterization library — provides `rasterization()` and `lidar_rasterization()` kernels that our `NeuradBackend` calls at render time
- **Paper:** [SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving](https://arxiv.org/abs/2411.16816)

### [neurad-studio](https://github.com/georghess/neurad-studio)
- **What it is:** The training pipeline for SplatAD/NeuRAD models — includes dataparsers (KITTI, nuScenes, Waymo), model definitions, training loop, and evaluation
- **How we use it:** We train a SplatAD model on KITTI data using neurad-studio, then load the trained checkpoint via `eval_setup()` in our `NeuradBackend`. The trained model contains 5M Gaussians + learned RGB decoder (CNN) + learned LiDAR decoder (MLP)
- **Our engine wraps the trained model** — it doesn't retrain or modify the architecture, but adds config-driven sensor rig management, trajectory-based batch simulation, V&V metrics, and fusion visualization on top

### [KITTI Multi-Object Tracking Dataset](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
- **What it is:** Real-world driving data with stereo cameras, Velodyne HDL-64E LiDAR, GPS/IMU, and object tracking annotations
- **How we use it:** Sequence 0006 (270 frames) as training data for the 3DGS scene reconstruction, and as ground truth for V&V metrics

---

## Bug Fixes for neurad-studio

When integrating with neurad-studio, we discovered and fixed three issues. Apply these before training:

### 1. Missing Velodyne HDL-64E elevation mapping
neurad-studio had beam mappings for HDL-32E, VLP-32C, Velodyne-128, and Pandar-64, but **not** the HDL-64E used in KITTI.

**File:** `nerfstudio/data/utils/lidar_elevation_mappings.py`
```python
# Add this mapping:
VELODYNE_HDL64E_ELEVATION_MAPPING = dict(
    zip(
        np.arange(64),
        tuple(np.concatenate([
            np.linspace(2.0, -8.33, 32),    # upper block (beams 0-31)
            np.linspace(-8.53, -24.33, 32),  # lower block (beams 32-63)
        ])),
    )
)
```

**File:** `nerfstudio/cameras/lidars.py`
```python
# Add to imports:
from nerfstudio.data.utils.lidar_elevation_mappings import VELODYNE_HDL64E_ELEVATION_MAPPING

# In get_lidar_elevation_mapping(), add the VELODYNE64E case:
elif lidar_type == LidarType.VELODYNE64E:
    return VELODYNE_HDL64E_ELEVATION_MAPPING
```

### 2. PyTorch 2.6 checkpoint loading
PyTorch 2.6 defaults to `weights_only=True` in `torch.load()`, which rejects numpy scalars in the checkpoint.

**File:** `nerfstudio/utils/eval_utils.py`
```python
# Change:
loaded_state = torch.load(load_path, map_location="cpu")
# To:
loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)
```

### 3. KITTI tracking timestamps
The KITTI tracking dataset doesn't include per-sensor timestamps (those are in the raw dataset, with no official tracking-to-raw mapping). Use `--use-sensor-timestamps False` when training — the dataparser falls back to uniform 10 Hz timing, which has negligible impact on quality (~2.5cm positioning error).

---

## Quick Start

### Synthetic scene (no training needed)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml scipy matplotlib open3d tqdm pillow

python demo.py --synthetic --n-frames 20 --output ./output
```

### Trained SplatAD model (real KITTI data)
```bash
# 1. Clone and install neurad-studio
git clone https://github.com/georghess/neurad-studio.git
cd neurad-studio && pip install -e .

# 2. Download KITTI MOT data (sequence 0006)
#    - data_tracking_image_2.zip, data_tracking_image_3.zip
#    - data_tracking_velodyne.zip, data_tracking_calib.zip
#    - data_tracking_label_2.zip, data_tracking_oxts.zip

# 3. Apply the bug fixes above

# 4. Train SplatAD
ns-train splatad \
  --output-dir ./outputs \
  --vis tensorboard \
  --max-num-iterations 30001 \
  --pipeline.model.implementation torch \
  kittimot-data \
  --data /path/to/kittimot \
  --sequence 0006 \
  --use-sensor-timestamps False

# 5. Render with virtual sensor suite
python demo_neurad.py \
  --config-path ./outputs/unnamed/splatad/<timestamp>/config.yml \
  --output ./output_kitti \
  --n-frames 135
```

---

## Key Features

### Config-Driven Sensor Rig
Define your full sensor layout in YAML — camera intrinsics/extrinsics, LiDAR channels/range/FoV, radar placeholder. One ego pose in, all sensors fire synchronously.

```python
scene = GaussianSplatScene.create_synthetic()
rig = VirtualSensorRig.from_config("configs/kitti_rig.yaml", scene=scene)
trajectory = Trajectory.generate_synthetic(n_frames=100, speed_mps=10.0)
sensor_log = rig.simulate(trajectory, output_dir="./output/")
```

### NeuRAD Backend (Real Data)
Wraps a trained SplatAD model for CUDA-accelerated rendering:
```python
from engine.neurad_backend import NeuradBackend

backend = NeuradBackend("/path/to/config.yml")
result = backend.render_train_camera(0)   # RGB + depth
lidar  = backend.render_train_lidar(0)    # 3D points + intensity + ray drop
```

### Monte Carlo LiDAR Ray-Casting
Virtual laser rays march through the Gaussian scene via volumetric rendering. Stochastic ray dropping models beam penetration through soft matter (vegetation, rain, dust) using learned `ray_drop_prob`.

### V&V Metrics
- **Depth Error:** L1/MAE/RMSE, absolute/squared relative error, delta thresholds (1.25, 1.25^2, 1.25^3)
- **Frustum-Culled Validation:** Crops ground-truth LiDAR to the virtual camera's exact FoV before computing error — spatially meaningful validation

### Sensor Fusion Visualization
- LiDAR-to-Camera projection overlay (colored by depth)
- Bird's Eye View (BEV) point cloud (colored by height and intensity)
- Depth map comparison with error heatmap
- Ray drop probability heatmap

---

## Results

### Rendering Performance (RTX 4090)
| Sensor | Resolution | FPS | Gaussians |
|--------|-----------|-----|-----------|
| Camera (RGB + Depth) | 1242x375 | 58.8 | 5,000,000 |
| LiDAR (64 beams, 116K pts) | 64x4000 | ~15 | 5,000,000 |
| Camera + LiDAR combined | — | ~15 | 5,000,000 |

### V&V: Rendered LiDAR vs KITTI Ground Truth

Evaluated on **held-out frames** (model never saw these during training):

| Metric | Full Scene | Frustum Culled |
|--------|-----------|----------------|
| MAE | 6.6 cm | 10.9 cm |
| RMSE | 12.0 cm | 16.2 cm |
| Median AE | 3.2 cm | 7.5 cm |
| Abs Rel Error | 0.47% | 0.59% |
| delta < 1.25 | 100.0% | 100.0% |
| Matched points (avg) | 115,226 | 17,688 |

---

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

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NumPy, SciPy, PyYAML, matplotlib, Pillow, tqdm, open3d
- For real data: [neurad-studio](https://github.com/georghess/neurad-studio) + KITTI MOT dataset

## Future Work

- **Scene editing:** Insert/remove dynamic agents (pedestrians, vehicles) by manipulating Gaussian clusters — enabling counterfactual scenario generation (e.g. "what if a pedestrian stepped into the lane?")
- **Radar simulation:** Extend the engine with material-based radar cross-section (RCS) estimation on Gaussians, completing the full AV sensor suite (camera + LiDAR + radar)
- **Closed-loop integration:** Feed synthetic sensor output into a perception stack (e.g. PointPillars, CenterPoint) and evaluate downstream task performance — measuring detection mAP on synthetic vs real data to quantify simulation-to-real transfer
- **Novel view synthesis:** Render from arbitrary shifted/modified camera poses (not just training views) for data augmentation and safety-critical edge case generation
- **Multi-sequence generalization:** Train on multiple KITTI sequences and evaluate cross-sequence rendering fidelity

## References

- [SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving](https://arxiv.org/abs/2411.16816) (CVPR 2025)
- [NeuRAD: Neural Rendering for Autonomous Driving](https://github.com/georghess/neurad-studio)
- [KITTI Multi-Object Tracking Benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (SIGGRAPH 2023)
