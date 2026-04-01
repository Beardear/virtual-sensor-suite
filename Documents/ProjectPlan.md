# Probabilistic Virtual Sensor Suite for Closed-Loop AV Testing

## Target Role
**Machine Learning Engineer - 3D Sensor Simulation @ Zoox (Foster City, CA)**

## Objective
Build a lightweight, config-driven sensor simulation engine on top of SplatAD's 3DGS CUDA rasterizer. The demo should prove the ability to design modular, production-grade simulation tooling — not just train a model.

---

## Architecture

```
virtual-sensor-suite/
├── configs/
│   └── kitti_rig.yaml              # Sensor layout definition (camera, LiDAR, radar placeholder)
├── engine/
│   ├── __init__.py
│   ├── sensor_rig.py               # VirtualSensorRig: unified multi-sensor class
│   ├── camera.py                   # Virtual camera renderer (novel view synthesis)
│   ├── lidar.py                    # Monte Carlo ray-casting LiDAR with ray_drop_prob
│   └── trajectory.py               # Trajectory loader (timestamped poses, KITTI format)
├── metrics/
│   ├── depth_error.py              # L1 / MAE depth error vs ground truth
│   └── frustum_validation.py       # Frustum-culled V&V comparison
├── visualization/
│   └── fusion_overlay.py           # Project LiDAR points onto camera image (sensor fusion proof)
├── demo.py                         # Main entry point / interactive demo
└── README.md
```

---

## Components

### 1. Base Reconstruction (Pre-requisite)
- **Action:** Clone SplatAD repo, install dependencies, download KITTI data, train on a single dynamic sequence.
- **JD alignment:** "Research, implement, and optimize state of the art 3D rendering of sensor data, leveraging GenAI/ML and 3D graphics."
- **Proves:** Ability to handle real-world AV data logs and use PyTorch to optimize explicit 3D geometry.

### 2. Config-Driven Sensor Rig
- **Action:** Define a `VirtualSensorRig` class that reads a YAML config specifying the full sensor layout (camera intrinsics/extrinsics, LiDAR channels/range/extrinsics, radar placeholder). One ego pose in, all sensors fire synchronously.
- **JD alignment:** "Collaborate with Perception and Safety teams" — a config-driven API is what downstream teams actually consume. Also maps to the hiring manager's note that their team maintains a sensor simulation engine.
- **Proves:** Senior-level software architecture and production-system thinking.

**Example config (`kitti_rig.yaml`):**
```yaml
sensors:
  - name: front_camera
    type: camera
    intrinsics: {fx: 721.5, fy: 721.5, cx: 609.5, cy: 172.8}
    extrinsics: {x: 1.5, y: 0.0, z: 1.6, roll: 0, pitch: 0, yaw: 0}
    resolution: [1242, 375]

  - name: roof_lidar
    type: lidar
    channels: 64
    range_m: 80.0
    points_per_channel: 1024
    extrinsics: {x: 0.0, y: 0.0, z: 1.8, roll: 0, pitch: 0, yaw: 0}

  - name: front_radar  # placeholder — demonstrates awareness of full sensor suite
    type: radar
    extrinsics: {x: 2.0, y: 0.0, z: 0.8, roll: 0, pitch: 0, yaw: 0}
    # TODO: radar reflectivity model (material-based RCS estimation)
```

### 3. Virtual Camera Renderer (Novel View Synthesis)
- **Action:** Render RGB images from arbitrary camera poses by computing new extrinsics relative to the trained Gaussian Splat scene. Example: shift camera 1m right, or simulate a bumper-mount vs roof-mount.
- **JD alignment:** "Familiarity with 3D graphics algorithms, such as 3D geometry and camera models."
- **Proves:** Understanding of 3D linear algebra, camera intrinsics, and coordinate frame transformations (T_world = T_car x T_sensor).

### 4. Virtual LiDAR via Monte Carlo Ray-Casting (Core Differentiator)
- **Action:** Fire virtual laser rays into the Gaussian Splat scene. Sample the `ray_drop_prob` learned by SplatAD to stochastically model beam penetration through soft matter (vegetation, rain, dust). Output synthetic `.pcd` point cloud files with intensity values.
- **JD alignment:** "Strong mathematical skills and understanding of... probabilistic techniques."
- **Proves:** Understanding of LiDAR physics, probabilistic modeling, and the ability to bridge ML outputs with physically-motivated simulation. This is the "hire me" feature.
- **Visualization:** Render a ray drop probability heatmap over the scene; compare stochastic output vs deterministic depth-only rendering.

### 5. Trajectory-Based Batch Simulation
- **Action:** Accept a `Trajectory` (sequence of timestamped ego poses, loaded from KITTI format) and generate a synchronized sensor log across all frames. Write results to disk in a structured output directory.
- **JD alignment:** "Improve rendering and ML inference tooling for generating realistic data at scale."
- **Proves:** Scalability thinking; the demo isn't single-frame — it generates full drive sequences.

**Interface:**
```python
trajectory = Trajectory.from_kitti(sequence_id="0001")
rig = VirtualSensorRig.from_config("configs/kitti_rig.yaml", scene=trained_scene)
sensor_log = rig.simulate(trajectory, output_dir="./output/")
```

### 6. Sensor Fusion Visualization
- **Action:** Project the synthetic 3D LiDAR point cloud onto the synthetic 2D camera image, colored by depth. This is the universal "Hello World" of sensor fusion calibration.
- **JD alignment:** Hiring manager's team works on sensor fusion. This directly demonstrates spatial alignment between virtual sensors.
- **Proves:** Mathematical modeling of virtual sensors is consistent across modalities.

### 7. V&V Metrics
- **Action:** Implement two validation steps:
  1. **L1 / MAE Depth Error:** Compare synthetic LiDAR depth against KITTI ground-truth LiDAR for matched frames.
  2. **Frustum-Culled Comparison:** Crop real KITTI LiDAR to the virtual camera's exact field of view before computing error — spatially meaningful validation.
- **JD alignment:** "Develop realism metrics with V&V to show measurable impact of your improved sensor fidelity."
- **Proves:** Production engineering mindset — quantifying fidelity, not just making pretty pictures.

### 8. Performance Benchmark
- **Action:** Measure and report rendering FPS for both camera and LiDAR. SplatAD claims real-time rendering (order of magnitude faster than NeRF).
- **JD alignment:** "Real-time sensor data for hardware-in-the-loop simulation."
- **Proves:** Awareness that simulation speed matters for closed-loop and HIL testing.

---

## Execution Order

| Phase | Task | Estimated Effort |
|-------|------|-----------------|
| **0** | Clone SplatAD, install deps, download KITTI data | Setup |
| **1** | Train SplatAD on one KITTI dynamic sequence | Overnight training |
| **2** | Implement `VirtualSensorRig` + YAML config loader | Core scaffold |
| **3** | Implement virtual camera renderer | Wraps SplatAD's existing camera rendering |
| **4** | Implement Monte Carlo LiDAR ray-casting | Core novel work |
| **5** | Implement `Trajectory` loader + batch simulation | Integration |
| **6** | Implement fusion overlay visualization | Visualization |
| **7** | Implement V&V metrics (depth error + frustum culling) | Metrics |
| **8** | Benchmark FPS, polish demo script | Polish |

---

## Interview Pitch (One-Liner)
> "I built a lightweight sensor simulation engine on top of SplatAD's CUDA rasterizer. You define a sensor rig in a config file, pass it a driving trajectory, and it outputs synchronized, physically-modeled Camera and LiDAR streams with automated V&V against ground truth — all in real time."

---

## Future Extensions (Mention in Interview, Don't Need to Build)
- **Scene editing:** Insert/remove dynamic agents (pedestrians, vehicles) by manipulating Gaussian clusters — maps to JD bonus: "generative models for 3D content pipelines."
- **Radar simulation:** Extend engine with material-based radar cross-section (RCS) estimation on Gaussians.
- **Closed-loop integration:** Feed synthetic sensor output into a perception stack and evaluate downstream task performance (detection mAP on synthetic vs real data).

---

## Key References
- [SplatAD Paper (CVPR 2025)](https://arxiv.org/abs/2411.16816)
- [SplatAD GitHub](https://github.com/carlinds/splatad)
- [Industrial-Grade Sensor Simulation via Gaussian Splatting](https://arxiv.org/abs/2503.11731)
- [GSAVS: Gaussian Splatting-based AV Simulator](https://arxiv.org/html/2412.18816v1)
