The raw Gaussians are just a static 3D representation — a cloud of ellipsoids with color and opacity. They're not sensor data.    
                                                                                                                                                    
  Your engine adds the sensor simulation layer on top:                                                                                              
                                                                                                                                                    
  - Camera rendering — Projects Gaussians into a 2D image from any arbitrary viewpoint, not just the original KITTI camera positions. Want to       
  simulate a bumper-mounted camera instead of roof-mounted? Your engine handles the coordinate transform and renders it.
  - LiDAR simulation — Fires virtual laser rays through the Gaussians and returns a point cloud. The raw Gaussians have no concept of "what a LiDAR 
  would see." Your engine does the ray marching, applies range limits, channel patterns, and stochastic ray dropping.                               
  - Config-driven multi-sensor rig — One ego pose in, synchronized camera + LiDAR outputs out, matching whatever sensor layout you define in YAML.
  - Trajectory simulation — Generates full driving sequences, not just single frames.                                                               
  - V&V metrics — Quantifies how close the synthetic output is to ground truth.                                                                     
                                                                                                                                                    
  Without your engine, you'd have trained Gaussians and SplatAD's built-in single-camera renderer. With your engine, you have a production-style    
  sensor simulation pipeline — which is exactly what the Zoox team builds and maintains.