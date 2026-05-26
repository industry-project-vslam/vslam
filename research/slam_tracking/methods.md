# Methods for tracking camera position in SLAM
- Summary of [A Comprehensive Survey of Visual SLAM Algorithms](https://www.mdpi.com/2218-6581/11/1/24)

## Visual-mapping
- Visual-only: no sensors, but mono image. Not accurate enaugh due to blur
- Visual-inertial: visual + accelerometer + gyroscope
- RGB-D: depth + visual

Due to onboard telemetry being purely relative (Accelerometer, gyroscope) the telemetry-based position tracking is not feasible due to error accumulation. Beacon positioning is not available due to hardware limitations.
These limitations force us to use visual-only approach.

### Repositories tested:
- [MonoNav](https://github.com/natesimon/MonoNav): Is fast and mapping works, but position tracking is unreliable and requires abs position.
- [MAST3R_SLAM](https://github.com/rmurai0610/MASt3R-SLAM): Depth estimation model is too large. Unable to launch on our machines.
- [MonoGS](https://github.com/muskie82/MonoGS): Is Gaussian splatting mapping project. Is too noisy and handles position tracking badly.
- [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R): Is slow. Position tracking is the best of all.

**SLAM3R** is the repo of choice for our project. It is modified to handle multiple agents on one map capturing frames live.