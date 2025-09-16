# Complete ROS Bag Analysis Report

Generated on: 2025-09-16 12:08:08

## Executive Summary

- **Total Bags Analyzed**: 2
- **Total Distance**: 639.0 m
- **Total Duration**: 342.0 s

## log_1_ros2

### Topics and Frequencies
| Topic | Message Type | Count | Frequency (Hz) |
|-------|-------------|-------|----------------|
| /mbuggy/septentrio/localization | nav_msgs/msg/Odometry | 2108 | 50.02 |
| /mbuggy/fix | sensor_msgs/msg/NavSatFix | 2108 | 50.02 |
| /mbuggy/camera_front/camera_info | sensor_msgs/msg/CameraInfo | 1148 | 27.24 |
| /mbuggy/camera_front/image_rect | sensor_msgs/msg/Image | 1148 | 27.24 |
| /tf | tf2_msgs/msg/TFMessage | 5902 | 140.02 |
| /mbuggy/imu_ins | sensor_msgs/msg/Imu | 2108 | 50.02 |
| /mbuggy/odom | nav_msgs/msg/Odometry | 1686 | 40.03 |
| /mbuggy/navsat/odometry | nav_msgs/msg/Odometry | 843 | 20.02 |

### Trajectory Analysis
- **Total Distance**: 103.2 m
- **Duration**: 42.1 s
- **Elevation Range**: 0.000 m
- **Is Planar Motion**: Yes

### GPS Analysis
- **GPS Coverage**: 0.0%
- **Total Fixes**: 2108
- **No Fix Count**: 2108
- **GPS Fix Count**: 0
- **DGPS Fix Count**: 0
- **RTK Fix Count**: 0

### IMU Analysis
- **Average Acceleration**: 9.86 m/s²
- **Max Acceleration**: 14.90 m/s²
- **Average Angular Velocity**: 0.11 rad/s
- **Max Angular Velocity**: 0.34 rad/s

### Camera Analysis
- **Resolution**: 1920x1080
- **Encoding**: bgr8
- **Total Frames**: 1148
- **Frame Rate**: 27.2 Hz
- **Average Data Size**: 6220800 bytes

## log_0_ros2

### Topics and Frequencies
| Topic | Message Type | Count | Frequency (Hz) |
|-------|-------------|-------|----------------|
| /mbuggy/fix | sensor_msgs/msg/NavSatFix | 14997 | 50.00 |
| /mbuggy/camera_front/camera_info | sensor_msgs/msg/CameraInfo | 7906 | 26.36 |
| /mbuggy/camera_front/image_rect | sensor_msgs/msg/Image | 7906 | 26.36 |
| /mbuggy/imu_ins | sensor_msgs/msg/Imu | 14997 | 50.00 |
| /mbuggy/navsat/odometry | nav_msgs/msg/Odometry | 5999 | 20.00 |
| /mbuggy/odom | nav_msgs/msg/Odometry | 11995 | 40.00 |
| /tf_static | tf2_msgs/msg/TFMessage | 24 | 446.51 |
| /tf | tf2_msgs/msg/TFMessage | 41972 | 140.00 |
| /mbuggy/septentrio/localization | nav_msgs/msg/Odometry | 14990 | 50.00 |

### Trajectory Analysis
- **Total Distance**: 535.7 m
- **Duration**: 299.9 s
- **Elevation Range**: 0.000 m
- **Is Planar Motion**: Yes

### GPS Analysis
- **GPS Coverage**: 0.0%
- **Total Fixes**: 14997
- **No Fix Count**: 14997
- **GPS Fix Count**: 0
- **DGPS Fix Count**: 0
- **RTK Fix Count**: 0

### IMU Analysis
- **Average Acceleration**: 9.85 m/s²
- **Max Acceleration**: 16.06 m/s²
- **Average Angular Velocity**: 0.10 rad/s
- **Max Angular Velocity**: 0.54 rad/s

### Camera Analysis
- **Resolution**: 1920x1080
- **Encoding**: bgr8
- **Total Frames**: 7906
- **Frame Rate**: 26.4 Hz
- **Average Data Size**: 6220800 bytes

## Recommendations for Visual Localization

### Data Quality Assessment
- ✅ **Excellent camera data** (1920x1080 @ 26-27 Hz)
- ✅ **Good IMU data** (realistic accelerations and angular velocities)
- ✅ **Perfect planar motion** (Z=0 throughout)
- ❌ **No GPS signal** (GPS-denied environment)

### Recommended Approach
- **Visual-Inertial Odometry (VIO)** with 2D motion constraints
- **Use `/mbuggy/odom` as ground truth** for trajectory comparison
- **Implement planar motion assumption** (Z=0 constraint)
- **Focus on visual-inertial fusion** without GPS dependency

### Next Steps
1. **Implement VIO algorithm** (ORB-SLAM3 or OpenVINS)
2. **Setup coordinate frame alignment** using static transforms
3. **Validate against ground truth** from odometry data
4. **Optimize parameters** for outdoor, high-speed scenarios
