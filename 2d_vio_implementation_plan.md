# 🎯 **2D Visual-Inertial Odometry Implementation Plan**
*Tailored for ROS2 Jazzy with Modern Visual Features*

## **📋 Phase I: Environment Validation & Data Integrity (Day 1)**

### **1.1 System Requirements Check**
```bash
# Verify ROS2 Jazzy environment
source /opt/ros/jazzy/setup.bash
ros2 --version

# Check CUDA compatibility
nvidia-smi
nvcc --version

# Verify OpenCV with CUDA
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}'); print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

### **1.2 ROS Bag Data Integrity Check**
```bash
# Inspect topics
ros2 bag info data/log_1_ros2

# Check camera and IMU topics
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic hz /mbuggy/odom  # Your IMU-equivalent data
ros2 topic echo /camera/image_raw --once
ros2 topic echo /mbuggy/odom --once

# Verify TF tree
rqt_tf_tree
```

### **1.3 Create Data Validation Node**
```python
#!/usr/bin/env python3
\"\"\"
Data synchronization validator for 2D VIO pipeline
\"\"\"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

class DataValidator(Node):
    def __init__(self):
        super().__init__('data_validator')
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/mbuggy/odom', self.odom_callback, 10)
        
    def camera_callback(self, msg):
        self.get_logger().info(f'Camera: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} | Frame: {msg.header.frame_id}')
        
    def odom_callback(self, msg):
        self.get_logger().info(f'Odometry: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} | Frame: {msg.header.frame_id}')

if __name__ == '__main__':
    rclpy.init()
    validator = DataValidator()
    rclpy.spin(validator)
```

## **📋 Phase II: OpenVINS Installation & Setup (Day 1-2)**

### **2.1 Install Dependencies**
```bash
# System dependencies
sudo apt install libeigen3-dev libboost-all-dev libceres-dev

# Create dedicated workspace
mkdir -p ~/vio_ws/src
cd ~/vio_ws/src

# Clone OpenVINS
git clone https://github.com/rpng/open_vins.git
cd open_vins
git checkout master  # Latest stable
```

### **2.2 Build OpenVINS**
```bash
cd ~/vio_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ov_core ov_init ov_msckf --symlink-install

# Source the workspace
source install/setup.bash
```

### **2.3 Install robot_localization for 2D Constraints**
```bash
sudo apt install ros-jazzy-robot-localization
```

## **📋 Phase III: Configuration for Your Data (Day 2-3)**

### **3.1 Camera Configuration**
Create `~/vio_ws/src/open_vins/ov_msckf/config/mbuggy_cam0.yaml`:
```yaml
%YAML:1.0 # need to specify the file type at the top!

cam0:
  cam_k: [987.14, 979.95, 976.47, 581.69]  # From your camera calibration
  cam_d: [-0.286237, 0.061453, -0.000785, -0.001012]  # Distortion coefficients
  resolution: [1920, 1080]
  rostopic: \"/camera/image_raw\"
```

### **3.2 IMU/Odometry Configuration** 
Create `~/vio_ws/src/open_vins/ov_msckf/config/mbuggy_imu0.yaml`:
```yaml
%YAML:1.0

imu0:
  rostopic: \"/mbuggy/odom\"  # Using odometry as pseudo-IMU
  # Noise parameters (to be tuned)
  gyroscope_noise_density: 0.01
  accelerometer_noise_density: 0.1
  gyroscope_random_walk: 0.001
  accelerometer_random_walk: 0.01
```

### **3.3 Main Estimator Configuration**
Create `~/vio_ws/src/open_vins/ov_msckf/config/mbuggy_estimator.yaml`:
```yaml
%YAML:1.0

verbosity: \"INFO\"

use_stereo: false
max_cameras: 1

cam0:
  T_imu_cam: !!opencv-matrix  # Camera-IMU extrinsics (identity for now)
    rows: 4
    cols: 4
    dt: d
    data: [1.0, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0, 0.0,
           0.0, 0.0, 0.0, 1.0]

# 2D-specific settings
init_window_time: 2.0
init_imu_thresh: 0.5
gravity_mag: 9.81

# Feature tracking
num_pts: 200
fast_threshold: 20
grid_x: 5
grid_y: 5
min_px_dist: 15

# Disable online calibration for stability
calib_cam_extrinsics: false
calib_cam_intrinsics: false
calib_cam_timeoffset: false
```

## **📋 Phase IV: 2D Constraint Integration (Day 3-4)**

### **4.1 robot_localization EKF Configuration**
Create `~/vio_ws/config/2d_localization.yaml`:
```yaml
ekf_filter_node:
  ros__parameters:
    # Enable 2D mode
    two_d_mode: true
    
    # Input topics
    odom0: /ov_msckf/odometry
    odom0_config: [true,  true,  false,  # x, y, z
                   false, false, true,   # roll, pitch, yaw  
                   true,  true,  false,  # vx, vy, vz
                   false, false, true,   # vroll, vpitch, vyaw
                   false, false, false]  # ax, ay, az
    
    # Output
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom
    
    # Frequency
    frequency: 30.0
    
    # Process noise
    process_noise_covariance: [0.05, 0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0.05, 0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0.06, 0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0.03, 0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0.03, 0,    0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0.06, 0,     0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0.025, 0,     0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0.025, 0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0.04, 0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0.01, 0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0.01, 0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0.02, 0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0.01, 0,    0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0.01, 0,
                               0,    0,    0,    0,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0,    0.015]
```

## **📋 Phase V: Modern Visual Features Integration (Day 4-5)**

### **5.1 Enhanced Feature Detection Pipeline**
Create custom image preprocessing node:
```python
#!/usr/bin/env python3
\"\"\"
Enhanced image preprocessing with modern features for 2D VIO
\"\"\"
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FeatureEnhancer(Node):
    def __init__(self):
        super().__init__('feature_enhancer')
        self.bridge = CvBridge()
        
        # Modern feature detectors
        self.orb = cv2.ORB_create(nfeatures=500)
        self.akaze = cv2.AKAZE_create()
        self.sift = cv2.SIFT_create(nfeatures=300)
        
        # Subscribers and publishers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.enhanced_pub = self.create_publisher(Image, '/camera/image_enhanced', 10)
        
    def image_callback(self, msg):
        # Convert to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, \"bgr8\")
        
        # 2D-specific preprocessing
        enhanced = self.preprocess_for_2d(cv_image)
        
        # Publish enhanced image
        enhanced_msg = self.bridge.cv2_to_imgmsg(enhanced, \"bgr8\")
        enhanced_msg.header = msg.header
        self.enhanced_pub.publish(enhanced_msg)
        
    def preprocess_for_2d(self, image):
        \"\"\"2D-specific image enhancements\"\"\"
        # Sky masking for outdoor environments
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        
        # Ground plane focus (bottom 2/3 of image)
        h, w = image.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[h//3:, :] = 255  # Focus on ground plane
        
        # Combine masks
        final_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(sky_mask))
        
        # Apply mask
        enhanced = image.copy()
        enhanced[final_mask == 0] = [0, 0, 0]
        
        # Enhance contrast for better features
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
```

## **📋 Phase VI: Testing & Validation (Day 5-6)**

### **6.1 Launch Complete 2D VIO Pipeline**
Create launch file `~/vio_ws/launch/2d_vio_launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Feature enhancer
        Node(
            package='vio_pipeline',
            executable='feature_enhancer.py',
            name='feature_enhancer'
        ),
        
        # OpenVINS estimator
        Node(
            package='ov_msckf',
            executable='run_estimator',
            name='ov_estimator',
            parameters=['/home/midhun/vio_ws/config/mbuggy_estimator.yaml']
        ),
        
        # 2D localization filter
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_2d_filter',
            parameters=['/home/midhun/vio_ws/config/2d_localization.yaml']
        ),
        
        # RViz2 for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/midhun/vio_ws/config/2d_vio.rviz']
        )
    ])
```

### **6.2 Quantitative Evaluation Setup**
```bash
# Install evo for trajectory evaluation
pip install evo --upgrade --no-binary evo

# Record VIO output
ros2 bag record /odometry/filtered -o vio_output

# Evaluate against ground truth
evo_ape tum ground_truth.txt vio_trajectory.txt --plot --save_results
evo_rpe tum ground_truth.txt vio_trajectory.txt --plot --save_results
```

## **📋 Phase VII: Advanced 2D Optimizations (Day 6-7)**

### **7.1 Planar Constraint Integration**
```bash
# Build ov_plane extension for planar constraints
cd ~/vio_ws/src/open_vins
git submodule update --init --recursive
cd ~/vio_ws
colcon build --packages-select ov_plane --symlink-install
```

### **7.2 Performance Monitoring**
```python
#!/usr/bin/env python3
\"\"\"
Real-time 2D VIO performance monitor
\"\"\"
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class VIOMonitor(Node):
    def __init__(self):
        super().__init__('vio_monitor')
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        
        # Performance tracking
        self.positions = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.speeds = deque(maxlen=1000)
        
    def odom_callback(self, msg):
        # Extract 2D position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        
        # Calculate speed
        if len(self.positions) > 1:
            dx = x - self.positions[-2][0]
            dy = y - self.positions[-2][1]
            dt = timestamp - self.timestamps[-2]
            speed = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
            self.speeds.append(speed)
            
            # Log performance metrics
            if len(self.speeds) % 100 == 0:
                avg_speed = np.mean(list(self.speeds)[-100:])
                self.get_logger().info(f'2D VIO Performance - Avg Speed: {avg_speed:.2f} m/s')
```

## **📋 Expected Results & Success Metrics**

### **Success Criteria:**
1. **Initialization**: VIO initializes within 5 seconds
2. **Tracking**: Maintains tracking for >95% of dataset
3. **Accuracy**: APE < 1% of trajectory length
4. **Real-time**: Processing at >20 Hz
5. **2D Constraint**: Z-drift < 0.1m over full trajectory

### **Performance Benchmarks:**
- **Filter-based VIO**: Smooth, continuous trajectory
- **2D Constraints**: Perfect planar motion
- **Modern Features**: Robust tracking in challenging conditions
- **ROS2 Integration**: Native message passing and TF tree

This plan provides a systematic, phase-by-phase approach to implementing a robust 2D VIO pipeline using modern techniques while leveraging your existing ROS2 Jazzy environment and camera calibration data.
