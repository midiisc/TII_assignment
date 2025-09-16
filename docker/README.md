# OpenVINS Docker Environment

This Docker setup provides a complete OpenVINS Visual-Inertial Odometry environment with:

- **CUDA 12.2** - Perfect match with RTX 3080 Laptop GPU
- **ROS2 Humble** - Stable LTS release with full OpenVINS compatibility
- **OpenCV 4.8** - Built with CUDA support and optimized for RTX 3080
- **OpenVINS** - Latest version with all dependencies
- **GPU Acceleration** - Full CUDA compute and memory access

## 🚀 Quick Start

### 1. Build Container
```bash
cd /home/midhun/Documents/TII_assignment
./docker/build.sh
```

### 2. Run Container
```bash
./docker/run.sh
```

### 3. Alternative: Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml exec openvins bash
```

## 📁 Volume Mounts

The container automatically mounts:

- `data/` → `/workspace/data` (read-only) - ROS bag files
- `config/` → `/workspace/config` (read-only) - Camera calibration
- `results/` → `/workspace/results` (read-write) - VIO outputs
- `scripts/` → `/workspace/scripts` (read-only) - Analysis scripts
- `reports/` → `/workspace/reports` (read-write) - Generated reports

## 🎯 Usage Examples

### Test GPU Access
```bash
# Inside container
nvidia-smi
nvcc --version
```

### Run OpenVINS
```bash
# Inside container
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

# Run VIO on ROS bag
ros2 launch ov_msckf pgeneva_ros_eth.launch.py \
    config:=/workspace/config/log_1_ros2_camera_config.yaml \
    bag:=/workspace/data/log_1_ros2
```

### 2D Constrained VIO
```bash
# Inside container
ros2 run ov_msckf run_subscribe_msckf \
    --config=/workspace/config/estimator_config_2d.yaml \
    --bag=/workspace/data/log_1_ros2
```

## 🔧 Configuration

### Camera Configuration
- Located in `/workspace/config/`
- Generated from ROS bag camera info
- Compatible with OpenVINS format

### 2D Constraints
- Modify `estimator_config_2d.yaml`
- Set `use_planar_constraint: true`
- Configure motion model for ground plane

## 📊 Performance

**Tested Configuration:**
- RTX 3080 Laptop GPU (16GB VRAM)
- CUDA 12.2
- Ubuntu 24.04 host
- Docker with NVIDIA runtime

**Expected Performance:**
- Real-time VIO processing
- GPU-accelerated feature extraction
- CUDA-optimized OpenCV operations

## 🐛 Troubleshooting

### GPU Not Accessible
```bash
# Check host GPU
nvidia-smi

# Check Docker NVIDIA support
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu20.04 nvidia-smi
```

### X11 GUI Issues
```bash
# Enable X11 forwarding
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
```

### Build Issues
```bash
# Clean rebuild
docker rmi openvins:cuda12.2-humble
./docker/build.sh
```

## 📚 References

- [OpenVINS Documentation](https://docs.openvins.com/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
