# OpenVINS Docker Environment

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.2.2-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

A **production-ready Docker environment** for **OpenVINS Visual-Inertial Odometry (VIO)** development with maximum performance optimizations, CUDA 12.2.2 support, and comprehensive tooling.

## 🎯 Features

### 🚀 **Performance Optimized**
- **Custom OpenCV 4.8.0** built from source with CUDA acceleration
- **Optimized Ceres Solver** with OpenBLAS, METIS, OpenMP, and TBB
- **Advanced compiler optimizations**: `-O3 -march=native -ffast-math -flto`
- **Memory-optimized compilation** with safe parallel job limits
- **TCMalloc integration** for superior memory allocation performance

### 🛠️ **Development Ready**
- **Complete ROS 2 Humble** desktop environment
- **Comprehensive tooling**: SSH, screen, ncdu, iftop, jq, git-lfs, FlameGraph
- **GPU acceleration** with NVIDIA Docker support
- **User permission handling** with dynamic UID/GID mapping
- **Optimized storage**: tmpfs for `/tmp`, persistent volumes in home directory

### 🔧 **Advanced Capabilities**
- **Multi-threading support**: OpenMP, MPI, TBB integration
- **Profiling tools**: perf, FlameGraph for performance analysis
- **Build caching**: Docker BuildKit with persistent cache mounts
- **Validation scripts**: Build verification and optimization checking
- **Custom configurations**: Pre-configured for mbuggy VIO data

## 📋 Requirements

### System Requirements
- **Docker** with BuildKit support
- **NVIDIA Docker** runtime (for GPU support)
- **Linux** (tested on Ubuntu 22.04)
- **16GB+ RAM** (recommended for compilation)
- **50GB+ disk space** (for build cache and images)

### Hardware Requirements
- **NVIDIA GPU** with CUDA 12.2.2 support (RTX 3080 tested)
- **Multi-core CPU** (AMD Ryzen 9 5900HX optimized)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/midiisc/TII_assignment.git
cd TII_assignment
```

### 2. Build Docker Image
```bash
./docker/build.sh
```

### 3. Run Container
```bash
./docker/run.sh
```

### 4. Test OpenVINS
```bash
# Inside container
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
ros2 pkg list | grep ov_
```

## 📁 Project Structure

```
TII_assignment/
├── docker/
│   ├── Dockerfile.openvins     # Main Dockerfile with optimizations
│   ├── build.sh               # Build script with fallback methods
│   ├── run.sh                 # Run script with GPU and volume support
│   └── docker-compose.yml     # Docker Compose configuration
├── config/
│   ├── custom_estimator_config.yaml  # OpenVINS config for mbuggy data
│   └── log_*_camera_config.yaml      # Camera calibration configs
├── scripts/
│   ├── run_openvins.sh        # OpenVINS launch helper
│   ├── validate_optimizations.sh     # Optimization verification
│   ├── validate_build_paths.sh       # Build path validation
│   └── optimization_summary.sh       # Performance summary
├── data/                      # ROS bag data directory
├── reports/                   # Analysis reports
└── OPTIMIZATION_REPORT.md     # Detailed optimization documentation
```

## 🔧 Configuration

### Docker Build Arguments
```bash
# Customize build parameters
export CUDA_ARCH="8.6"          # Your GPU compute capability
export OPENCV_VERSION="4.8.0"   # OpenCV version
export CERES_VERSION="2.1.0"    # Ceres Solver version

./docker/build.sh
```

### Volume Mounting
The container uses optimized storage:
- **Persistent data**: `/home/midhun/docker/volumes/openvins_*`
- **Project files**: Read-only mounts from host
- **Temporary files**: tmpfs (RAM-based) for `/tmp`

### GPU Configuration
```bash
# Verify GPU support
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

## 🎯 OpenVINS Usage

### Running VIO Pipeline
```bash
# Launch container
./docker/run.sh

# Inside container - run OpenVINS with custom config
./scripts/run_openvins.sh

# Or manually
ros2 run ov_msckf run_subscribe_msckf /workspace/config/custom_estimator_config.yaml
```

### Available ROS Topics
- `/mbuggy/imu_ins` - IMU data
- `/mbuggy/camera_front/image_rect` - Rectified camera images
- `/mbuggy/camera_front/camera_info` - Camera calibration

### Performance Monitoring
```bash
# Inside container - check optimization status
./scripts/validate_optimizations.sh

# Performance profiling with FlameGraph
perf record -g ./your_openvins_command
perf script | stackcollapse-perf.pl | flamegraph.pl > performance.svg
```

## 🛠️ Development

### Build Validation
```bash
# Validate build paths and linking
./scripts/validate_build_paths.sh

# Check library optimization
./scripts/verify_library_paths.sh
```

### Custom Configuration
Edit `config/custom_estimator_config.yaml` for your specific:
- Camera calibration parameters
- IMU noise characteristics
- VIO algorithm settings
- ROS topic names

### Adding New Features
1. Modify `docker/Dockerfile.openvins`
2. Update build arguments in `docker/build.sh`
3. Test with `./docker/build.sh`
4. Validate with provided scripts

## 📊 Performance Optimizations

### Compilation Optimizations
- **GCC Flags**: `-O3 -march=native -mtune=native -ffast-math -funroll-loops`
- **Link-Time Optimization**: `-flto` (where compatible)
- **Parallel Compilation**: Memory-safe job limits prevent OOM errors

### Runtime Optimizations
- **OpenBLAS**: High-performance linear algebra
- **TBB**: Intel Threading Building Blocks for parallelism
- **OpenMP**: Multi-core processing support
- **TCMalloc**: Google's optimized memory allocator

### Library Integration
- **Ceres Solver**: OpenBLAS + METIS + OpenMP threading
- **OpenCV**: CUDA + OpenBLAS + TBB + IPP optimizations
- **OpenVINS**: All acceleration libraries linked and verified

## 🔍 Troubleshooting

### Common Issues

#### Build Errors
```bash
# Out of memory during compilation
# Solution: Reduce parallel jobs in Dockerfile (already implemented)

# CUDA compatibility issues
# Solution: Verify CUDA_ARCH matches your GPU
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### Container Issues
```bash
# Permission problems
# Solution: Container automatically maps host UID/GID

# GPU not accessible
# Solution: Install nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

#### OpenVINS Runtime Issues
```bash
# Config file not found
# Solution: Use absolute paths
ros2 run ov_msckf run_subscribe_msckf /workspace/config/custom_estimator_config.yaml

# No ROS topics
# Solution: Check bag file is playing
ros2 bag play /workspace/data/your_bag.bag
```

### Debug Mode
```bash
# Build with debug information
docker build --build-arg CMAKE_BUILD_TYPE=Debug -f docker/Dockerfile.openvins -t openvins:debug .

# Run with debug tools
docker run -it --rm --gpus all openvins:debug gdb
```

## 📈 Performance Metrics

### Build Performance
- **Build time**: ~18 minutes (with cache)
- **Image size**: 14.9GB (development environment)
- **Cache efficiency**: 95%+ on rebuilds

### Runtime Performance
- **OpenVINS throughput**: Optimized for real-time VIO
- **Memory usage**: TCMalloc reduces allocation overhead
- **CPU utilization**: Multi-threading across all available cores

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** your changes with provided validation scripts
4. **Submit** a pull request with detailed description

### Development Guidelines
- Maintain Docker best practices
- Update documentation for new features
- Validate optimizations with provided scripts
- Test on multiple GPU architectures when possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenVINS Team** - Original VIO framework
- **NVIDIA** - CUDA toolkit and Docker runtime
- **ROS Community** - ROS 2 ecosystem
- **OpenCV Contributors** - Computer vision library

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/midiisc/TII_assignment/issues)
- **Documentation**: Check `OPTIMIZATION_REPORT.md` for detailed technical information
- **Validation**: Use provided scripts in `scripts/` directory

---

**🎯 Ready for high-performance Visual-Inertial Odometry development!**