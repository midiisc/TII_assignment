#!/bin/bash

# OpenVINS Docker Run Script
# This script runs the container with optimized storage:
# - A bind mount to a specific folder in /home/midhun for persistent data.
# - A tmpfs mount for temporary files to protect the host disk.
# - Full GPU support with CUDA and graphics capabilities.

set -e

# --- Configuration ---
# The full path on your computer for the container's persistent data.
HOST_DATA_PATH="/home/midhun/docker/volumes/openvins_data"
HOST_RESULTS_PATH="/home/midhun/docker/volumes/openvins_results"
HOST_REPORTS_PATH="/home/midhun/docker/volumes/openvins_reports"

# The paths inside the container where persistent data will be stored.
CONTAINER_DATA_PATH="/workspace/data_persistent"
CONTAINER_RESULTS_PATH="/workspace/results"
CONTAINER_REPORTS_PATH="/workspace/reports"

# The Docker image to run.
IMAGE_NAME="openvins:development"

# Container configuration
CONTAINER_NAME="openvins_vio"
TMPFS_SIZE="2G"

# User permissions (prevents root-owned files in home directory)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
# --- End Configuration ---

echo "🚀 Starting OpenVINS Container..."

# Check if container image exists
if ! docker images | grep -q "openvins.*development"; then
    echo "❌ Error: OpenVINS container not found"
    echo "   Run: ./docker/build.sh first"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker/docker-compose.yml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Ensure the host directories for our volumes exist.
echo "✅ Ensuring data directories exist:"
echo "   📂 Data: ${HOST_DATA_PATH}"
echo "   📂 Results: ${HOST_RESULTS_PATH}"
echo "   📂 Reports: ${HOST_REPORTS_PATH}"
mkdir -p "${HOST_DATA_PATH}"
mkdir -p "${HOST_RESULTS_PATH}"
mkdir -p "${HOST_REPORTS_PATH}"

# Set up X11 forwarding for GUI applications
echo "🖥️  Setting up X11 forwarding..."
xhost +local:docker > /dev/null 2>&1 || echo "⚠️  X11 forwarding may not work"

# Run the container with all optimizations.
echo "🐳 Starting container with GPU support and optimized storage..."
docker run -it --rm --gpus all \
    --name "${CONTAINER_NAME}" \
    --network host \
    -e HOST_UID=${HOST_UID} \
    -e HOST_GID=${HOST_GID} \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    # --- ROS2 PERFORMANCE OPTIMIZATIONS ---
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e RCLCPP_LOGGING_BUFFERED_STREAM=1 \
    -e RCLCPP_DISABLE_SIGNAL_HANDLER=1 \
    -e ROS_DOMAIN_ID=0 \
    -e RCLCPP_EXECUTOR_NUM_THREADS=16 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/root/.ros/fastdds_shm.xml \
    -v "$(pwd)/data:/workspace/data:ro" \
    -v "$(pwd)/config:/workspace/config:ro" \
    -v "$(pwd)/scripts:/workspace/scripts:ro" \
    -v "${HOST_DATA_PATH}:${CONTAINER_DATA_PATH}" \
    -v "${HOST_RESULTS_PATH}:${CONTAINER_RESULTS_PATH}" \
    -v "${HOST_REPORTS_PATH}:${CONTAINER_REPORTS_PATH}" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/dev/shm:/dev/shm" \
    --device="/dev/dri:/dev/dri" \
    --workdir=/workspace \
    --tmpfs /tmp:size=${TMPFS_SIZE},noexec,nosuid,nodev \
    "${IMAGE_NAME}" \
    /bin/bash

echo "🏁 Container session ended"
