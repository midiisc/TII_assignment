#!/bin/bash

# OpenVINS Docker Run Script
# This script runs the OpenVINS container with proper GPU support and volume mounts

set -e

echo "🚀 Starting OpenVINS Container..."

# Check if container image exists
if ! docker images | grep -q "openvins.*cuda12.2-humble"; then
    echo "❌ Error: OpenVINS container not found"
    echo "   Run: ./docker/build.sh first"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker/docker-compose.yml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Set up X11 forwarding for GUI applications
echo "🖥️  Setting up X11 forwarding..."
xhost +local:docker > /dev/null 2>&1 || echo "⚠️  X11 forwarding may not work"

# Run container with all necessary mounts and GPU support
echo "🐳 Starting container with GPU support..."
docker run -it --rm \
    --name openvins_vio \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$(pwd)/data:/workspace/data:ro" \
    -v "$(pwd)/config:/workspace/config:ro" \
    -v "$(pwd)/results:/workspace/results:rw" \
    -v "$(pwd)/scripts:/workspace/scripts:ro" \
    -v "$(pwd)/reports:/workspace/reports:rw" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/dev/shm:/dev/shm" \
    --device="/dev/dri:/dev/dri" \
    --network=host \
    --workdir=/workspace \
    openvins:cuda12.2-humble \
    /bin/bash

echo "🏁 Container session ended"
