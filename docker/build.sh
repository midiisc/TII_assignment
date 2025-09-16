#!/bin/bash

# OpenVINS Docker Build Script
# This script builds the OpenVINS Docker container with CUDA support

set -e

echo "🐳 Building OpenVINS Docker Container..."
echo "📍 Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "docker/Dockerfile.openvins" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   Expected: /home/midhun/Documents/TII_assignment"
    exit 1
fi

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Check NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: NVIDIA Docker support not working"
    exit 1
fi

echo "✅ Pre-checks passed"

# Build the container with optimizations
echo "🔨 Building OpenVINS container with acceleration (15-20 minutes with optimizations)..."
echo "🚀 Using Docker BuildKit for parallel builds..."

# Enable BuildKit and build with optimizations
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

docker build \
    --progress=plain \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg CUDA_ARCH=${CUDA_ARCH:-"8.6"} \
    --build-arg OPENCV_VERSION=${OPENCV_VERSION:-"4.8.0"} \
    --build-arg CERES_VERSION=${CERES_VERSION:-"2.1.0"} \
    --target builder \
    -f docker/Dockerfile.openvins \
    -t openvins:development \
    .

echo "✅ Development build completed successfully!"
echo "🚀 Container ready: openvins:development"

# Show final image size
echo "📊 Final image size:"
docker images | grep openvins

echo ""
echo "🎯 Next steps:"
echo "   1. Run: ./docker/run.sh"
echo "   2. Or use: docker-compose -f docker/docker-compose.yml up -d"
