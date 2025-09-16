#!/bin/bash

# OpenVINS Production Docker Build Script
# This script builds the production-optimized OpenVINS container (multi-stage)

set -e

echo "🚀 Building Production OpenVINS Container..."
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
if ! docker run --rm --gpus all nvidia/cuda:12.2.2-devel-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: NVIDIA Docker support not working"
    exit 1
fi

echo "✅ Pre-checks passed"

# Build arguments for flexibility
CUDA_ARCH=${CUDA_ARCH:-"8.6"}
OPENCV_VERSION=${OPENCV_VERSION:-"4.8.0"}
CERES_VERSION=${CERES_VERSION:-"2.1.0"}

echo "🔧 Build Configuration:"
echo "  - CUDA Architecture: ${CUDA_ARCH}"
echo "  - OpenCV Version: ${OPENCV_VERSION}"
echo "  - Ceres Version: ${CERES_VERSION}"
echo ""

# Build the production container with optimizations
echo "🔨 Building Production OpenVINS container (multi-stage build)..."
echo "🚀 Using Docker BuildKit with cache mounts..."
echo "⏱️  Expected time: 15-20 minutes with accelerations"

# Enable BuildKit and build with optimizations
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

docker build \
    --progress=plain \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg CUDA_ARCH=${CUDA_ARCH} \
    --build-arg OPENCV_VERSION=${OPENCV_VERSION} \
    --build-arg CERES_VERSION=${CERES_VERSION} \
    --target runtime \
    -f docker/Dockerfile.openvins \
    -t openvins:production \
    .

echo "✅ Production build completed successfully!"
echo "🚀 Container ready: openvins:production"

# Show final image sizes
echo "📊 Image size comparison:"
docker images | grep openvins

echo ""
echo "🎯 Production image features:"
echo "  ✅ Multi-stage optimized (smaller size)"
echo "  ✅ Runtime dependencies only"
echo "  ✅ No build tools in final image"
echo "  ✅ CUDA 12.2 + OpenVINS ready"
echo ""
echo "🏃 To run production container:"
echo "   docker run -it --gpus all openvins:production"
echo ""
echo "🔧 To run development container (with build tools):"
echo "   docker build --target builder -t openvins:dev ."
echo "   docker run -it --gpus all openvins:dev"
