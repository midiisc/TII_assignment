#!/bin/bash

# OpenVINS Docker Build Script
# This script builds the OpenVINS Docker container with CUDA support

set -e

echo "🐳 Building OpenVINS Docker Container..."
echo "📁 Working directory: $(pwd)"

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

# Check NVIDIA Docker support (skip test if no CUDA images available)
if docker images | grep -q nvidia/cuda; then
    CUDA_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep nvidia/cuda | head -1)
    echo "🔍 Testing NVIDIA support with available image: $CUDA_IMAGE"
    if ! docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi > /dev/null 2>&1; then
        echo "❌ Error: NVIDIA Docker support not working"
        exit 1
    fi
else
    echo "⚠️  No CUDA images available for NVIDIA test - skipping test"
    echo "   NVIDIA support will be tested during the build process"
fi

echo "✅ Pre-checks passed"

# Build the container with optimizations
echo "🔨 Building OpenVINS container with AMD Ryzen 9 5900HX optimizations..."
echo "🚀 Using Docker BuildKit for parallel builds..."
echo "🎯 Applied optimizations:"
echo "   - AMD znver3 architecture optimizations"
echo "   - OpenVINS cmake flags cleaned (removed ignored flags)"
echo "   - FastDDS shared memory transport"
echo "   - OpenBLAS, TBB, OpenMP integration"
echo "   - ROS2 performance optimizations"

# Enable BuildKit and build with optimizations
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Method 1: Multi-line with proper continuations
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

# Method 2: Single line (alternative if line continuation issues persist)
# docker build --progress=plain --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg CUDA_ARCH=${CUDA_ARCH:-"8.6"} --build-arg OPENCV_VERSION=${OPENCV_VERSION:-"4.8.0"} --build-arg CERES_VERSION=${CERES_VERSION:-"2.1.0"} --target builder -f docker/Dockerfile.openvins -t openvins:development .

echo "✅ Development build completed successfully!"
echo "🚀 Container ready: openvins:development"

# Show final image size
echo "📊 Final image size:"
docker images | grep openvins

echo ""
echo "🎯 Next steps:"
echo "   1. Run: ./docker/run.sh"
echo "   2. Or use: docker-compose -f docker/docker-compose.yml up -d"