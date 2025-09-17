#!/bin/bash

# OpenVINS Launch Script for mbuggy data
# This script runs OpenVINS with the custom configuration

set -e

echo "🚀 Starting OpenVINS VIO Pipeline..."
echo "📁 Using config: /workspace/config/custom_estimator_config.yaml"
echo "📊 Data topics: /mbuggy/imu_ins, /mbuggy/camera_front/image_rect"
echo "💾 Results will be saved to: /workspace/results/"
echo ""

# Check if ROS bag is playing
echo "🔍 Checking for active ROS topics..."
if ! ros2 topic list | grep -q "/mbuggy/imu_ins"; then
    echo "❌ Error: IMU topic /mbuggy/imu_ins not found"
    echo "   Make sure to run: ros2 bag play /workspace/data/log_1_ros2/ --loop"
    echo "   in another terminal first"
    exit 1
fi

if ! ros2 topic list | grep -q "/mbuggy/camera_front/image_rect"; then
    echo "❌ Error: Camera topic /mbuggy/camera_front/image_rect not found"
    echo "   Make sure to run: ros2 bag play /workspace/data/log_1_ros2/ --loop"
    echo "   in another terminal first"
    exit 1
fi

echo "✅ All required topics found!"

# Ensure results directory exists
mkdir -p /workspace/results

# Run OpenVINS
echo "🎯 Launching OpenVINS MSCKF..."
echo "   Press Ctrl+C to stop"
echo ""

ros2 run ov_msckf run_subscribe_msckf /workspace/config/custom_estimator_config.yaml

echo ""
echo "🏁 OpenVINS finished!"
echo "📊 Results saved to: /workspace/results/"
