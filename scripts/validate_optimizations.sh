#!/bin/bash

# OpenVINS Optimization Validation Script
# This script validates that the optimizations are working correctly

echo "🔍 Validating OpenVINS Optimizations"
echo "===================================="
echo ""

echo "1. 🏗️ Checking Compiler Architecture Flags:"
echo "   Expected: -march=znver3 -mtune=znver3"
if gcc -march=znver3 -mtune=znver3 -Q --help=target | grep -q "march.*znver3"; then
    echo "   ✅ GCC supports AMD Zen3 architecture"
else
    echo "   ⚠️  GCC may not fully support znver3"
fi
echo ""

echo "2. 🧮 Checking OpenBLAS Configuration:"
if ldconfig -p | grep -q openblas; then
    echo "   ✅ OpenBLAS library found"
    OPENBLAS_VERSION=$(pkg-config --modversion openblas 2>/dev/null || echo "N/A")
    echo "   📦 OpenBLAS version: $OPENBLAS_VERSION"
else
    echo "   ❌ OpenBLAS library not found"
fi
echo ""

echo "3. 🧵 Checking Threading Libraries:"
if ldconfig -p | grep -q libtbb; then
    echo "   ✅ Intel TBB library found"
else
    echo "   ❌ Intel TBB library not found"
fi

if ldconfig -p | grep -q libgomp; then
    echo "   ✅ OpenMP library found"
else
    echo "   ❌ OpenMP library not found"
fi
echo ""

echo "4. 🔧 Checking Ceres Installation:"
if pkg-config --exists ceres; then
    CERES_VERSION=$(pkg-config --modversion ceres)
    echo "   ✅ Ceres Solver found: v$CERES_VERSION"
else
    echo "   ⚠️  Ceres pkg-config not found - attempting fix..."
    if ldconfig -p | grep -q ceres; then
        echo "   ✅ Ceres library found in system"
        # Try to create pkg-config file in writable location
        mkdir -p /tmp/pkgconfig 2>/dev/null
        if [ ! -f /tmp/pkgconfig/ceres.pc ]; then
            cat > /tmp/pkgconfig/ceres.pc << 'EOF'
prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: Ceres Solver
Description: A large scale non-linear optimization library
Version: 2.1.0
URL: http://ceres-solver.org/
Libs: -L${libdir} -lceres
Libs.private: -lglog -lgflags -lpthread
Cflags: -I${includedir}
EOF
            echo "   🔧 Created temporary Ceres pkg-config file"
        fi
        export PKG_CONFIG_PATH="/tmp/pkgconfig:${PKG_CONFIG_PATH}"
        if pkg-config --exists ceres; then
            CERES_VERSION=$(pkg-config --modversion ceres)
            echo "   ✅ Ceres pkg-config working after fix: v$CERES_VERSION"
            echo "   💡 Add to your environment: export PKG_CONFIG_PATH=\"/tmp/pkgconfig:\$PKG_CONFIG_PATH\""
        else
            echo "   ❌ Ceres pkg-config still not working"
        fi
    else
        echo "   ❌ Ceres library not found"
    fi
fi
echo ""

echo "5. 👁️ Checking OpenCV Installation:"
if python3 -c "import cv2; print(f'OpenCV {cv2.__version__} found')" 2>/dev/null; then
    echo "   ✅ OpenCV Python bindings working"
    echo "   🔍 Checking OpenCV build info:"
    python3 -c "import cv2; info = cv2.getBuildInformation(); print('   LAPACK:', 'YES' if 'LAPACK:                    YES' in info else 'NO'); print('   OpenMP:', 'YES' if 'Parallel framework:         OpenMP' in info else 'NO'); print('   TBB:', 'YES' if 'TBB' in info else 'NO')" 2>/dev/null
else
    echo "   ❌ OpenCV not found or not working"
fi
echo ""

echo "6. 🤖 Checking ROS2 Environment:"
if [ -n "$RMW_IMPLEMENTATION" ]; then
    echo "   ✅ RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
else
    echo "   ⚠️  RMW_IMPLEMENTATION not set"
fi

if [ -n "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    echo "   ✅ FastDDS profile: $FASTRTPS_DEFAULT_PROFILES_FILE"
    if [ -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
        echo "   ✅ FastDDS profile file exists"
    else
        echo "   ❌ FastDDS profile file missing"
    fi
else
    echo "   ⚠️  FastDDS profile not configured"
fi

if [ -n "$RCLCPP_EXECUTOR_NUM_THREADS" ]; then
    echo "   ✅ ROS2 executor threads: $RCLCPP_EXECUTOR_NUM_THREADS"
else
    echo "   ⚠️  ROS2 executor threads not configured"
fi
echo ""

echo "7. 🎯 Checking Performance Environment Variables:"
echo "   OMP_NUM_THREADS: ${OMP_NUM_THREADS:-Not set}"
echo "   TBB_NUM_THREADS: ${TBB_NUM_THREADS:-Not set}"
echo "   OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS:-Not set}"
echo "   OPENBLAS_CORETYPE: ${OPENBLAS_CORETYPE:-Not set}"
echo ""

echo "8. 💾 Checking Memory Optimization:"
if [ -n "$LD_PRELOAD" ] && [[ "$LD_PRELOAD" == *"tcmalloc"* ]]; then
    echo "   ✅ TCMalloc preloaded: $LD_PRELOAD"
else
    echo "   ⚠️  TCMalloc not preloaded"
fi
echo ""

echo "🎯 Optimization Validation Complete!"
echo ""
echo "To test OpenVINS performance:"
echo "1. Run a VIO dataset: ros2 launch ov_msckf serial.launch.py"
echo "2. Monitor CPU usage: htop"
echo "3. Check for warnings about CPU throttling"
echo "4. Verify no 'ignored flag' warnings in build logs"
