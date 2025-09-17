#!/bin/bash

echo "🔍 COMPREHENSIVE BUILD PATH VALIDATION"
echo "======================================"
echo ""

# Track validation status
VALIDATION_PASSED=true

# Function to validate library path
validate_lib() {
    local lib_name="$1"
    local lib_path="$2"
    local description="$3"
    
    echo -n "  $description: "
    if [ -f "$lib_path" ]; then
        echo "✅ FOUND at $lib_path"
    else
        echo "❌ NOT FOUND at $lib_path"
        # Try to find the correct path
        echo "    🔍 Searching for correct path..."
        local found_path=$(find /usr -name "$(basename $lib_path)*" -type f 2>/dev/null | head -1)
        if [ -n "$found_path" ]; then
            echo "    💡 Found at: $found_path"
        else
            echo "    ❌ Library not found anywhere"
        fi
        VALIDATION_PASSED=false
    fi
}

# Function to validate header path
validate_header() {
    local header_name="$1"
    local header_path="$2"
    local description="$3"
    
    echo -n "  $description: "
    if [ -f "$header_path" ]; then
        echo "✅ FOUND at $header_path"
    else
        echo "❌ NOT FOUND at $header_path"
        # Try to find the correct path
        echo "    🔍 Searching for correct path..."
        local found_path=$(find /usr -name "$(basename $header_path)" -type f 2>/dev/null | head -1)
        if [ -n "$found_path" ]; then
            echo "    💡 Found at: $found_path"
        else
            echo "    ❌ Header not found anywhere"
        fi
        VALIDATION_PASSED=false
    fi
}

echo "1. OpenBLAS Libraries"
echo "===================="
validate_lib "openblas" "/usr/lib/x86_64-linux-gnu/libopenblas.so" "OpenBLAS main library"
validate_lib "blas" "/usr/lib/x86_64-linux-gnu/libblas.so" "BLAS library"
validate_lib "lapack" "/usr/lib/x86_64-linux-gnu/liblapack.so" "LAPACK library"
echo ""

echo "2. OpenBLAS Headers"
echo "=================="
validate_header "cblas.h" "/usr/include/x86_64-linux-gnu/cblas.h" "CBLAS header"
validate_header "lapacke.h" "/usr/include/lapacke.h" "LAPACKE header"
validate_header "openblas_config.h" "/usr/include/x86_64-linux-gnu/openblas-pthread/openblas_config.h" "OpenBLAS config header"
echo ""

echo "3. TBB Libraries"
echo "==============="
validate_lib "tbb" "/usr/lib/x86_64-linux-gnu/libtbb.so" "TBB main library"
validate_lib "tbb12" "/usr/lib/x86_64-linux-gnu/libtbb.so.12" "TBB versioned library"
echo ""

echo "4. OpenMP Libraries"
echo "=================="
validate_lib "gomp" "/usr/lib/x86_64-linux-gnu/libgomp.so.1" "GNU OpenMP (GOMP)"
validate_lib "omp" "/usr/lib/x86_64-linux-gnu/libomp.so.5" "LLVM OpenMP"
echo ""

echo "5. CUDA Libraries"
echo "================"
validate_lib "cudart" "/usr/local/cuda/lib64/libcudart.so" "CUDA Runtime"
validate_lib "cublas" "/usr/local/cuda/lib64/libcublas.so" "cuBLAS"
validate_lib "cusolver" "/usr/local/cuda/lib64/libcusolver.so" "cuSOLVER"
validate_lib "cusparse" "/usr/local/cuda/lib64/libcusparse.so" "cuSPARSE"
echo ""

echo "6. System Libraries"
echo "=================="
validate_lib "m" "/usr/lib/x86_64-linux-gnu/libm.so" "Math library"
validate_lib "pthread" "/usr/lib/x86_64-linux-gnu/libpthread.so.0" "Threading library"
validate_lib "dl" "/usr/lib/x86_64-linux-gnu/libdl.so.2" "Dynamic loader"
validate_lib "rt" "/usr/lib/x86_64-linux-gnu/librt.so.1" "Real-time library"
echo ""

echo "7. Compiler Tools"
echo "================"
echo -n "  GCC: "
which gcc && gcc --version | head -1 || echo "❌ NOT FOUND"
echo -n "  G++: "
which g++ && g++ --version | head -1 || echo "❌ NOT FOUND"
echo -n "  NVCC: "
which nvcc && nvcc --version | grep "release" || echo "❌ NOT FOUND"
echo ""

echo "8. Build Tools"
echo "=============="
echo -n "  CMake: "
which cmake && cmake --version | head -1 || echo "❌ NOT FOUND"
echo -n "  Ninja: "
which ninja && ninja --version || echo "❌ NOT FOUND"
echo -n "  ccache: "
which ccache && ccache --version | head -1 || echo "❌ NOT FOUND"
echo ""

echo "9. Package Verification"
echo "======================"
echo "  OpenBLAS packages:"
dpkg -l | grep -E "(openblas|blas|lapack)" | awk '{print "    " $2 " " $3}'
echo ""
echo "  TBB packages:"
dpkg -l | grep -E "tbb" | awk '{print "    " $2 " " $3}'
echo ""
echo "  OpenMP packages:"
dpkg -l | grep -E "(gomp|omp)" | awk '{print "    " $2 " " $3}'
echo ""

echo "🎯 VALIDATION SUMMARY"
echo "===================="
if [ "$VALIDATION_PASSED" = true ]; then
    echo "✅ ALL PATHS VALIDATED SUCCESSFULLY"
    echo "   Build should proceed without library path errors."
    exit 0
else
    echo "❌ VALIDATION FAILED"
    echo "   Some library paths are incorrect and need to be fixed."
    echo ""
    echo "🔧 RECOMMENDED CMAKE FLAGS BASED ON ACTUAL PATHS:"
    echo "================================================"
    
    # Generate corrected CMake flags
    if [ -f "/usr/lib/x86_64-linux-gnu/libopenblas.so" ]; then
        echo "-D LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \\"
        echo "-D BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \\"
    fi
    
    if [ -f "/usr/include/x86_64-linux-gnu/cblas.h" ]; then
        echo "-D LAPACK_CBLAS_H=/usr/include/x86_64-linux-gnu/cblas.h \\"
    fi
    
    if [ -f "/usr/include/lapacke.h" ]; then
        echo "-D LAPACK_LAPACKE_H=/usr/include/lapacke.h \\"
    fi
    
    if [ -f "/usr/lib/x86_64-linux-gnu/libtbb.so" ]; then
        echo "-D TBB_LIBRARIES=/usr/lib/x86_64-linux-gnu \\"
    fi
    
    echo ""
    exit 1
fi
