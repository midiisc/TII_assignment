#!/bin/bash

echo "🔍 COMPREHENSIVE LIBRARY PATH VERIFICATION"
echo "=========================================="
echo "Date: $(date)"
echo "System: $(lsb_release -d | cut -f2)"
echo "Architecture: $(uname -m)"
echo ""

# Function to check if file exists and show details
check_file() {
    local file="$1"
    local description="$2"
    
    echo -n "  $description: "
    if [ -f "$file" ]; then
        echo "✅ FOUND"
        echo "    Path: $file"
        echo "    Size: $(stat -c%s "$file") bytes"
        echo "    Modified: $(stat -c%y "$file" | cut -d. -f1)"
        if [[ "$file" == *.so* ]]; then
            echo "    Type: $(file "$file" | cut -d: -f2 | xargs)"
        fi
    else
        echo "❌ NOT FOUND"
    fi
    echo ""
}

# Function to check directory and list contents
check_directory() {
    local dir="$1"
    local description="$2"
    local pattern="$3"
    
    echo -n "  $description: "
    if [ -d "$dir" ]; then
        echo "✅ EXISTS"
        echo "    Path: $dir"
        if [ -n "$pattern" ]; then
            echo "    Contents matching '$pattern':"
            find "$dir" -name "$pattern" 2>/dev/null | head -10 | sed 's/^/      /'
            local count=$(find "$dir" -name "$pattern" 2>/dev/null | wc -l)
            if [ $count -gt 10 ]; then
                echo "      ... and $((count-10)) more files"
            fi
        fi
    else
        echo "❌ NOT FOUND"
    fi
    echo ""
}

# Function to check package installation
check_package() {
    local package="$1"
    echo -n "  Package $package: "
    if dpkg -l | grep -q "^ii.*$package "; then
        local version=$(dpkg -l | grep "^ii.*$package " | awk '{print $3}' | head -1)
        echo "✅ INSTALLED (version: $version)"
    else
        echo "❌ NOT INSTALLED"
    fi
}

echo "📦 PACKAGE VERIFICATION"
echo "======================"
check_package "libopenblas-dev"
check_package "libopenblas0-pthread"
check_package "liblapack-dev"
check_package "liblapacke-dev"
check_package "libtbb-dev"
check_package "libmetis-dev"
check_package "libsuitesparse-dev"
check_package "libeigen3-dev"
echo ""

echo "🔗 OPENBLAS LIBRARY VERIFICATION"
echo "==============================="
check_file "/usr/lib/x86_64-linux-gnu/libopenblas.so" "OpenBLAS shared library"
check_file "/usr/lib/x86_64-linux-gnu/libopenblas.a" "OpenBLAS static library"
check_file "/usr/lib/x86_64-linux-gnu/libopenblas-r0.3.20.so" "OpenBLAS versioned library"
check_file "/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so" "OpenBLAS pthread variant"

echo "📋 CBLAS HEADER VERIFICATION"
echo "==========================="
check_file "/usr/include/x86_64-linux-gnu/cblas.h" "CBLAS header (arch-specific)"
check_file "/usr/include/cblas.h" "CBLAS header (generic)"
check_file "/usr/include/openblas/cblas.h" "CBLAS header (openblas dir)"

# Find all cblas.h files
echo "  All CBLAS headers found:"
find /usr/include -name "cblas.h" 2>/dev/null | sed 's/^/    /'
echo ""

echo "📐 LAPACK/LAPACKE VERIFICATION"
echo "============================="
check_file "/usr/lib/x86_64-linux-gnu/liblapack.so" "LAPACK shared library"
check_file "/usr/lib/x86_64-linux-gnu/liblapacke.so" "LAPACKE shared library"
check_file "/usr/include/lapacke.h" "LAPACKE header (generic)"
check_file "/usr/include/x86_64-linux-gnu/lapacke.h" "LAPACKE header (arch-specific)"

# Find all lapacke.h files
echo "  All LAPACKE headers found:"
find /usr/include -name "lapacke.h" 2>/dev/null | sed 's/^/    /'
echo ""

echo "🧵 TBB VERIFICATION"
echo "=================="
check_file "/usr/lib/x86_64-linux-gnu/libtbb.so" "TBB shared library"
check_file "/usr/include/tbb/tbb.h" "TBB main header"
check_file "/usr/include/oneapi/tbb.h" "TBB OneAPI header"
check_directory "/usr/include/tbb" "TBB headers directory" "*.h"

echo "🕸️ SUITESPARSE VERIFICATION"
echo "==========================="
check_file "/usr/lib/x86_64-linux-gnu/libcholmod.so" "CHOLMOD library"
check_file "/usr/lib/x86_64-linux-gnu/libspqr.so" "SPQR library"
check_file "/usr/lib/x86_64-linux-gnu/libamd.so" "AMD library"
check_file "/usr/lib/x86_64-linux-gnu/libcolamd.so" "COLAMD library"
check_file "/usr/lib/x86_64-linux-gnu/libcamd.so" "CAMD library"
check_file "/usr/lib/x86_64-linux-gnu/libccolamd.so" "CCOLAMD library"
check_file "/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so" "SuiteSparse config library"
check_directory "/usr/include/suitesparse" "SuiteSparse headers" "*.h"

echo "🔢 METIS VERIFICATION"
echo "===================="
check_file "/usr/lib/x86_64-linux-gnu/libmetis.so" "METIS shared library"
check_file "/usr/include/metis.h" "METIS header"

echo "🧮 EIGEN VERIFICATION"
echo "===================="
check_directory "/usr/include/eigen3" "Eigen3 headers" "*.h"
check_file "/usr/include/eigen3/Eigen/Dense" "Eigen Dense module"
check_file "/usr/include/eigen3/Eigen/Sparse" "Eigen Sparse module"

echo "🎯 CUDA VERIFICATION"
echo "==================="
check_file "/usr/local/cuda/lib64/libcudart.so" "CUDA runtime library"
check_file "/usr/local/cuda/lib64/libcublas.so" "cuBLAS library"
check_file "/usr/local/cuda/lib64/libcusolver.so" "cuSOLVER library"
check_file "/usr/local/cuda/lib64/libcusparse.so" "cuSPARSE library"
check_file "/usr/local/cuda/include/cuda.h" "CUDA header"

echo "🐍 PYTHON VERIFICATION"
echo "======================"
check_file "/usr/bin/python3" "Python3 executable"
check_file "/usr/include/python3.10/Python.h" "Python3 headers"
check_file "/usr/lib/x86_64-linux-gnu/libpython3.10.so" "Python3 library"
check_directory "/usr/lib/python3/dist-packages/numpy" "NumPy installation" "*.py"

echo "🔧 CMAKE & BUILD TOOLS"
echo "======================"
check_file "/usr/bin/cmake" "CMake"
check_file "/usr/bin/ninja" "Ninja build system"
check_file "/usr/bin/ccache" "CCache"
check_file "/usr/bin/gcc" "GCC compiler"
check_file "/usr/bin/g++" "G++ compiler"

echo "📊 LIBRARY ALTERNATIVES"
echo "======================"
echo "  BLAS alternatives:"
update-alternatives --list libblas.so.3-x86_64-linux-gnu 2>/dev/null | sed 's/^/    /' || echo "    No alternatives configured"
echo ""
echo "  LAPACK alternatives:"
update-alternatives --list liblapack.so.3-x86_64-linux-gnu 2>/dev/null | sed 's/^/    /' || echo "    No alternatives configured"
echo ""

echo "🔍 PKG-CONFIG VERIFICATION"
echo "=========================="
echo "  Available .pc files for our libraries:"
find /usr/lib/x86_64-linux-gnu/pkgconfig /usr/lib/pkgconfig /usr/local/lib/pkgconfig 2>/dev/null -name "*.pc" | grep -E "(blas|lapack|tbb|eigen|opencv|ceres)" | sed 's/^/    /'
echo ""

echo "💾 LDCONFIG CACHE CHECK"
echo "======================"
echo "  Libraries in ldconfig cache:"
ldconfig -p | grep -E "(blas|lapack|tbb|cholmod|metis)" | sed 's/^/    /'
echo ""

echo "🎉 VERIFICATION COMPLETE"
echo "======================="
echo "Timestamp: $(date)"
echo ""
echo "💡 RECOMMENDED CMAKE PATHS:"
echo "=========================="
echo "Based on the verification above, use these CMake flags:"
echo ""

# Generate recommended flags based on what we found
if [ -f "/usr/lib/x86_64-linux-gnu/libopenblas.so" ]; then
    echo "  # OpenBLAS/LAPACK"
    echo "  -D WITH_LAPACK=ON \\"
    echo "  -D LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \\"
    
    if [ -f "/usr/include/x86_64-linux-gnu/cblas.h" ]; then
        echo "  -D LAPACK_CBLAS_H=/usr/include/x86_64-linux-gnu/cblas.h \\"
    elif [ -f "/usr/include/cblas.h" ]; then
        echo "  -D LAPACK_CBLAS_H=/usr/include/cblas.h \\"
    fi
    
    if [ -f "/usr/include/lapacke.h" ]; then
        echo "  -D LAPACK_LAPACKE_H=/usr/include/lapacke.h \\"
    elif [ -f "/usr/include/x86_64-linux-gnu/lapacke.h" ]; then
        echo "  -D LAPACK_LAPACKE_H=/usr/include/x86_64-linux-gnu/lapacke.h \\"
    fi
fi

if [ -f "/usr/lib/x86_64-linux-gnu/libtbb.so" ]; then
    echo ""
    echo "  # TBB"
    echo "  -D WITH_TBB=ON \\"
    echo "  -D TBB_INCLUDE_DIRS=/usr/include \\"
    echo "  -D TBB_LIBRARIES=/usr/lib/x86_64-linux-gnu \\"
fi

echo ""
echo "📝 Save this output for reference when configuring OpenCV and Ceres!"
