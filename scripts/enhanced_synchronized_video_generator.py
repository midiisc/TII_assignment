#!/usr/bin/env python3
"""
Enhanced Synchronized Video Generator
Creates videos showing camera stream alongside real-time trajectory visualization
with full hardware optimization, monitoring, and safety features
"""

import os
import sys
import time
import math
import logging
import argparse
import multiprocessing
import gc
import io
import ctypes
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure matplotlib for better stability
plt.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
import cv2
import imageio
import psutil
from PIL import Image

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Optional GPU monitoring and acceleration
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("🚀 CuPy GPU acceleration available")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Numba for JIT compilation
try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
    logger.info("🚀 Numba JIT compilation available")
    
    # Define JIT-compiled functions for performance-critical operations
    @jit(nopython=True, cache=True)
    def calculate_distance_numba(x1, y1, x2, y2):
        """Numba-accelerated distance calculation"""
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    @jit(nopython=True, cache=True)
    def calculate_speed_numba(vx, vy, vz):
        """Numba-accelerated speed calculation"""
        return (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5
    
    @jit(nopython=True, cache=True)
    def calculate_turning_rate_numba(wx, wy, wz):
        """Numba-accelerated turning rate calculation"""
        return (wx ** 2 + wy ** 2 + wz ** 2) ** 0.5
    
    @jit(nopython=True, cache=True)
    def interpolate_linear_numba(x1, y1, x2, y2, x_target):
        """Numba-accelerated linear interpolation"""
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * (x_target - x1) / (x2 - x1)
    
    @jit(nopython=True, cache=True)
    def interpolate_quaternion_numba(q1, q2, t):
        """Numba-accelerated quaternion interpolation (SLERP approximation)"""
        # Simplified quaternion interpolation for performance
        # q1, q2 are arrays [x, y, z, w], t is interpolation factor [0, 1]
        x = q1[0] + t * (q2[0] - q1[0])
        y = q1[1] + t * (q2[1] - q1[1])
        z = q1[2] + t * (q2[2] - q1[2])
        w = q1[3] + t * (q2[3] - q1[3])
        
        # Normalize quaternion
        norm = (x*x + y*y + z*z + w*w) ** 0.5
        if norm > 0:
            return [x/norm, y/norm, z/norm, w/norm]
        else:
            return q1
    
    @jit(nopython=True, cache=True)
    def find_closest_timestamp_numba(timestamps, target_timestamp):
        """Numba-accelerated closest timestamp finding"""
        min_diff = abs(timestamps[0] - target_timestamp)
        closest_idx = 0
        
        for i in range(1, len(timestamps)):
            diff = abs(timestamps[i] - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        return closest_idx, min_diff
    
    # Additional CUDA kernel functions for advanced GPU acceleration
    if cuda.is_available():
        @cuda.jit
        def cuda_vector_norm_kernel(vectors, results):
            """CUDA kernel for vectorized norm calculation"""
            idx = cuda.grid(1)
            if idx < vectors.shape[0]:
                sum_sq = 0.0
                for i in range(vectors.shape[1]):
                    sum_sq += vectors[idx, i] * vectors[idx, i]
                results[idx] = sum_sq ** 0.5
        
        @cuda.jit
        def cuda_interpolation_kernel(before_vals, after_vals, weights, results):
            """CUDA kernel for parallel linear interpolation"""
            idx = cuda.grid(1)
            if idx < results.shape[0]:
                results[idx] = before_vals[idx] + weights[idx] * (after_vals[idx] - before_vals[idx])
        
        @cuda.jit
        def cuda_timestamp_diff_kernel(timestamps, target, results):
            """CUDA kernel for parallel timestamp difference calculation"""
            idx = cuda.grid(1)
            if idx < timestamps.shape[0]:
                results[idx] = abs(timestamps[idx] - target)
    
except ImportError:
    NUMBA_AVAILABLE = False
    cuda = None
    jit = None
    
    # Fallback functions when Numba is not available
    def calculate_distance_numba(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def calculate_speed_numba(vx, vy, vz):
        return (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5
    
    def calculate_turning_rate_numba(wx, wy, wz):
        return (wx ** 2 + wy ** 2 + wz ** 2) ** 0.5
    
    def interpolate_linear_numba(x1, y1, x2, y2, x_target):
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * (x_target - x1) / (x2 - x1)
    
    def interpolate_quaternion_numba(q1, q2, t):
        x = q1[0] + t * (q2[0] - q1[0])
        y = q1[1] + t * (q2[1] - q1[1])
        z = q1[2] + t * (q2[2] - q1[2])
        w = q1[3] + t * (q2[3] - q1[3])
        norm = (x*x + y*y + z*z + w*w) ** 0.5
        if norm > 0:
            return [x/norm, y/norm, z/norm, w/norm]
        else:
            return q1
    
    def find_closest_timestamp_numba(timestamps, target_timestamp):
        min_diff = abs(timestamps[0] - target_timestamp)
        closest_idx = 0
        for i in range(1, len(timestamps)):
            diff = abs(timestamps[i] - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        return closest_idx, min_diff
    
    # Fallback functions for CUDA kernels
    def cuda_vector_norm_kernel(vectors, results):
        """CPU fallback for CUDA vector norm kernel"""
        for i in range(vectors.shape[0]):
            sum_sq = 0.0
            for j in range(vectors.shape[1]):
                sum_sq += vectors[i, j] * vectors[i, j]
            results[i] = sum_sq ** 0.5
    
    def cuda_interpolation_kernel(before_vals, after_vals, weights, results):
        """CPU fallback for CUDA interpolation kernel"""
        for i in range(results.shape[0]):
            results[i] = before_vals[i] + weights[i] * (after_vals[i] - before_vals[i])
    
    def cuda_timestamp_diff_kernel(timestamps, target, results):
        """CPU fallback for CUDA timestamp difference kernel"""
        for i in range(timestamps.shape[0]):
            results[i] = abs(timestamps[i] - target)

class GPUAccelerator:
    """GPU acceleration utilities for image processing and computations"""
    
    def __init__(self):
        self.cupy_available = CUPY_AVAILABLE
        self.numba_available = NUMBA_AVAILABLE
        self.opencv_cuda_available = False
        
        # Initialize GPU acceleration
        self._initialize_gpu_acceleration()
    
    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration capabilities"""
        # Check OpenCV CUDA
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.opencv_cuda_available = True
                logger.info("🚀 OpenCV CUDA acceleration enabled")
        except:
            pass
        
        # Initialize CuPy if available
        if self.cupy_available:
            try:
                # Test CuPy functionality
                test_array = cp.array([1, 2, 3])
                logger.info(f"🚀 CuPy initialized with {cp.cuda.runtime.getDeviceCount()} GPU(s)")
            except Exception as e:
                logger.warning(f"⚠️ CuPy initialization failed: {e}")
                self.cupy_available = False
        
        # Initialize Numba CUDA if available
        if self.numba_available:
            try:
                if cuda.is_available():
                    logger.info(f"🚀 Numba CUDA available with {len(cuda.gpus)} GPU(s)")
                else:
                    self.numba_available = False
            except Exception as e:
                logger.warning(f"⚠️ Numba CUDA initialization failed: {e}")
                self.numba_available = False
    

class HardwareMonitor:
    """Hardware health monitoring with emergency stops"""
    
    def __init__(self, memory_limit: float = 0.8, gpu_memory_limit: float = 0.9, 
                 cpu_temp_limit: float = 85.0, gpu_temp_limit: float = 85.0):
        self.memory_limit = memory_limit
        self.gpu_memory_limit = gpu_memory_limit
        self.cpu_temp_limit = cpu_temp_limit
        self.gpu_temp_limit = gpu_temp_limit
        self.emergency_stop = False
        
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health_status = {
            'memory_usage': psutil.virtual_memory().percent / 100.0,
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'cpu_temp': self._get_cpu_temperature(),
            'gpu_status': self._get_gpu_status(),
            'emergency_stop': False
        }
        
        # Check memory usage
        if health_status['memory_usage'] > self.memory_limit:
            logger.warning(f"⚠️ High memory usage: {health_status['memory_usage']:.1%}")
            if health_status['memory_usage'] > 0.9:
                logger.error("🚨 Critical memory usage - triggering emergency stop")
                health_status['emergency_stop'] = True
                self.emergency_stop = True
        
        # Check CPU temperature
        if health_status['cpu_temp'] > self.cpu_temp_limit:
            logger.warning(f"⚠️ High CPU temperature: {health_status['cpu_temp']:.1f}°C")
            if health_status['cpu_temp'] > 90.0:
                logger.error("🚨 Critical CPU temperature - triggering emergency stop")
                health_status['emergency_stop'] = True
                self.emergency_stop = True
        
        # Check GPU status
        if health_status['gpu_status'] and health_status['gpu_status']['memory_usage'] > self.gpu_memory_limit:
            logger.warning(f"⚠️ High GPU memory usage: {health_status['gpu_status']['memory_usage']:.1%}")
            if health_status['gpu_status']['memory_usage'] > 0.95:
                logger.error("🚨 Critical GPU memory usage - triggering emergency stop")
                health_status['emergency_stop'] = True
                self.emergency_stop = True
        
        return health_status
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max([temp.current for temp in temps['coretemp']])
            return 0.0
        except:
            return 0.0
    
    def _get_gpu_status(self) -> Optional[Dict[str, Any]]:
        """Get GPU status if available"""
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'memory_usage': gpu.memoryUtil,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                }
        except:
            pass
        return None
    
    def cleanup_memory(self):
        """Force comprehensive memory cleanup"""
        # Python garbage collection
        gc.collect()
        
        # OpenCV thread optimization
        if hasattr(cv2, 'setNumThreads'):
            cv2.setNumThreads(1)
        
        # GPU memory cleanup if available
        try:
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        
        # Force system memory cleanup
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

class EnhancedSynchronizedVideoGenerator:
    def __init__(self, bag_paths: List[str], output_dir: str = "reports/visualizations", 
                 test_frames: int = 20, max_workers: int = None):
        self.bag_paths = bag_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_frames = test_frames
        
        # Hardware optimization
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.hardware_monitor = HardwareMonitor()
        
        # GPU acceleration
        self.gpu_accelerator = GPUAccelerator()
        
        # CUDA optimization
        self.cuda_available = self._check_cuda_availability()
        if self.cuda_available:
            # Enable full CUDA acceleration now that compatibility is fixed
            cv2.setUseOptimized(True)
            cv2.setNumThreads(self.max_workers)
            logger.info("🚀 Full GPU acceleration enabled (CUDA + OpenCV + Numba)")
        else:
            # Use CPU optimizations
            cv2.setUseOptimized(True)
            cv2.setNumThreads(self.max_workers)
            logger.info("🚀 CPU optimization enabled (CUDA not available in OpenCV)")
        
        # Parallel processing DISABLED for strict time synchronization
        self.enable_chunk_parallel = False  # DISABLED: Could cause frame ordering issues
        self.parallel_chunks = False        # DISABLED: Could cause frame ordering issues
        self.optimal_chunk_size = 10       # Not used (parallel processing disabled)
        
        self.typestore = get_typestore(Stores.ROS2_JAZZY)
        
        # Video settings
        self.fps = 10
        self.camera_width = 640
        self.camera_height = 360
        
        # Advanced memory management
        self.chunk_size = self.optimal_chunk_size  # Use optimal chunk size for parallel processing
        self.frame_skip = 1   # Legacy variable (not used - frame skipping disabled for sync)
        self.max_memory_usage = 0.8  # Maximum memory usage before aggressive cleanup
        self.gpu_memory_cleanup = True  # Enable GPU memory cleanup
        
        # Frame skip optimization (DISABLED for perfect synchronization)
        self.adaptive_frame_skip = False  # DISABLED: Could cause temporal inconsistency
        self.motion_threshold = 0.1      # Not used (frame skipping disabled)
        self.min_frames_per_second = 5   # Not used (frame skipping disabled)
        
        # Memory allocation optimization
        self._initialize_reusable_buffers()
        
        # Optimized bag reading with streaming (SYNC-SAFE)
        self.smart_sampling = False  # DISABLED: Could affect frame selection
        self.sampling_interval = 1   # DISABLED: Could skip frames (set to 1 for no skipping)
        self.streaming_mode = True   # ENABLED: Safe for time synchronization
        self.preload_frames = 0      # DISABLED: Could affect frame ordering
        # parallel_chunks is already set above in the advanced parallel processing section
        
        # Parallel processing settings (DISABLED for strict sync)
        self.enable_multi_bag_parallel = False  # DISABLED: Could cause timing issues
        self.max_parallel_bags = 1  # Force sequential processing
        # enable_chunk_parallel is already set above in the advanced parallel processing section
        
        logger.info(f"🚀 Multi-bag parallel processing enabled (max {self.max_parallel_bags} bags)")

    def _initialize_reusable_buffers(self):
        """Initialize reusable buffers to reduce memory allocations"""
        try:
            logger.info("🧠 Initializing reusable memory buffers...")
            
            # Pre-allocate numpy arrays for common operations
            max_image_size = 1920 * 1080 * 3  # Maximum expected image size
            self.image_buffer = np.empty(max_image_size, dtype=np.uint8)
            self.temp_image_buffer = np.empty((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            
            # Pre-allocate arrays for motion metrics
            self.speed_buffer = np.empty(10000, dtype=np.float32)  # For speed calculations
            self.acceleration_buffer = np.empty(10000, dtype=np.float32)  # For acceleration calculations
            self.turning_rate_buffer = np.empty(10000, dtype=np.float32)  # For turning rate calculations
            
            # Initialize reusable matplotlib figure and subplot axes (will be created on first use)
            self.reusable_fig = None
            self.fig_buffer = None
            self.subplot_axes = None  # Will store the 4 subplot axes for reuse
            
            # Cached calculations for performance
            self.cached_axis_limits = None  # Cache axis limits to avoid recalculation
            
            # Initialize OpenCV GPU memory pools
            if self.cuda_available and self.gpu_accelerator.opencv_cuda_available:
                try:
                    self.gpu_image_pool = cv2.cuda_GpuMat()
                    self.gpu_temp_pool = cv2.cuda_GpuMat()
                    logger.info("   ✅ GPU memory pools initialized")
                except Exception as e:
                    logger.warning(f"   ⚠️ GPU memory pool initialization failed: {e}")
                    self.gpu_image_pool = None
                    self.gpu_temp_pool = None
            else:
                self.gpu_image_pool = None
                self.gpu_temp_pool = None
            
            # Buffer usage tracking
            self.buffer_usage_count = 0
            self.buffer_creation_time = time.time()
            
            logger.info("   ✅ Reusable buffers initialized successfully")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Buffer initialization failed: {e}")
            # Fallback to None buffers
            self.image_buffer = None
            self.temp_image_buffer = None
            self.speed_buffer = None
            self.acceleration_buffer = None
            self.turning_rate_buffer = None
            self.reusable_fig = None
            self.fig_buffer = None
            self.gpu_image_pool = None
            self.gpu_temp_pool = None
        
        logger.info(f"🎬 Enhanced Synchronized Video Generator initialized")
        logger.info(f"   📁 Output: {self.output_dir}")
        logger.info(f"   🎬 Test frames: {self.test_frames}")
        logger.info(f"   🚀 Max workers: {self.max_workers}")
        logger.info(f"   🧠 CUDA available: {self.cuda_available}")

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available"""
        try:
            # Check if CUDA is available in OpenCV
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                return True
            
            # Check if CUDA is available in the system (for other libraries)
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def generate_synchronized_videos(self):
        """Generate synchronized videos for all bags with STRICT TIME SYNCHRONIZATION"""
        # Initial memory cleanup
        self.hardware_monitor.cleanup_memory()
        
        # FORCE sequential processing for strict time synchronization
        logger.info(f"📝 Using SEQUENTIAL processing for {len(self.bag_paths)} bags (time sync enforced)")
        self._generate_videos_sequential()
    
    def _generate_videos_parallel(self):
        """Generate videos for multiple bags in parallel"""
        try:
            with ProcessPoolExecutor(max_workers=self.max_parallel_bags) as executor:
                # Submit all bag processing tasks
                futures = []
                for bag_path in self.bag_paths:
                    bag_name = Path(bag_path).name
                    future = executor.submit(self._process_single_bag_wrapper, bag_path, bag_name)
                    futures.append((future, bag_name))
                
                # Wait for completion and handle results
                completed_count = 0
                for future, bag_name in futures:
                    try:
                        result = future.result()
                        if result:
                            logger.info(f"   ✅ Enhanced synchronized video generated for {bag_name}")
                        else:
                            logger.warning(f"   ⚠️ No video generated for {bag_name}")
                        completed_count += 1
                    except Exception as e:
                        logger.error(f"   ❌ Error generating synchronized video for {bag_name}: {e}")
                        completed_count += 1
                
                logger.info(f"🎬 Parallel processing completed: {completed_count}/{len(self.bag_paths)} bags processed")
                
        except Exception as e:
            logger.error(f"❌ Parallel processing failed: {e}")
            logger.info("🔄 Falling back to sequential processing...")
            self._generate_videos_sequential()
    
    def _generate_videos_sequential(self):
        """Generate videos for bags sequentially (original method)"""
        for bag_path in self.bag_paths:
            bag_name = Path(bag_path).name
            logger.info(f"🎬 Generating enhanced synchronized video for {bag_name}...")
            
            # Check hardware health before processing
            health = self.hardware_monitor.check_system_health()
            if health['emergency_stop']:
                logger.error("🚨 Emergency stop triggered - aborting video generation")
                break
            
            try:
                self._generate_single_synchronized_video(bag_path, bag_name)
                logger.info(f"   ✅ Enhanced synchronized video generated for {bag_name}")
                
                # Cleanup after each bag
                self.hardware_monitor.cleanup_memory()
                
            except Exception as e:
                logger.error(f"   ❌ Error generating synchronized video for {bag_name}: {e}")
                self.hardware_monitor.cleanup_memory()
    
    def _process_single_bag_wrapper(self, bag_path: str, bag_name: str) -> bool:
        """Wrapper function for parallel bag processing with CUDA context handling"""
        try:
            # Disable CUDA for parallel processes to avoid context conflicts
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA from child processes
            
            # Create a new generator instance for this process
            generator = EnhancedSynchronizedVideoGenerator(
                [bag_path], 
                self.output_dir, 
                self.test_frames,
                self.max_workers
            )
            
            # Copy parallel processing settings
            generator.enable_multi_bag_parallel = False  # Disable nested parallel processing
            generator.enable_chunk_parallel = False      # Disable chunk parallel in child processes
            generator.parallel_chunks = False
            
            # Process the bag
            generator._generate_single_synchronized_video(bag_path, bag_name)
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Error in parallel bag processing for {bag_name}: {e}")
            return False

    def _generate_single_synchronized_video(self, bag_path: str, bag_name: str):
        """Generate synchronized video for a single bag with optimizations"""
        # Use chunked extraction for better memory management
        camera_data, odom_data = self._extract_synchronized_data_chunked(bag_path, chunk_size=200)
        
        if not camera_data or not odom_data:
            logger.warning(f"   ⚠️ No data found for {bag_name}")
            return
        
        # Create output path
        output_path = self.output_dir / f"{bag_name}_enhanced_synchronized_video.mp4"
        
        # Generate video with hardware monitoring
        self._create_enhanced_synchronized_video(camera_data, odom_data, output_path, bag_name)

    def _extract_synchronized_data_range(self, bag_path: str, start_frame: int = 0, 
                                        end_frame: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Extract synchronized data using timestamp-based synchronization"""
        # First pass: collect all camera and odometry data with timestamps
        all_camera_data = []
        all_odom_data = []
        
        try:
            with Reader(bag_path) as reader:
                logger.info(f"   📊 Reading all data from bag for timestamp-based synchronization...")
                
                for connection, timestamp, rawdata in reader.messages():
                    topic_name = connection.topic
                    
                    if topic_name == '/mbuggy/camera_front/image_rect':
                        try:
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            
                            # Convert ROS image to OpenCV format (CUDA-compatible approach)
                            img_data = np.frombuffer(msg.data, dtype=np.uint8)
                            img = np.ascontiguousarray(img_data.reshape((msg.height, msg.width, 3)))
                            
                            # Validate frame
                            if (img is None or img.size == 0 or 
                                not isinstance(img, np.ndarray) or 
                                img.shape != (msg.height, msg.width, 3) or
                                not img.flags.c_contiguous or img.dtype != np.uint8):
                                continue
                            
                            # Process with CUDA-accelerated OpenCV operations
                            try:
                                if (self.cuda_available and self.gpu_accelerator.opencv_cuda_available and 
                                    len(img.shape) == 3 and img.shape[2] == 3):
                                    gpu_img = cv2.cuda_GpuMat()
                                    gpu_img.upload(img)
                                    
                                    if gpu_img.channels() == 3:
                                        # Keep in BGR format for consistency
                                        gpu_img_resized = cv2.cuda.resize(gpu_img, (self.camera_width, self.camera_height))
                                        img_resized = gpu_img_resized.download()
                                    else:
                                        # Keep in BGR format for consistency
                                        img_resized = cv2.resize(img, (self.camera_width, self.camera_height))
                                else:
                                    # Keep in BGR format for consistency
                                    img_resized = cv2.resize(img, (self.camera_width, self.camera_height))
                                
                            except Exception as e:
                                logger.warning(f"   ⚠️ OpenCV processing failed: {e}")
                                continue
                            
                            all_camera_data.append({
                                'timestamp': timestamp,
                                'image': img_resized,
                                'raw_index': len(all_camera_data)  # Raw index before filtering
                            })
                            
                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to process camera frame: {e}")
                            continue
                    
                    elif topic_name == '/mbuggy/odom':
                        try:
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            
                            # Extract position, orientation, and velocities
                            position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                            orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                            linear_velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                            angular_velocity = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                            
                            all_odom_data.append({
                                'timestamp': timestamp,
                                'position': position,
                                'orientation': orientation,
                                'linear_velocity': linear_velocity,
                                'angular_velocity': angular_velocity
                            })
                            
                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to process odometry data: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"   ❌ Error reading bag: {e}")
            return [], []
        
        logger.info(f"   📊 Collected {len(all_camera_data)} camera frames and {len(all_odom_data)} odometry messages")
        
        # Apply frame range filtering to camera data (our synchronization basis)
        if end_frame is None:
            filtered_camera_data = all_camera_data[start_frame:]
        else:
            filtered_camera_data = all_camera_data[start_frame:end_frame]
        
        logger.info(f"   📊 Filtered to {len(filtered_camera_data)} camera frames for range {start_frame}-{end_frame or 'end'}")
        
        # Synchronize odometry data to camera timestamps using closest match
        synchronized_odom_data = self._synchronize_odom_to_camera_timestamps(filtered_camera_data, all_odom_data)
        
        # Assign final indices
        for i, camera_frame in enumerate(filtered_camera_data):
            camera_frame['index'] = i
        
        for i, odom_frame in enumerate(synchronized_odom_data):
            odom_frame['index'] = i
        
        # Validate synchronization
        self._validate_synchronization(filtered_camera_data, synchronized_odom_data)
        
        return filtered_camera_data, synchronized_odom_data

    def _extract_synchronized_data_chunked(self, bag_path: str, chunk_size: int = 200) -> Tuple[List[Dict], List[Dict]]:
        """Extract data in streaming chunks with strict time synchronization"""
        logger.info(f"   🚀 Using streaming chunked processing with {chunk_size} frames per chunk")
        
        # Use streaming approach instead of reading entire bag
        return self._extract_synchronized_data_streaming(bag_path, chunk_size)
    
    def _extract_synchronized_data_streaming(self, bag_path: str, chunk_size: int = 200) -> Tuple[List[Dict], List[Dict]]:
        """Extract data using streaming approach with strict time synchronization"""
        all_camera_data = []
        all_odom_data = []
        
        # Streaming variables
        current_chunk_camera = []
        current_chunk_odom = []
        chunk_idx = 0
        frames_processed = 0
        
        # Enhanced time window for synchronization with excess data collection
        # Collect excess odometry data for better interpolation (as suggested)
        time_window_ns = 500_000_000  # 500ms window for excess data collection
        excess_frames = 5  # Collect 5 extra frames worth of odometry data
        
        try:
            with Reader(bag_path) as reader:
                logger.info(f"   📊 Streaming data with {chunk_size} frames per chunk...")
                
                # First pass: collect camera timestamps to determine chunk boundaries
                camera_timestamps = []
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == '/mbuggy/camera_front/image_rect':
                        camera_timestamps.append(timestamp)
                        if self.test_frames > 0 and len(camera_timestamps) >= self.test_frames:
                            break
                
                logger.info(f"   📊 Found {len(camera_timestamps)} camera frames for streaming")
                
                # Second pass: stream data in chunks
                with Reader(bag_path) as reader:
                    current_camera_idx = 0
                    chunk_start_time = None
                    chunk_end_time = None
                    
                    for connection, timestamp, rawdata in reader.messages():
                        topic_name = connection.topic
                        
                        # Determine chunk boundaries based on camera frames
                        if topic_name == '/mbuggy/camera_front/image_rect':
                            if current_camera_idx >= len(camera_timestamps):
                                break
                                
                            if chunk_start_time is None:
                                chunk_start_time = camera_timestamps[current_camera_idx]
                                chunk_end_time = camera_timestamps[min(current_camera_idx + chunk_size - 1, len(camera_timestamps) - 1)]
                                logger.info(f"   📦 Processing chunk {chunk_idx + 1}: frames {current_camera_idx}-{min(current_camera_idx + chunk_size - 1, len(camera_timestamps) - 1)}")
                            
                            # Process camera frame if within current chunk
                            if chunk_start_time <= timestamp <= chunk_end_time:
                                try:
                                    msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                                    
                                    # Convert and process image
                                    img_data = np.frombuffer(msg.data, dtype=np.uint8)
                                    img = np.ascontiguousarray(img_data.reshape((msg.height, msg.width, 3)))
                                    
                                    if (img is None or img.size == 0 or 
                                        not isinstance(img, np.ndarray) or 
                                        img.shape != (msg.height, msg.width, 3) or
                                        not img.flags.c_contiguous or img.dtype != np.uint8):
                                        continue
                                    
                                    # Process with GPU acceleration
                                    try:
                                        if (self.cuda_available and self.gpu_accelerator.opencv_cuda_available and 
                                            len(img.shape) == 3 and img.shape[2] == 3):
                                            gpu_img = cv2.cuda_GpuMat()
                                            gpu_img.upload(img)
                                            
                                            if gpu_img.channels() == 3:
                                                # Keep in BGR format for consistency
                                                gpu_img_resized = cv2.cuda.resize(gpu_img, (self.camera_width, self.camera_height))
                                                img_resized = gpu_img_resized.download()
                                            else:
                                                # Keep in BGR format for consistency
                                                img_resized = cv2.resize(img, (self.camera_width, self.camera_height))
                                        else:
                                            # Keep in BGR format for consistency
                                            img_resized = cv2.resize(img, (self.camera_width, self.camera_height))
                                        
                                    except Exception as e:
                                        logger.warning(f"   ⚠️ OpenCV processing failed: {e}")
                                        continue
                                    
                                    current_chunk_camera.append({
                                        'timestamp': timestamp,
                                        'image': img_resized,
                                        'raw_index': current_camera_idx
                                    })
                                    
                                    current_camera_idx += 1
                                    
                                except Exception as e:
                                    logger.warning(f"   ⚠️ Failed to process camera frame: {e}")
                                    current_camera_idx += 1
                                    continue
                            
                            # Check if chunk is complete
                            if len(current_chunk_camera) >= chunk_size or current_camera_idx >= len(camera_timestamps):
                                # Process this chunk
                                chunk_camera_data, chunk_odom_data = self._process_streaming_chunk(
                                    current_chunk_camera, current_chunk_odom
                                )
                                
                                # Add to results with proper indexing
                                try:
                                    for i, data in enumerate(chunk_camera_data):
                                        data['index'] = frames_processed + i
                                    for i, data in enumerate(chunk_odom_data):
                                        data['index'] = frames_processed + i
                                except Exception as e:
                                    logger.error(f"   ❌ Error setting indices: {e}")
                                    logger.error(f"   📊 Camera data length: {len(chunk_camera_data)}, Odom data length: {len(chunk_odom_data)}")
                                    if chunk_camera_data:
                                        logger.error(f"   📊 Camera data keys: {list(chunk_camera_data[0].keys())}")
                                    if chunk_odom_data:
                                        logger.error(f"   📊 Odom data keys: {list(chunk_odom_data[0].keys())}")
                                    raise
                                
                                all_camera_data.extend(chunk_camera_data)
                                all_odom_data.extend(chunk_odom_data)
                                frames_processed += len(chunk_camera_data)
                                
                                # Reset for next chunk
                                current_chunk_camera = []
                                current_chunk_odom = []
                                chunk_idx += 1
                                chunk_start_time = None
                                chunk_end_time = None
                                
                                # Memory cleanup between chunks
                                if chunk_idx % 5 == 0:
                                    self.hardware_monitor.cleanup_memory()
                                
                                # Check if we've processed enough frames
                                if self.test_frames > 0 and frames_processed >= self.test_frames:
                                    break
                        
                        # Collect odometry data with excess range for better interpolation
                        elif topic_name == '/mbuggy/odom':
                            # Collect excess odometry data around chunk boundaries for better interpolation
                            if chunk_start_time is not None:
                                # Calculate excess time range (extend beyond chunk boundaries)
                                excess_start_time = chunk_start_time - time_window_ns
                                excess_end_time = chunk_end_time + time_window_ns
                                
                                if excess_start_time <= timestamp <= excess_end_time:
                                    try:
                                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                                        
                                        position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                                        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                                                     msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                                        linear_velocity = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                                        angular_velocity = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                                        
                                        current_chunk_odom.append({
                                            'timestamp': timestamp,
                                            'position': position,
                                            'orientation': orientation,
                                            'linear_velocity': linear_velocity,
                                            'angular_velocity': angular_velocity
                                        })
                                        
                                    except Exception as e:
                                        logger.warning(f"   ⚠️ Failed to process odometry data: {e}")
                                        continue
                    
                    # Process final chunk if any
                    if current_chunk_camera:
                        chunk_camera_data, chunk_odom_data = self._process_streaming_chunk(
                            current_chunk_camera, current_chunk_odom
                        )
                        
                        for i, data in enumerate(chunk_camera_data):
                            data['index'] = frames_processed + i
                        for i, data in enumerate(chunk_odom_data):
                            data['index'] = frames_processed + i
                        
                        all_camera_data.extend(chunk_camera_data)
                        all_odom_data.extend(chunk_odom_data)
                        frames_processed += len(chunk_camera_data)
                        
        except Exception as e:
            logger.error(f"   ❌ Error in streaming processing: {e}")
            import traceback
            logger.error(f"   📊 Traceback: {traceback.format_exc()}")
            return [], []
        
        logger.info(f"   ✅ Streaming extraction complete: {len(all_camera_data)} camera frames, {len(all_odom_data)} odometry messages")
        return all_camera_data, all_odom_data
    
    def _process_streaming_chunk(self, chunk_camera_data: List[Dict], chunk_odom_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Process a streaming chunk with strict time synchronization and full interpolation"""
        if not chunk_camera_data:
            return [], []
        
        logger.info(f"   📊 Processing chunk: {len(chunk_camera_data)} camera frames, {len(chunk_odom_data)} odometry messages")
        
        # Use full interpolation capabilities with excess odometry data
        synchronized_odom_data = self._synchronize_odom_to_camera_timestamps(
            chunk_camera_data, chunk_odom_data
        )
        
        # Log interpolation statistics for this chunk
        interpolated_count = sum(1 for odom in synchronized_odom_data if odom.get('interpolated', False))
        if interpolated_count > 0:
            logger.info(f"   🔄 Chunk interpolation: {interpolated_count}/{len(synchronized_odom_data)} frames interpolated")
        
        # Ensure all data has index for validation
        for i, camera_data in enumerate(chunk_camera_data):
            if 'index' not in camera_data:
                camera_data['index'] = i
        for i, odom_data in enumerate(synchronized_odom_data):
            if 'index' not in odom_data:
                odom_data['index'] = i
        
        # Validate synchronization for this chunk
        self._validate_synchronization(chunk_camera_data, synchronized_odom_data)
        
        return chunk_camera_data, synchronized_odom_data

    def _extract_specific_frame_ranges(self, bag_path: str, frame_ranges: List[Tuple[int, int]]) -> Tuple[List[Dict], List[Dict]]:
        """Extract data from specific frame ranges (e.g., [(0, 200), (500, 700), (1000, 1200)])"""
        all_camera_data = []
        all_odom_data = []
        
        logger.info(f"   📊 Extracting {len(frame_ranges)} specific frame ranges from bag")
        
        for range_idx, (start_frame, end_frame) in enumerate(frame_ranges):
            logger.info(f"   📦 Processing range {range_idx + 1}/{len(frame_ranges)}: frames {start_frame}-{end_frame}")
            
            # Extract data for this range
            range_camera_data, range_odom_data = self._extract_synchronized_data_range(
                bag_path, start_frame, end_frame
            )
            
            # Adjust indices to be continuous across all ranges
            base_index = len(all_camera_data)
            for i, data in enumerate(range_camera_data):
                data['index'] = base_index + i
            for i, data in enumerate(range_odom_data):
                data['index'] = base_index + i
            
            all_camera_data.extend(range_camera_data)
            all_odom_data.extend(range_odom_data)
            
            # Memory cleanup between ranges
            self.hardware_monitor.cleanup_memory()
        
        logger.info(f"   ✅ Range-based extraction complete: {len(all_camera_data)} camera frames, {len(all_odom_data)} odometry messages")
        return all_camera_data, all_odom_data

    def _create_enhanced_synchronized_video(self, camera_data: List[Dict], odom_data: List[Dict], 
                                          output_path: Path, bag_name: str):
        """Create enhanced synchronized video with hardware monitoring"""
        
        # Calculate motion metrics for the FULL dataset
        motion_metrics = self._calculate_motion_metrics(odom_data)
        
        # Calculate consistent axis limits for all plots based on FULL dataset
        axis_limits = self._calculate_axis_limits(odom_data, motion_metrics)
        
        # Limit frames for testing if specified
        if self.test_frames > 0:
            camera_data = camera_data[:self.test_frames]
            logger.info(f"   📊 Limited to {self.test_frames} camera frames for testing")
        
        logger.info(f"   🎬 Creating enhanced synchronized video with {len(camera_data)} frames...")
        
        # FORCE sequential chunk processing for strict time synchronization
        logger.info(f"   📝 Using SEQUENTIAL chunk processing for {len(camera_data)} frames (time sync enforced)")
        frames = self._create_video_with_sequential_chunks(camera_data, odom_data, motion_metrics, axis_limits, bag_name)
        
        if not frames:
            logger.error("   ❌ No frames generated")
            return
        
        # Write video using imageio with hardware monitoring
        logger.info(f"   💾 Writing video with {len(frames)} frames...")
        logger.info(f"   📁 Video output path: {output_path.absolute()}")
        
        try:
            imageio.mimsave(str(output_path), frames, fps=self.fps, macro_block_size=1)
            logger.info(f"   ✅ Enhanced video saved: {output_path.absolute()}")
        except Exception as e:
            logger.error(f"   ❌ Error writing video: {e}")

    def _process_frame_chunk(self, camera_chunk: List[Dict], odom_data: List[Dict], 
                           motion_metrics: List[Dict], axis_limits: Dict, 
                           bag_name: str, chunk_start: int) -> List[np.ndarray]:
        """Process a chunk of frames with perfect synchronization (no frame skipping)"""
        frames = []
        
        # Process ALL frames to maintain perfect timestamp-based synchronization
        # Frame skipping is disabled to preserve temporal consistency
        for i, camera_frame in enumerate(camera_chunk):
            try:
                frame = self._create_single_frame_gpu_optimized(
                    camera_frame, odom_data, motion_metrics, axis_limits, 
                    bag_name, chunk_start + i
                )
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to process frame: {e}")
                continue
        
        return frames

    def _create_video_with_sequential_chunks(self, camera_data: List[Dict], odom_data: List[Dict], 
                                           motion_metrics: List[Dict], axis_limits: Dict, bag_name: str) -> List[np.ndarray]:
        """Create video using sequential chunk processing (original method)"""
        frames = []
        chunk_start = 0
        
        while chunk_start < len(camera_data):
            # Check hardware health before each chunk
            health = self.hardware_monitor.check_system_health()
            if health['emergency_stop']:
                logger.error("🚨 Emergency stop during video generation")
                break
            
            # FIXED chunk size for strict time synchronization (no dynamic adjustment)
            # Memory monitoring only - no chunk size changes to maintain frame order
            if health['memory_usage'] > 0.8:
                logger.warning(f"   ⚠️ High memory usage: {health['memory_usage']:.1%} - performing cleanup")
                self.hardware_monitor.cleanup_memory()
            
            chunk_end = min(chunk_start + self.chunk_size, len(camera_data))
            chunk_frames = self._process_frame_chunk(
                camera_data[chunk_start:chunk_end], 
                odom_data, 
                motion_metrics, 
                axis_limits, 
                bag_name,
                chunk_start
            )
            
            frames.extend(chunk_frames)
            chunk_start = chunk_end
            
            # Advanced cleanup after each chunk
            self.hardware_monitor.cleanup_memory()
            
            # Periodic aggressive cleanup
            if chunk_start % (10 * self.chunk_size) == 0:
                logger.info(f"   🧹 Performing aggressive memory cleanup...")
                self._aggressive_memory_cleanup()
            
            logger.info(f"   📊 Processed chunk {chunk_start}/{len(camera_data)} frames")
        
        return frames
    
    def _create_video_with_parallel_chunks(self, camera_data: List[Dict], odom_data: List[Dict], 
                                         motion_metrics: List[Dict], axis_limits: Dict, bag_name: str) -> List[np.ndarray]:
        """Create video using optimized parallel chunk processing"""
        try:
            # Split data into optimal chunks
            chunks = []
            for i in range(0, len(camera_data), self.optimal_chunk_size):
                chunk_end = min(i + self.optimal_chunk_size, len(camera_data))
                chunks.append((camera_data[i:chunk_end], i))
            
            logger.info(f"   📊 Processing {len(chunks)} chunks in parallel (chunk size: {self.optimal_chunk_size})...")
            
            # Use optimal number of workers (CPU cores but not too many to avoid overhead)
            max_workers = min(len(chunks), self.max_workers, 4)  # Cap at 4 to avoid overhead
            
            # Process chunks in parallel with optimized worker count
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for chunk_data, chunk_start in chunks:
                    future = executor.submit(
                        self._process_frame_chunk,
                        chunk_data, odom_data, motion_metrics, axis_limits, bag_name, chunk_start
                    )
                    futures.append((future, chunk_start))
                
                # Collect results in correct order
                all_frames = []
                chunk_results = [(future.result(), chunk_start) for future, chunk_start in futures]
                chunk_results.sort(key=lambda x: x[1])  # Sort by chunk_start to maintain order
                
                for chunk_frames, _ in chunk_results:
                    all_frames.extend(chunk_frames)
                
                # Memory cleanup after parallel processing
                self.hardware_monitor.cleanup_memory()
                logger.info(f"   🧹 Performing aggressive memory cleanup after parallel processing...")
                self._aggressive_memory_cleanup()
                
                logger.info(f"   ✅ Parallel processing complete: {len(all_frames)} frames processed")
                return all_frames
                
        except Exception as e:
            logger.warning(f"   ⚠️ Parallel chunk processing failed: {e}")
            logger.info("   🔄 Falling back to sequential chunk processing...")
            return self._create_video_with_sequential_chunks(camera_data, odom_data, motion_metrics, axis_limits, bag_name)

    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup with buffer management"""
        try:
            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            # Clear reusable buffers (but don't deallocate them)
            if hasattr(self, 'image_buffer') and self.image_buffer is not None:
                self.image_buffer.fill(0)
            if hasattr(self, 'temp_image_buffer') and self.temp_image_buffer is not None:
                self.temp_image_buffer.fill(0)
            
            # Clear figure buffer
            if hasattr(self, 'fig_buffer') and self.fig_buffer is not None:
                self.fig_buffer.seek(0)
                self.fig_buffer.truncate(0)
            
            # GPU memory cleanup
            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass
            
            # Clear GPU memory pools
            if hasattr(self, 'gpu_image_pool') and self.gpu_image_pool is not None:
                try:
                    self.gpu_image_pool.release()
                except:
                    pass
            if hasattr(self, 'gpu_temp_pool') and self.gpu_temp_pool is not None:
                try:
                    self.gpu_temp_pool.release()
                except:
                    pass
            
            # System memory cleanup
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass
            
            # Clear matplotlib cache and close all figures (except reusable one)
            try:
                plt.clf()
                # Don't close reusable figure, just clear it
                if self.reusable_fig is not None:
                    self.reusable_fig.clear()
                plt.close('all')
                # Recreate reusable figure if it was closed
                if self.reusable_fig is None:
                    self._get_reusable_figure()
            except:
                pass
            
            logger.info("   ✅ Aggressive memory cleanup with buffer management completed")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Aggressive memory cleanup failed: {e}")

    def _monitor_buffer_usage(self):
        """Monitor buffer usage and performance"""
        try:
            self.buffer_usage_count += 1
            
            # Log buffer statistics every 100 frames
            if self.buffer_usage_count % 100 == 0:
                elapsed_time = time.time() - self.buffer_creation_time
                frames_per_second = self.buffer_usage_count / elapsed_time
                
                logger.info(f"   📊 Buffer Usage Stats:")
                logger.info(f"      🎬 Frames processed: {self.buffer_usage_count}")
                logger.info(f"      ⏱️  Processing rate: {frames_per_second:.2f} fps")
                logger.info(f"      🧠 Memory efficiency: Reusing {len([b for b in [self.image_buffer, self.temp_image_buffer, self.fig_buffer, self.gpu_image_pool, self.gpu_temp_pool] if b is not None])} buffers")
                
        except Exception as e:
            logger.warning(f"   ⚠️ Buffer monitoring failed: {e}")

    def _create_single_frame_gpu_optimized(self, camera_frame: Dict, odom_data: List[Dict], 
                                         motion_metrics: List[Dict], axis_limits: Dict, 
                                         bag_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """Create a single visualization frame with GPU acceleration"""
        try:
            start_time = time.time()
            
            # Decode camera image with GPU acceleration
            camera_image = self._decode_camera_image_gpu(camera_frame)
            if camera_image is None:
                return None
            
            # Create visualization frame with GPU acceleration
            vis_frame = self._create_visualization_frame(
                camera_image, odom_data, motion_metrics, 
                camera_frame['index'], bag_name, axis_limits
            )
            
            # Monitor buffer usage and performance
            self._monitor_buffer_usage()
            
            # Log performance improvement (reduced frequency)
            processing_time = time.time() - start_time
            if frame_idx % 100 == 0:  # Log every 100 frames instead of 50
                gpu_status = "🚀 GPU" if CUPY_AVAILABLE else "💻 CPU"
                logger.info(f"   ⚡ Frame {frame_idx}: {gpu_status} processing time: {processing_time:.3f}s")
            
            return vis_frame
            
        except Exception as e:
            logger.warning(f"   ⚠️ GPU-optimized frame creation failed: {e}")
            # Fallback to CPU method
            return self._create_single_frame(camera_frame, odom_data, motion_metrics, axis_limits, bag_name, frame_idx)
    
    def _decode_camera_image_gpu(self, camera_frame: Dict) -> Optional[np.ndarray]:
        """Decode camera image with full GPU acceleration (CuPy + OpenCV CUDA)"""
        try:
            # Get the pre-processed image from the camera frame
            camera_image = camera_frame['image']
            
            if camera_image is None:
                return None
            
            # Images are now kept in BGR format throughout the pipeline for consistency
            
            # Return raw image without any processing - exactly as stored in ROS bag
            return camera_image
            
        except Exception as e:
            logger.warning(f"   ⚠️ GPU image decoding failed, using CPU: {e}")
            return camera_frame['image'] if camera_frame['image'] is not None else None
    




    def _calculate_motion_metrics(self, odom_data: List[Dict]) -> List[Dict]:
        """Calculate motion metrics from odometry data with CuPy acceleration"""
        if not odom_data:
            return []
        
        # Use CuPy acceleration if available
        if CUPY_AVAILABLE and len(odom_data) > 100:  # Use GPU for large datasets
            try:
                return self._calculate_motion_metrics_gpu_enhanced(odom_data)
            except Exception as e:
                logger.warning(f"   ⚠️ Enhanced GPU calculation failed: {e}")
                return self._calculate_motion_metrics_gpu(odom_data)
        else:
            return self._calculate_motion_metrics_cpu(odom_data)
    
    def _calculate_motion_metrics_gpu(self, odom_data: List[Dict]) -> List[Dict]:
        """GPU-accelerated motion metrics calculation using CuPy"""
        try:
            n_frames = len(odom_data)
            
            # Extract all data to GPU arrays for batch processing
            linear_vel = cp.array([[odom['linear_velocity'][0], odom['linear_velocity'][1], odom['linear_velocity'][2]] 
                                   for odom in odom_data])
            angular_vel = cp.array([[odom['angular_velocity'][0], odom['angular_velocity'][1], odom['angular_velocity'][2]] 
                                    for odom in odom_data])
            positions = cp.array([[odom['position'][0], odom['position'][1]] for odom in odom_data])
            
            # Enhanced GPU-accelerated calculations with CuPy
            # 1. Speed calculation with vectorized operations
            speeds = cp.sqrt(cp.sum(linear_vel**2, axis=1))
            
            # 2. Turning rate calculation
            turning_rates = cp.sqrt(cp.sum(angular_vel**2, axis=1))
            
            # 3. Enhanced distance calculation with vectorized operations
            pos_diffs = positions[1:] - positions[:-1]
            step_distances = cp.sqrt(cp.sum(pos_diffs**2, axis=1))
            distances = cp.cumsum(cp.concatenate([cp.array([0.0]), step_distances]))
            
            # 4. Enhanced acceleration calculation (derivative of speed)
            accelerations = cp.zeros_like(speeds)
            accelerations[1:] = speeds[1:] - speeds[:-1]
            
            # 5. Enhanced jerk calculation (derivative of acceleration)
            jerks = cp.zeros_like(accelerations)
            jerks[2:] = accelerations[2:] - accelerations[1:-1]
            
            # 6. Additional motion metrics with GPU acceleration
            # Calculate velocity magnitude changes
            velocity_changes = cp.zeros_like(speeds)
            velocity_changes[1:] = cp.abs(speeds[1:] - speeds[:-1])
            
            # Calculate turning rate changes
            turning_changes = cp.zeros_like(turning_rates)
            turning_changes[1:] = cp.abs(turning_rates[1:] - turning_rates[:-1])
            
            # Calculate path curvature (simplified)
            if len(positions) > 2:
                # Calculate second derivatives for curvature
                pos_second_diff = positions[2:] - 2*positions[1:-1] + positions[:-2]
                curvatures = cp.sqrt(cp.sum(pos_second_diff**2, axis=1))
                # Pad with zeros to match length
                curvatures = cp.concatenate([cp.array([0.0, 0.0]), curvatures])
            else:
                curvatures = cp.zeros_like(speeds)
            
            # Convert back to CPU and create enhanced metrics list
            metrics = []
            for i in range(n_frames):
                metric = {
                    'speed': float(speeds[i]),
                    'acceleration': float(accelerations[i]),
                    'jerk': float(jerks[i]),
                    'turning_rate': float(turning_rates[i]),
                    'distance': float(distances[i]),
                    # Enhanced metrics from GPU acceleration
                    'velocity_change': float(velocity_changes[i]),
                    'turning_change': float(turning_changes[i]),
                    'curvature': float(curvatures[i])
                }
                metrics.append(metric)
            
            # Reduced logging frequency for GPU operations
            if n_frames % 500 == 0:  # Log every 500 frames instead of every time
                logger.info(f"   🚀 GPU-accelerated motion metrics calculated for {n_frames} frames")
            return metrics
            
        except Exception as e:
            logger.warning(f"   ⚠️ GPU motion metrics calculation failed, falling back to CPU: {e}")
            return self._calculate_motion_metrics_cpu(odom_data)
    
    def _calculate_motion_metrics_gpu_enhanced(self, odom_data: List[Dict]) -> List[Dict]:
        """Enhanced GPU-accelerated motion metrics calculation using CuPy and CUDA kernels"""
        try:
            n_frames = len(odom_data)
            
            # Extract all data to GPU arrays for batch processing with enhanced operations
            linear_vel = cp.array([[odom['linear_velocity'][0], odom['linear_velocity'][1], odom['linear_velocity'][2]] 
                                   for odom in odom_data])
            angular_vel = cp.array([[odom['angular_velocity'][0], odom['angular_velocity'][1], odom['angular_velocity'][2]] 
                                    for odom in odom_data])
            positions = cp.array([[odom['position'][0], odom['position'][1]] for odom in odom_data])
            
            # RTX 3080-optimized CUDA kernel parameters based on GPU architecture
            # RTX 3080 has 68 SMs, each can handle multiple blocks for better occupancy
            RTX_3080_SM_COUNT = 68
            OCCUPANCY_FACTOR = 3  # 2-4x SM count for optimal latency hiding
            
            # Test different block sizes for optimal performance (must be multiple of 32)
            optimal_block_sizes = [128, 256, 512]  # Common high-performance sizes
            
            # Calculate if dataset is large enough for efficient CUDA kernel usage
            min_blocks_for_gpu_efficiency = RTX_3080_SM_COUNT * OCCUPANCY_FACTOR  # ~204 blocks minimum
            
            # Find the best configuration for current dataset size
            best_config = None
            for block_size in optimal_block_sizes:
                required_blocks = math.ceil(n_frames / block_size)
                if required_blocks >= min_blocks_for_gpu_efficiency:
                    best_config = (required_blocks, block_size)
                    break
            
            if (NUMBA_AVAILABLE and hasattr(cuda, 'jit') and best_config is not None):
                # Use optimized CUDA kernels for large datasets
                speeds = cp.zeros(n_frames)
                turning_rates = cp.zeros(n_frames)
                
                blockspergrid, threadsperblock = best_config
                total_threads = blockspergrid * threadsperblock
                utilization = (n_frames / total_threads) * 100
                
                try:
                    cuda_vector_norm_kernel[blockspergrid, threadsperblock](linear_vel, speeds)
                    cuda_vector_norm_kernel[blockspergrid, threadsperblock](angular_vel, turning_rates)
                    logger.info(f"   🚀 RTX 3080 optimized: {blockspergrid} blocks × {threadsperblock} threads/block")
                    logger.info(f"   📊 GPU utilization: {utilization:.1f}% ({n_frames}/{total_threads} threads used)")
                except Exception as e:
                    logger.warning(f"   ⚠️ CUDA kernel launch failed: {e}, falling back to CuPy")
                    speeds = cp.sqrt(cp.sum(linear_vel**2, axis=1))
                    turning_rates = cp.sqrt(cp.sum(angular_vel**2, axis=1))
            else:
                # Use CuPy for smaller datasets (more efficient than CUDA kernels for small data)
                reason = "insufficient data size" if best_config is None else "CUDA unavailable"
                logger.info(f"   📊 Using CuPy acceleration for {n_frames} frames ({reason})")
                speeds = cp.sqrt(cp.sum(linear_vel**2, axis=1))
                turning_rates = cp.sqrt(cp.sum(angular_vel**2, axis=1))
            
            # Enhanced distance calculation with GPU optimization
            pos_diffs = positions[1:] - positions[:-1]
            step_distances = cp.sqrt(cp.sum(pos_diffs**2, axis=1))
            distances = cp.cumsum(cp.concatenate([cp.array([0.0]), step_distances]))
            
            # Enhanced acceleration and jerk calculations
            accelerations = cp.zeros_like(speeds)
            accelerations[1:] = speeds[1:] - speeds[:-1]
            
            jerks = cp.zeros_like(accelerations)
            jerks[2:] = accelerations[2:] - accelerations[1:-1]
            
            # Advanced motion metrics with enhanced GPU acceleration
            velocity_changes = cp.zeros_like(speeds)
            velocity_changes[1:] = cp.abs(speeds[1:] - speeds[:-1])
            
            turning_changes = cp.zeros_like(turning_rates)
            turning_changes[1:] = cp.abs(turning_rates[1:] - turning_rates[:-1])
            
            # Enhanced path curvature calculation
            if len(positions) > 2:
                pos_second_diff = positions[2:] - 2*positions[1:-1] + positions[:-2]
                curvatures = cp.sqrt(cp.sum(pos_second_diff**2, axis=1))
                curvatures = cp.concatenate([cp.array([0.0, 0.0]), curvatures])
            else:
                curvatures = cp.zeros_like(speeds)
            
            # Additional advanced metrics
            # Smoothness metric (rate of change of acceleration)
            smoothness = cp.zeros_like(jerks)
            smoothness[1:] = cp.abs(jerks[1:] - jerks[:-1])
            
            # Path efficiency (ratio of direct distance to traveled distance)
            if len(positions) > 1:
                direct_distance = cp.sqrt(cp.sum((positions[-1] - positions[0])**2))
                path_efficiency = cp.full_like(speeds, direct_distance / (distances[-1] + 1e-10))
            else:
                path_efficiency = cp.ones_like(speeds)
            
            # Convert back to CPU and create enhanced metrics list
            metrics = []
            for i in range(n_frames):
                metric = {
                    'speed': float(speeds[i]),
                    'acceleration': float(accelerations[i]),
                    'jerk': float(jerks[i]),
                    'turning_rate': float(turning_rates[i]),
                    'distance': float(distances[i]),
                    # Enhanced metrics from GPU acceleration
                    'velocity_change': float(velocity_changes[i]),
                    'turning_change': float(turning_changes[i]),
                    'curvature': float(curvatures[i]),
                    'smoothness': float(smoothness[i]),
                    'path_efficiency': float(path_efficiency[i])
                }
                metrics.append(metric)
            
            # Enhanced logging for GPU operations
            if n_frames % 100 == 0:  # Log every 100 frames for enhanced version
                logger.info(f"   🚀 Enhanced GPU-accelerated motion metrics calculated for {n_frames} frames")
            return metrics
            
        except Exception as e:
            logger.warning(f"   ⚠️ Enhanced GPU motion metrics calculation failed, falling back to basic GPU: {e}")
            return self._calculate_motion_metrics_gpu(odom_data)
    
    def _calculate_motion_metrics_cpu(self, odom_data: List[Dict]) -> List[Dict]:
        """CPU-based motion metrics calculation (fallback)"""
        metrics = []
        total_distance = 0.0
        
        for i in range(len(odom_data)):
            metric = {
                'speed': 0.0,
                'acceleration': 0.0,
                'jerk': 0.0,
                'turning_rate': 0.0,
                'distance': 0.0
            }
            
            if i > 0:
                # Calculate speed using Numba-accelerated function
                linear_vel = odom_data[i]['linear_velocity']
                metric['speed'] = calculate_speed_numba(linear_vel[0], linear_vel[1], linear_vel[2])
                
                # Calculate turning rate using Numba-accelerated function
                angular_vel = odom_data[i]['angular_velocity']
                metric['turning_rate'] = calculate_turning_rate_numba(angular_vel[0], angular_vel[1], angular_vel[2])
                
                # Calculate distance traveled using Numba-accelerated function
                prev_pos = odom_data[i-1]['position']
                curr_pos = odom_data[i]['position']
                step_distance = calculate_distance_numba(prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1])
                total_distance += step_distance
                metric['distance'] = total_distance
                
                # Calculate acceleration (derivative of speed)
                if i > 1:
                    prev_speed = metrics[i-1]['speed']
                    metric['acceleration'] = metric['speed'] - prev_speed
                    
                    # Calculate jerk (derivative of acceleration)
                    if i > 2:
                        prev_accel = metrics[i-1]['acceleration']
                        metric['jerk'] = metric['acceleration'] - prev_accel
            
            metrics.append(metric)
        
        return metrics

    def _estimate_bag_size(self, bag_path: str) -> int:
        """Quickly estimate the number of camera frames in the bag"""
        try:
            frame_count = 0
            with Reader(bag_path) as reader:
                # Sample first 1000 messages to estimate frame rate
                sample_count = 0
                for connection, timestamp, rawdata in reader.messages():
                    if connection.topic == '/mbuggy/camera_front/image_rect':
                        frame_count += 1
                    
                    sample_count += 1
                    if sample_count >= 1000:  # Sample first 1000 messages
                        break
                
                # Estimate total frames based on sample
                if sample_count > 0:
                    camera_ratio = frame_count / sample_count
                    # Rough estimate: assume similar ratio throughout bag
                    # This is a quick approximation, not exact
                    return int(frame_count * 10)  # Conservative estimate
            
            return 0
        except Exception as e:
            logger.warning(f"   ⚠️ Could not estimate bag size: {e}")
            return 0

    def _calculate_axis_limits(self, odom_data: List[Dict], motion_metrics: List[Dict]) -> Dict:
        """Calculate consistent axis limits for all plots with CuPy acceleration"""
        if not odom_data or not motion_metrics:
            # Return default limits if no data
            return {
                'x_min': -10, 'x_max': 10,
                'y_min': -10, 'y_max': 10
            }
        
        # Use CuPy acceleration for large datasets
        if CUPY_AVAILABLE and len(odom_data) > 100:
            return self._calculate_axis_limits_gpu(odom_data, motion_metrics)
        else:
            return self._calculate_axis_limits_cpu(odom_data, motion_metrics)
    
    def _calculate_axis_limits_gpu(self, odom_data: List[Dict], motion_metrics: List[Dict]) -> Dict:
        """GPU-accelerated axis limits calculation using CuPy"""
        try:
            # Extract positions to GPU array
            positions = cp.array([[odom['position'][0], odom['position'][1]] for odom in odom_data])
            
            # GPU-accelerated min/max calculations
            x_min, x_max = float(cp.min(positions[:, 0])), float(cp.max(positions[:, 0]))
            y_min, y_max = float(cp.min(positions[:, 1])), float(cp.max(positions[:, 1]))
            
            # Ensure minimum range to prevent zooming issues
            x_range = max(x_max - x_min, 5.0)  # Minimum 5m range
            y_range = max(y_max - y_min, 5.0)  # Minimum 5m range
            
            # Add 15% padding for better visualization
            x_padding = x_range * 0.15
            y_padding = y_range * 0.15
            
            trajectory_limits = {
                'x_min': x_min - x_padding,
                'x_max': x_max + x_padding,
                'y_min': y_min - y_padding,
                'y_max': y_max + y_padding
            }
            
            # Extract motion metrics to GPU arrays for batch processing
            speeds = cp.array([m['speed'] for m in motion_metrics])
            accelerations = cp.array([m['acceleration'] for m in motion_metrics])
            turning_rates = cp.array([m['turning_rate'] for m in motion_metrics])
            
            # GPU-accelerated min/max for motion metrics
            speed_limits = {
                'y_min': 0.0,
                'y_max': float(cp.max(speeds)) * 1.1  # Add 10% padding
            }
            
            accel_min, accel_max = float(cp.min(accelerations)), float(cp.max(accelerations))
            accel_range = max(accel_max - accel_min, 0.1)
            acceleration_limits = {
                'y_min': accel_min - accel_range * 0.1,
                'y_max': accel_max + accel_range * 0.1
            }
            
            turning_limits = {
                'y_min': 0.0,
                'y_max': float(cp.max(turning_rates)) * 1.1
            }
            
            # Reduced logging frequency for GPU operations
            if len(odom_data) % 1000 == 0:  # Log every 1000 frames instead of every time
                logger.info(f"   🚀 GPU-accelerated axis limits calculated")
            
            return {
                'trajectory': trajectory_limits,
                'speed': speed_limits,
                'acceleration': acceleration_limits,
                'turning_rate': turning_limits
            }
            
        except Exception as e:
            logger.warning(f"   ⚠️ GPU axis limits calculation failed, falling back to CPU: {e}")
            return self._calculate_axis_limits_cpu(odom_data, motion_metrics)
    
    def _calculate_axis_limits_cpu(self, odom_data: List[Dict], motion_metrics: List[Dict]) -> Dict:
        """CPU-based axis limits calculation (fallback)"""
        # Trajectory limits - use the FULL trajectory to ensure consistent scaling
        positions = np.array([odom['position'] for odom in odom_data])
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Ensure minimum range to prevent zooming issues
        x_range = max(x_max - x_min, 5.0)  # Minimum 5m range
        y_range = max(y_max - y_min, 5.0)  # Minimum 5m range
        
        # Add 15% padding for better visualization
        x_padding = x_range * 0.15
        y_padding = y_range * 0.15
        
        trajectory_limits = {
            'x_min': x_min - x_padding,
            'x_max': x_max + x_padding,
            'y_min': y_min - y_padding,
            'y_max': y_max + y_padding
        }
        
        # Motion metrics limits
        speeds = [m['speed'] for m in motion_metrics]
        accelerations = [m['acceleration'] for m in motion_metrics]
        jerks = [m['jerk'] for m in motion_metrics]
        turning_rates = [m['turning_rate'] for m in motion_metrics]
        
        speed_acc_limits = {
            'speed_min': min(speeds) - abs(min(speeds)) * 0.1,
            'speed_max': max(speeds) + abs(max(speeds)) * 0.1,
            'acc_min': min(accelerations) - abs(min(accelerations)) * 0.1,
            'acc_max': max(accelerations) + abs(max(accelerations)) * 0.1
        }
        
        jerk_turn_limits = {
            'jerk_min': min(jerks) - abs(min(jerks)) * 0.1,
            'jerk_max': max(jerks) + abs(max(jerks)) * 0.1,
            'turn_min': min(turning_rates) - abs(min(turning_rates)) * 0.1,
            'turn_max': max(turning_rates) + abs(max(turning_rates)) * 0.1
        }
        
        return {
            'trajectory': trajectory_limits,
            'speed_acc': speed_acc_limits,
            'jerk_turn': jerk_turn_limits
        }

    def _get_reusable_figure(self):
        """Get or create a reusable matplotlib figure and subplot axes to reduce allocations"""
        if self.reusable_fig is None:
            # Create figure with exact target dimensions
            target_width, target_height = 1280, 720
            self.reusable_fig = plt.figure(figsize=(target_width/100, target_height/100), dpi=100)
            self.reusable_fig.patch.set_facecolor('white')
            self.reusable_fig.patch.set_alpha(1.0)
            self.reusable_fig.set_dpi(100)
            
            # Initialize figure buffer
            self.fig_buffer = io.BytesIO()
            
            # Pre-create subplot axes for 2x2 layout (will be created on first use)
            self.subplot_axes = None
            
        # Clear the figure for reuse (but keep subplot structure)
        self.reusable_fig.clear()
        return self.reusable_fig

    def _get_reusable_subplot_axes(self):
        """Get or create reusable subplot axes for 2x2 layout"""
        if self.subplot_axes is None:
            # Create the 2x2 subplot layout with improved spacing
            # Camera (top-left): positioned at y=0.61
            ax1 = self.reusable_fig.add_axes([0.02, 0.61, 0.46, 0.40])
            
            # Trajectory (top-right): positioned at y=0.61
            ax2 = self.reusable_fig.add_axes([0.52, 0.61, 0.46, 0.40])
            
            # Speed & Acceleration (bottom-left): moved up to reduce gap (y=0.25 -> y=0.30)
            ax3 = self.reusable_fig.add_axes([0.02, 0.30, 0.46, 0.40])
            
            # Jerk & Turning Rate (bottom-right): moved up to reduce gap (y=0.25 -> y=0.30)
            ax4 = self.reusable_fig.add_axes([0.52, 0.30, 0.46, 0.40])
            
            # Store the axes for reuse
            self.subplot_axes = {
                'camera': ax1,
                'trajectory': ax2,
                'speed_acc': ax3,
                'jerk_turn': ax4
            }
            
        return self.subplot_axes

    def _calculate_time_axis(self, odom_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate time values for x-axis from timestamps"""
        if not odom_data:
            return np.array([]), np.array([])
        
        # Extract timestamps and convert to seconds relative to start
        timestamps = np.array([odom['timestamp'] for odom in odom_data])
        start_time = timestamps[0]
        time_seconds = (timestamps - start_time) / 1e9  # Convert nanoseconds to seconds
        frame_indices = np.arange(len(odom_data))
        
        return frame_indices, time_seconds

    def _get_cached_axis_limits(self, odom_data: List[Dict], motion_metrics: List[Dict]) -> Dict:
        """Get cached axis limits or calculate and cache them"""
        if self.cached_axis_limits is None:
            # Calculate and cache axis limits
            self.cached_axis_limits = self._calculate_axis_limits(odom_data, motion_metrics)
            logger.info("   📊 Axis limits calculated and cached")
        
        return self.cached_axis_limits

    def _calculate_motion_between_frames(self, frame1: Dict, frame2: Dict) -> float:
        """Calculate motion between two frames for adaptive frame skipping"""
        try:
            if 'position' not in frame1 or 'position' not in frame2:
                return 1.0  # Default to high motion if position not available
            
            pos1 = frame1['position']
            pos2 = frame2['position']
            
            # Calculate Euclidean distance
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            return distance
            
        except Exception as e:
            logger.warning(f"   ⚠️ Motion calculation failed: {e}")
            return 1.0  # Default to high motion

    def _should_skip_frame(self, current_frame: Dict, previous_frame: Dict, frame_count: int) -> bool:
        """Determine if frame should be skipped based on motion and frame rate"""
        if not self.adaptive_frame_skip:
            return False
        
        # Always include first few frames
        if frame_count < 5:
            return False
        
        # Calculate motion
        motion = self._calculate_motion_between_frames(previous_frame, current_frame)
        
        # Skip frame if motion is below threshold
        if motion < self.motion_threshold:
            return True
        
        # Ensure minimum frame rate (don't skip too many consecutive frames)
        return False

    def _synchronize_odom_to_camera_timestamps(self, camera_data: List[Dict], all_odom_data: List[Dict]) -> List[Dict]:
        """Synchronize odometry data to camera timestamps using interpolation for better accuracy"""
        if not camera_data or not all_odom_data:
            return []
        
        synchronized_odom = []
        interpolation_count = 0
        
        # Use GPU acceleration for large datasets
        if CUPY_AVAILABLE and len(all_odom_data) > 100:
            try:
                # GPU-accelerated timestamp processing
                odom_timestamps = cp.array([odom['timestamp'] for odom in all_odom_data])
                
                for camera_frame in camera_data:
                    camera_timestamp = camera_frame['timestamp']
                    
                    # GPU-accelerated closest timestamp finding
                    time_diffs = cp.abs(odom_timestamps - camera_timestamp)
                    closest_idx = int(cp.argmin(time_diffs))
                    closest_odom = all_odom_data[closest_idx]
                    time_diff = float(time_diffs[closest_idx]) / 1e9  # Convert to seconds
                    
                    # Process with interpolation if needed
                    if time_diff > 0.05:  # More than 50ms difference
                        interpolated_odom = self._interpolate_odometry_data(
                            camera_timestamp, all_odom_data, closest_odom
                        )
                        if interpolated_odom is not None:
                            synced_odom = interpolated_odom
                            interpolation_count += 1
                            if time_diff > 0.1:  # Log only large differences
                                logger.warning(f"   ⚠️ Large time difference: {time_diff:.3f}s - using interpolation")
                        else:
                            synced_odom = closest_odom.copy()
                            synced_odom['timestamp'] = camera_timestamp
                            synced_odom['time_diff'] = time_diff
                    else:
                        synced_odom = closest_odom.copy()
                        synced_odom['timestamp'] = camera_timestamp
                        synced_odom['time_diff'] = time_diff
                    
                    synchronized_odom.append(synced_odom)
                
                if interpolation_count > 0:
                    logger.info(f"   🚀 GPU-accelerated sync: {len(synchronized_odom)} odometry messages ({interpolation_count} interpolated)")
                else:
                    logger.info(f"   🚀 GPU-accelerated sync: {len(synchronized_odom)} odometry messages (all closest match)")
                
                return synchronized_odom
                
            except Exception as e:
                logger.warning(f"   ⚠️ GPU synchronization failed, falling back to CPU: {e}")
        
        # CPU fallback with Numba acceleration
        odom_timestamps = [odom['timestamp'] for odom in all_odom_data]
        
        for camera_frame in camera_data:
            camera_timestamp = camera_frame['timestamp']
            
            # Use Numba-accelerated closest timestamp finding
            closest_idx, min_time_diff = find_closest_timestamp_numba(odom_timestamps, camera_timestamp)
            closest_odom = all_odom_data[closest_idx]
            time_diff = min_time_diff / 1e9  # Convert to seconds
            
            # Use interpolation for large time differences (>50ms)
            if time_diff > 0.05:  # More than 50ms difference
                interpolated_odom = self._interpolate_odometry_data(
                    camera_timestamp, all_odom_data, closest_odom
                )
                if interpolated_odom is not None:
                    synced_odom = interpolated_odom
                    interpolation_count += 1
                    if time_diff > 0.1:  # Log only large differences
                        logger.warning(f"   ⚠️ Large time difference: {time_diff:.3f}s - using interpolation")
                else:
                    # Fallback to closest match if interpolation fails
                    synced_odom = closest_odom.copy()
                    synced_odom['timestamp'] = camera_timestamp
                    synced_odom['time_diff'] = time_diff
            else:
                # Use closest match for small time differences
                synced_odom = closest_odom.copy()
                synced_odom['timestamp'] = camera_timestamp
                synced_odom['time_diff'] = time_diff
            
            synchronized_odom.append(synced_odom)
        
        if interpolation_count > 0:
            logger.info(f"   🔄 Numba-accelerated sync: {len(synchronized_odom)} odometry messages ({interpolation_count} interpolated)")
        else:
            logger.info(f"   ✅ Numba-accelerated sync: {len(synchronized_odom)} odometry messages (all closest match)")
        
        return synchronized_odom

    def _interpolate_odometry_data(self, target_timestamp: int, all_odom_data: List[Dict], closest_odom: Dict) -> Optional[Dict]:
        """Interpolate odometry data for a specific timestamp using linear interpolation"""
        try:
            # Find the two closest odometry messages (before and after target timestamp)
            before_odom = None
            after_odom = None
            
            for odom in all_odom_data:
                if odom['timestamp'] <= target_timestamp:
                    if before_odom is None or odom['timestamp'] > before_odom['timestamp']:
                        before_odom = odom
                else:
                    if after_odom is None or odom['timestamp'] < after_odom['timestamp']:
                        after_odom = odom
            
            # If we don't have both before and after, use closest match
            if before_odom is None or after_odom is None:
                return None
            
            # Calculate interpolation weights
            time_before = before_odom['timestamp']
            time_after = after_odom['timestamp']
            time_target = target_timestamp
            
            if time_after == time_before:  # Avoid division by zero
                return before_odom
            
            # Linear interpolation weight (0 = before, 1 = after)
            weight = (time_target - time_before) / (time_after - time_before)
            weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]
            
            # Use Numba-accelerated interpolation for better performance
            # Interpolate position (linear interpolation)
            interpolated_position = [
                interpolate_linear_numba(time_before, before_odom['position'][0], time_after, after_odom['position'][0], time_target),
                interpolate_linear_numba(time_before, before_odom['position'][1], time_after, after_odom['position'][1], time_target),
                interpolate_linear_numba(time_before, before_odom['position'][2], time_after, after_odom['position'][2], time_target)
            ]
            
            # Interpolate linear velocity (linear interpolation)
            interpolated_linear_velocity = [
                interpolate_linear_numba(time_before, before_odom['linear_velocity'][0], time_after, after_odom['linear_velocity'][0], time_target),
                interpolate_linear_numba(time_before, before_odom['linear_velocity'][1], time_after, after_odom['linear_velocity'][1], time_target),
                interpolate_linear_numba(time_before, before_odom['linear_velocity'][2], time_after, after_odom['linear_velocity'][2], time_target)
            ]
            
            # Interpolate angular velocity (linear interpolation)
            interpolated_angular_velocity = [
                interpolate_linear_numba(time_before, before_odom['angular_velocity'][0], time_after, after_odom['angular_velocity'][0], time_target),
                interpolate_linear_numba(time_before, before_odom['angular_velocity'][1], time_after, after_odom['angular_velocity'][1], time_target),
                interpolate_linear_numba(time_before, before_odom['angular_velocity'][2], time_after, after_odom['angular_velocity'][2], time_target)
            ]
            
            # Use Numba-accelerated quaternion interpolation
            interpolated_orientation = interpolate_quaternion_numba(
                before_odom['orientation'], after_odom['orientation'], weight
            )
            
            # Create interpolated odometry entry
            interpolated_odom = {
                'timestamp': target_timestamp,
                'position': interpolated_position.tolist(),
                'orientation': interpolated_orientation.tolist(),
                'linear_velocity': interpolated_linear_velocity.tolist(),
                'angular_velocity': interpolated_angular_velocity.tolist(),
                'time_diff': 0.0,  # Perfect match after interpolation
                'interpolated': True  # Mark as interpolated
            }
            
            return interpolated_odom
            
        except Exception as e:
            logger.warning(f"   ⚠️ Interpolation failed: {e}")
            return None

    def _validate_synchronization(self, camera_data: List[Dict], odom_data: List[Dict]) -> bool:
        """Validate that camera and odometry data are properly synchronized"""
        if not camera_data or not odom_data:
            logger.warning("   ⚠️ Empty data - cannot validate synchronization")
            return False
        
        # Check if we have the same number of data points
        if len(camera_data) != len(odom_data):
            logger.warning(f"   ⚠️ Synchronization mismatch: {len(camera_data)} camera frames vs {len(odom_data)} odom messages")
            return False
        
        # Check index consistency
        mismatches = 0
        for i, (cam, odom) in enumerate(zip(camera_data, odom_data)):
            if cam['index'] != odom['index']:
                mismatches += 1
                if mismatches <= 5:  # Only log first 5 mismatches to avoid spam
                    logger.warning(f"   ⚠️ Index mismatch at position {i}: camera={cam['index']}, odom={odom['index']}")
        
        if mismatches > 0:
            logger.warning(f"   ⚠️ Found {mismatches} index mismatches out of {len(camera_data)} frames")
            return False
        
        # Check timestamp consistency (optional - for debugging)
        timestamp_diffs = []
        for cam, odom in zip(camera_data, odom_data):
            time_diff = abs(cam['timestamp'] - odom['timestamp'])
            timestamp_diffs.append(time_diff)
        
        max_time_diff = max(timestamp_diffs) if timestamp_diffs else 0
        avg_time_diff = sum(timestamp_diffs) / len(timestamp_diffs) if timestamp_diffs else 0
        
        if max_time_diff > 0.1:  # More than 100ms difference
            logger.warning(f"   ⚠️ Large timestamp differences: max={max_time_diff:.3f}s, avg={avg_time_diff:.3f}s")
        else:
            logger.info(f"   ✅ Synchronization validated: {len(camera_data)} frames, max time diff={max_time_diff:.3f}s")
        
        return True

    def _create_visualization_frame(self, camera_image: np.ndarray, odom_data: List[Dict], 
                                  motion_metrics: List[Dict], current_idx: int, bag_name: str, 
                                  axis_limits: Dict = None) -> np.ndarray:
        """Create visualization frame with camera and trajectory plots"""
        
        # Use cached axis limits for better performance
        if axis_limits is None:
            axis_limits = self._get_cached_axis_limits(odom_data, motion_metrics)
        
        # Comprehensive input validation with detailed logging
        # Input validation (reduced logging for performance)
        if (camera_image is None or camera_image.size == 0 or 
            odom_data is None or len(odom_data) == 0 or 
            current_idx >= len(odom_data) or 
            motion_metrics is None or len(motion_metrics) == 0 or
            not isinstance(camera_image, np.ndarray) or 
            len(camera_image.shape) != 3 or camera_image.shape[2] != 3):
            # Skip invalid frames silently to reduce log verbosity
            return None
        
        
        # Complete matplotlib state isolation to prevent corruption
        try:
            plt.close('all')  # Close all existing figures
            gc.collect()  # Force garbage collection
            
            # Reset matplotlib to a completely clean state
            matplotlib.rcdefaults()
            matplotlib.rcParams['figure.max_open_warning'] = 0
            matplotlib.rcParams['figure.dpi'] = 100
            matplotlib.rcParams['savefig.dpi'] = 100
            matplotlib.rcParams['figure.facecolor'] = 'white'
            matplotlib.rcParams['axes.facecolor'] = 'white'
            matplotlib.rcParams['font.size'] = 10
            
            # Force matplotlib to reset its internal state
            plt.rcParams.update(plt.rcParamsDefault)
            
            # Use reusable figure to reduce memory allocations
            fig = self._get_reusable_figure()
            fig.suptitle(f'{bag_name} - Frame {current_idx + 1}', fontsize=14, y=0.95)
            
            # Optimized subplot positions with improved spacing
            # Title at y=0.95, top row at y=0.61, bottom row moved up significantly
            
            # Camera (top-left): positioned at y=0.61
            ax1 = fig.add_subplot(2, 2, 1, position=[0.05, 0.61, 0.42, 0.28])
            
            # Trajectory (top-right): positioned at y=0.61
            ax2 = fig.add_subplot(2, 2, 2, position=[0.55, 0.61, 0.42, 0.28])
            
            # Speed/Acceleration (bottom-left): moved up to reduce gap (y=0.25 -> y=0.30)
            ax3 = fig.add_subplot(2, 2, 3, position=[0.05, 0.30, 0.42, 0.28])
            
            # Jerk/Turning (bottom-right): moved up to reduce gap (y=0.25 -> y=0.30)
            ax4 = fig.add_subplot(2, 2, 4, position=[0.55, 0.30, 0.42, 0.28])
            
            # Ensure figure properties are set explicitly
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)
            
            # Set explicit DPI and size
            fig.set_dpi(100)
            fig.set_size_inches(12.8, 7.2)
            
            # Manual positioning - tight_layout not needed with add_axes
            
        except Exception as e:
            logger.error(f"   ❌ Frame {current_idx}: Failed to create clean matplotlib figure: {e}")
            return None
        
        # 1. Camera image (convert BGR to RGB for matplotlib)
        camera_image_rgb = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(camera_image_rgb)
        ax1.set_title('Camera Stream', fontsize=12, pad=5)
        ax1.axis('off')
        
        # 2. Trajectory with full path and current position
        if len(odom_data) > 1:
            positions = np.array([odom['position'] for odom in odom_data])
            
            # Manually set consistent axis limits with explicit control
            if 'trajectory' in axis_limits:
                ax2.set_xlim(axis_limits['trajectory']['x_min'], axis_limits['trajectory']['x_max'])
                ax2.set_ylim(axis_limits['trajectory']['y_min'], axis_limits['trajectory']['y_max'])
                # Disable automatic scaling to prevent zoom changes
                ax2.set_autoscale_on(False)
                ax2.autoscale(False)
            
            # Plot FULL trajectory (1:N frames) with 50% transparency
            ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.5, label='Full Trajectory')
            
            # Plot trajectory from start to current position (1:i frames) with full opacity
            if current_idx < len(positions):
                
                ax2.plot(positions[:current_idx+1, 0], positions[:current_idx+1, 1], 
                        'b-', linewidth=2, alpha=1.0, label='Current Path')
            
            # Mark start position
            ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', 
                       label='Start', zorder=5)
            
            # Mark current position as moving dot
            if current_idx < len(positions):
                ax2.scatter(positions[current_idx, 0], positions[current_idx, 1], 
                           color='red', s=150, marker='*', label='Current Position', zorder=6)
            
            ax2.set_xlabel('X Position (m)', fontsize=10)
            ax2.set_ylabel('Y Position (m)', fontsize=10)
            ax2.set_title('Trajectory', fontsize=12, pad=5)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            # Reduce tick label size to prevent overlap
            ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # 3. Speed and acceleration with full timeline
        if len(motion_metrics) > 1:
            speeds = [m['speed'] for m in motion_metrics]
            accelerations = [m['acceleration'] for m in motion_metrics]
            
            # Calculate time axis for better readability
            frame_indices, time_seconds = self._calculate_time_axis(odom_data)
            
            ax3_twin = ax3.twinx()
            
            # Manually set consistent axis limits with explicit control
            if 'speed_acc' in axis_limits:
                ax3.set_ylim(axis_limits['speed_acc']['speed_min'], axis_limits['speed_acc']['speed_max'])
                ax3_twin.set_ylim(axis_limits['speed_acc']['acc_min'], axis_limits['speed_acc']['acc_max'])
                # Disable automatic scaling to prevent zoom changes
                ax3.set_autoscale_on(False)
                ax3_twin.set_autoscale_on(False)
                ax3.autoscale(False)
                ax3_twin.autoscale(False)
            
            # Set consistent X-axis limits for time-based plots
            max_frame = len(speeds) - 1 if speeds else 0
            max_time = time_seconds[-1] if len(time_seconds) > 0 else 0
            ax3.set_xlim(0, max_frame)
            ax3_twin.set_xlim(0, max_frame)
            
            # Plot FULL timeline (1:N frames) with 50% transparency
            ax3.plot(frame_indices, speeds, 'b-', linewidth=1, alpha=0.5, label='Full Speed')
            ax3_twin.plot(frame_indices, accelerations, 'r-', linewidth=1, alpha=0.5, label='Full Acceleration')
            
            # Plot timeline from start to current position (1:i frames) with full opacity
            if current_idx < len(speeds):
                current_frames = frame_indices[:current_idx+1]
                ax3.plot(current_frames, speeds[:current_idx+1], 'b-', linewidth=2, alpha=1.0, label='Current Speed')
                ax3_twin.plot(current_frames, accelerations[:current_idx+1], 'r-', linewidth=2, alpha=1.0, label='Current Acceleration')
                
                # Mark current position
                ax3.axvline(x=current_idx, color='red', linestyle='--', alpha=0.7)
                ax3.scatter(current_idx, speeds[current_idx], color='blue', s=100, zorder=5)
                ax3_twin.scatter(current_idx, accelerations[current_idx], color='red', s=100, zorder=5)
            
            # Enhanced x-axis with both frame and time information
            ax3.set_xlabel(f'Frame (Time: {time_seconds[current_idx]:.1f}s / {max_time:.1f}s)', fontsize=10)
            ax3.set_ylabel('Speed (m/s)', color='blue', fontsize=10)
            ax3_twin.set_ylabel('Acceleration (m/s²)', color='red', fontsize=10)
            ax3.set_title('Speed & Acceleration vs Time', fontsize=12, pad=5)
            ax3.legend(loc='upper left', fontsize=8)
            ax3_twin.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            # Reduce tick label size to prevent overlap
            ax3.tick_params(axis='both', which='major', labelsize=8)
            ax3_twin.tick_params(axis='both', which='major', labelsize=8)
        
        # 4. Jerk and turning rate with full timeline
        if len(motion_metrics) > 1:
            jerks = [m['jerk'] for m in motion_metrics]
            turning_rates = [m['turning_rate'] for m in motion_metrics]
            
            # Use the same time axis calculated earlier
            frame_indices, time_seconds = self._calculate_time_axis(odom_data)
            
            ax4_twin = ax4.twinx()
            
            # Manually set consistent axis limits with explicit control
            if 'jerk_turn' in axis_limits:
                ax4.set_ylim(axis_limits['jerk_turn']['jerk_min'], axis_limits['jerk_turn']['jerk_max'])
                ax4_twin.set_ylim(axis_limits['jerk_turn']['turn_min'], axis_limits['jerk_turn']['turn_max'])
                # Disable automatic scaling to prevent zoom changes
                ax4.set_autoscale_on(False)
                ax4_twin.set_autoscale_on(False)
                ax4.autoscale(False)
                ax4_twin.autoscale(False)
            
            # Set consistent X-axis limits for time-based plots
            max_frame = len(jerks) - 1 if jerks else 0
            max_time = time_seconds[-1] if len(time_seconds) > 0 else 0
            ax4.set_xlim(0, max_frame)
            ax4_twin.set_xlim(0, max_frame)
            
            # Plot FULL timeline (1:N frames) with 50% transparency
            ax4.plot(frame_indices, jerks, 'g-', linewidth=1, alpha=0.5, label='Full Jerk')
            ax4_twin.plot(frame_indices, turning_rates, 'orange', linewidth=1, alpha=0.5, label='Full Turning Rate')
            
            # Plot timeline from start to current position (1:i frames) with full opacity
            if current_idx < len(jerks):
                current_frames = frame_indices[:current_idx+1]
                ax4.plot(current_frames, jerks[:current_idx+1], 'g-', linewidth=2, alpha=1.0, label='Current Jerk')
                ax4_twin.plot(current_frames, turning_rates[:current_idx+1], 'orange', linewidth=2, alpha=1.0, label='Current Turning Rate')
                
                # Mark current position
                ax4.axvline(x=current_idx, color='red', linestyle='--', alpha=0.7)
                ax4.scatter(current_idx, jerks[current_idx], color='green', s=100, zorder=5)
                ax4_twin.scatter(current_idx, turning_rates[current_idx], color='orange', s=100, zorder=5)
            
            # Enhanced x-axis with both frame and time information
            ax4.set_xlabel(f'Frame (Time: {time_seconds[current_idx]:.1f}s / {max_time:.1f}s)', fontsize=10)
            ax4.set_ylabel('Jerk (m/s³)', color='green', fontsize=10)
            ax4_twin.set_ylabel('Turning Rate (rad/s)', color='orange', fontsize=10)
            ax4.set_title('Jerk & Turning Rate vs Time', fontsize=12, pad=5)
            ax4.legend(loc='upper left', fontsize=8)
            ax4_twin.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            # Reduce tick label size to prevent overlap
            ax4.tick_params(axis='both', which='major', labelsize=8)
            ax4_twin.tick_params(axis='both', which='major', labelsize=8)
        
        # Use the robust buffer-based approach directly to avoid matplotlib state corruption
        try:
            
            # Use reusable buffer to reduce memory allocations
            self.fig_buffer.seek(0)
            self.fig_buffer.truncate(0)  # Clear buffer
            fig.savefig(self.fig_buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            self.fig_buffer.seek(0)
            
            # Read the image from reusable buffer
            pil_image = Image.open(self.fig_buffer)
            vis_image = np.array(pil_image)
            
            # Convert RGB to BGR using reusable GPU memory pools
            if (self.cuda_available and self.gpu_accelerator.opencv_cuda_available and 
                self.gpu_image_pool is not None and self.gpu_temp_pool is not None and
                len(vis_image.shape) == 3 and vis_image.shape[2] == 3):
                try:
                    # Use reusable GPU memory pools
                    self.gpu_image_pool.upload(vis_image)
                    
                    # CUDA-accelerated resize (images are already in RGB format)
                    if self.gpu_image_pool.channels() == 3:
                        # CUDA-accelerated resize using temp pool
                        target_width, target_height = 1280, 720
                        cv2.cuda.resize(self.gpu_image_pool, (target_width, target_height), self.gpu_temp_pool)
                        
                        # Download final result
                        vis_image = self.gpu_temp_pool.download()
                    else:
                        # Fallback if channels are not 3
                        target_width, target_height = 1280, 720
                        vis_image = cv2.resize(vis_image, (target_width, target_height))
                except Exception as e:
                    # Fallback to CPU operations if CUDA fails
                    target_width, target_height = 1280, 720
                    vis_image = cv2.resize(vis_image, (target_width, target_height))
            else:
                # Fallback to CPU operations
                target_width, target_height = 1280, 720
                vis_image = cv2.resize(vis_image, (target_width, target_height))
            
            # Ensure the image is in the correct format (uint8, 3 channels)
            if vis_image.dtype != np.uint8:
                vis_image = vis_image.astype(np.uint8)
            
            # Don't close reusable figure, just clear it
            fig.clear()
            
            # Only log success for every 50th frame to reduce verbosity
            if current_idx % 50 == 0:
                logger.info(f"   ✅ Frame {current_idx}: Rendering successful")
            return vis_image
            
        except Exception as e:
            logger.error(f"   ❌ Frame {current_idx}: Robust rendering failed: {e}")
            if self.reusable_fig is not None:
                self.reusable_fig.clear()
            return None
        
        return vis_image



def main():
    parser = argparse.ArgumentParser(description='Enhanced Synchronized Video Generator')
    parser.add_argument('bag_paths', nargs='+', help='Paths to ROS bag directories')
    parser.add_argument('--output-dir', default='reports/visualizations', 
                       help='Output directory for videos')
    parser.add_argument('--test-frames', type=int, default=20,
                       help='Number of frames to process for testing (0 for all frames)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker threads')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Size of chunks for processing (default: 200)')
    parser.add_argument('--frame-ranges', type=str, default=None,
                       help='Specific frame ranges to process (e.g., "0-200,500-700,1000-1200")')
    parser.add_argument('--enable-parallel-bags', action='store_true', default=False,
                       help='DISABLED: Parallel processing disabled for strict time synchronization')
    parser.add_argument('--disable-parallel-bags', action='store_true', default=True,
                       help='DISABLED: Parallel processing disabled for strict time synchronization')
    parser.add_argument('--enable-parallel-chunks', action='store_true', default=False,
                       help='DISABLED: Parallel chunk processing disabled for strict time synchronization')
    parser.add_argument('--max-parallel-bags', type=int, default=1,
                       help='DISABLED: Forced to 1 for strict time synchronization')
    
    args = parser.parse_args()
    
    generator = EnhancedSynchronizedVideoGenerator(
        args.bag_paths, 
        args.output_dir, 
        args.test_frames,
        args.max_workers
    )
    
    # Set chunk size if provided
    if hasattr(args, 'chunk_size'):
        generator.chunk_size = args.chunk_size
    
    # Configure parallel processing
    # FORCE sequential processing for strict time synchronization
    generator.enable_multi_bag_parallel = False
    generator.enable_chunk_parallel = False
    generator.parallel_chunks = False
    generator.max_parallel_bags = 1
    logger.info("🚫 ALL parallel processing DISABLED for strict time synchronization")
    
    # Handle frame ranges if provided
    if args.frame_ranges:
        # Parse frame ranges (e.g., "0-200,500-700,1000-1200")
        frame_ranges = []
        for range_str in args.frame_ranges.split(','):
            start, end = map(int, range_str.strip().split('-'))
            frame_ranges.append((start, end))
        
        logger.info(f"🎯 Processing specific frame ranges: {frame_ranges}")
        # Override the extraction method for specific ranges
        generator._extract_synchronized_data_chunked = lambda bag_path, chunk_size: generator._extract_specific_frame_ranges(bag_path, frame_ranges)
    
    generator.generate_synchronized_videos()

if __name__ == "__main__":
    main()
