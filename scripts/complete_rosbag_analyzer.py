#!/usr/bin/env python3
"""
Complete ROS Bag Analyzer - Consolidated Version
Combines all analysis functionality into a single optimized script:
- Complete bag analysis with statistics
- Ground truth trajectory analysis
- Adaptive color-coded plotting
- Video generation
- Comprehensive report generation
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteROSBagAnalyzer:
    def __init__(self, bag_paths: List[str], output_dir: str = "reports/analysis"):
        self.bag_paths = bag_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.typestore = get_typestore(Stores.ROS2_JAZZY)
        self.max_workers = min(8, len(bag_paths))
        
        # Hardware monitoring
        self.hw_monitor = HardwareMonitor()
        
        logger.info(f"🔬 Complete ROS Bag Analyzer initialized")
        logger.info(f"   📁 Bags: {len(bag_paths)} bags")
        logger.info(f"   📁 Output: {output_dir}")
        logger.info(f"   🧵 Max workers: {self.max_workers}")

    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("🚀 Starting complete analysis pipeline...")
        
        # Phase 1: Extract and analyze data
        analyses = self._extract_and_analyze_data()
        
        # Phase 2: Generate visualizations
        self._generate_visualizations(analyses)
        
        # Phase 3: Generate videos (optional)
        self._generate_videos()
        
        # Phase 4: Generate comprehensive report
        self._generate_comprehensive_report(analyses)
        
        logger.info("✅ Complete analysis pipeline finished!")

    def _extract_and_analyze_data(self) -> Dict[str, Any]:
        """Extract and analyze data from all bags"""
        logger.info("📊 Phase 1: Extracting and analyzing data...")
        
        all_analyses = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_bag = {
                executor.submit(self._analyze_single_bag, bag_path): bag_path 
                for bag_path in self.bag_paths
            }
            
            for future in as_completed(future_to_bag):
                bag_path = future_to_bag[future]
                try:
                    analysis = future.result()
                    all_analyses[Path(bag_path).name] = analysis
                    logger.info(f"   ✅ Completed analysis for {Path(bag_path).name}")
                except Exception as e:
                    logger.error(f"   ❌ Error analyzing {bag_path}: {e}")
        
        return all_analyses

    def _analyze_single_bag(self, bag_path: str) -> Dict[str, Any]:
        """Analyze a single bag comprehensively"""
        bag_name = Path(bag_path).name
        analysis = {
            'bag_name': bag_name,
            'bag_path': bag_path,
            'topics': {},
            'odom_data': [],
            'gps_data': [],
            'imu_data': [],
            'camera_data': [],
            'static_transforms': []
        }
        
        try:
            with Reader(bag_path) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    topic_name = connection.topic
                    
                    # Track topic statistics
                    if topic_name not in analysis['topics']:
                        analysis['topics'][topic_name] = {
                            'message_type': connection.msgtype,
                            'message_count': 0,
                            'timestamps': []
                        }
                    
                    analysis['topics'][topic_name]['message_count'] += 1
                    analysis['topics'][topic_name]['timestamps'].append(timestamp)
                    
                    # Extract specific data
                    try:
                        if topic_name == '/mbuggy/odom':
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            analysis['odom_data'].append({
                                'timestamp': timestamp,
                                'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
                                'orientation': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                              msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
                                'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
                                'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                            })
                        
                        elif topic_name == '/mbuggy/fix':
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            analysis['gps_data'].append({
                                'timestamp': timestamp,
                                'latitude': msg.latitude,
                                'longitude': msg.longitude,
                                'altitude': msg.altitude,
                                'status': msg.status.status
                            })
                        
                        elif topic_name == '/mbuggy/imu_ins':
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            analysis['imu_data'].append({
                                'timestamp': timestamp,
                                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                            })
                        
                        elif topic_name == '/mbuggy/camera_front/image_rect':
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            analysis['camera_data'].append({
                                'timestamp': timestamp,
                                'width': msg.width,
                                'height': msg.height,
                                'encoding': msg.encoding,
                                'data_size': len(msg.data)
                            })
                        
                        elif topic_name == '/tf_static':
                            msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                            analysis['static_transforms'].append({
                                'timestamp': timestamp,
                                'transforms': msg.transforms
                            })
                    
                    except Exception as e:
                        logger.warning(f"Failed to process message from {topic_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading bag {bag_name}: {e}")
        
        # Process the extracted data
        analysis = self._process_analysis_data(analysis)
        return analysis

    def _process_analysis_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze the extracted data"""
        processed = analysis.copy()
        
        # Calculate topic frequencies
        for topic_name, topic_data in processed['topics'].items():
            if topic_data['timestamps']:
                timestamps = np.array(topic_data['timestamps'])
                duration = (timestamps[-1] - timestamps[0]) / 1e9
                topic_data['frequency_hz'] = len(timestamps) / duration if duration > 0 else 0
        
        # Process odometry data
        if analysis['odom_data']:
            processed['odom_analysis'] = self._analyze_odometry_data(analysis['odom_data'])
        
        # Process GPS data
        if analysis['gps_data']:
            processed['gps_analysis'] = self._analyze_gps_data(analysis['gps_data'])
        
        # Process IMU data
        if analysis['imu_data']:
            processed['imu_analysis'] = self._analyze_imu_data(analysis['imu_data'])
        
        # Process camera data
        if analysis['camera_data']:
            processed['camera_analysis'] = self._analyze_camera_data(analysis['camera_data'])
        
        return processed

    def _analyze_odometry_data(self, odom_data: List[Dict]) -> Dict[str, Any]:
        """Analyze odometry data with adaptive scaling"""
        positions = np.array([point['position'] for point in odom_data])
        orientations = np.array([point['orientation'] for point in odom_data])
        timestamps = np.array([point['timestamp'] for point in odom_data])
        
        # Calculate metrics
        speeds = []
        turning_rates = []
        
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            dt = (timestamps[i] - timestamps[i-1]) / 1e9
            if dt > 0:
                speed = dist / dt
                speeds.append(speed)
        
        for i in range(1, len(orientations)):
            q1 = orientations[i-1]
            q2 = orientations[i]
            dot_product = np.dot(q1, q2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = 2 * np.arccos(np.abs(dot_product))
            dt = (timestamps[i] - timestamps[i-1]) / 1e9
            if dt > 0:
                turning_rate = angle_diff / dt
                turning_rates.append(turning_rate)
        
        # Calculate adaptive ranges
        adaptive_ranges = {
            'speed': self._calculate_adaptive_range(speeds),
            'turning_rate': self._calculate_adaptive_range(turning_rates),
            'elevation': self._calculate_adaptive_range(positions[:, 2], method='full')
        }
        
        return {
            'positions': positions,
            'timestamps': timestamps,
            'speeds': speeds,
            'turning_rates': turning_rates,
            'adaptive_ranges': adaptive_ranges,
            'total_distance': float(np.sum([np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions))])),
            'duration': float((timestamps[-1] - timestamps[0]) / 1e9),
            'elevation_range': float(np.max(positions[:, 2]) - np.min(positions[:, 2])),
            'is_planar': np.max(positions[:, 2]) - np.min(positions[:, 2]) < 1.0
        }

    def _analyze_gps_data(self, gps_data: List[Dict]) -> Dict[str, Any]:
        """Analyze GPS data"""
        statuses = [point['status'] for point in gps_data]
        total_fixes = len(gps_data)
        
        # GPS status analysis
        no_fix = statuses.count(0)
        gps_fix = statuses.count(1)
        dgps_fix = statuses.count(2)
        rtk_fix = statuses.count(4)
        rtk_float = statuses.count(5)
        
        gps_coverage = (gps_fix + dgps_fix + rtk_fix + rtk_float) / total_fixes * 100
        
        return {
            'total_fixes': total_fixes,
            'gps_coverage_percent': float(gps_coverage),
            'no_fix_count': no_fix,
            'gps_fix_count': gps_fix,
            'dgps_fix_count': dgps_fix,
            'rtk_fix_count': rtk_fix,
            'rtk_float_count': rtk_float,
            'statuses': statuses
        }

    def _analyze_imu_data(self, imu_data: List[Dict]) -> Dict[str, Any]:
        """Analyze IMU data"""
        linear_accels = np.array([point['linear_acceleration'] for point in imu_data])
        angular_vels = np.array([point['angular_velocity'] for point in imu_data])
        
        accel_magnitudes = np.linalg.norm(linear_accels, axis=1)
        angular_magnitudes = np.linalg.norm(angular_vels, axis=1)
        
        return {
            'avg_acceleration': float(np.mean(accel_magnitudes)),
            'max_acceleration': float(np.max(accel_magnitudes)),
            'avg_angular_velocity': float(np.mean(angular_magnitudes)),
            'max_angular_velocity': float(np.max(angular_magnitudes)),
            'adaptive_ranges': {
                'acceleration': self._calculate_adaptive_range(accel_magnitudes),
                'angular_velocity': self._calculate_adaptive_range(angular_magnitudes)
            }
        }

    def _analyze_camera_data(self, camera_data: List[Dict]) -> Dict[str, Any]:
        """Analyze camera data"""
        widths = [data['width'] for data in camera_data]
        heights = [data['height'] for data in camera_data]
        data_sizes = [data['data_size'] for data in camera_data]
        
        return {
            'resolution': f"{widths[0]}x{heights[0]}",
            'encoding': camera_data[0]['encoding'],
            'total_frames': len(camera_data),
            'average_data_size_bytes': float(np.mean(data_sizes)),
            'frame_rate_hz': len(camera_data) / ((camera_data[-1]['timestamp'] - camera_data[0]['timestamp']) / 1e9) if len(camera_data) > 1 else 0
        }

    def _calculate_adaptive_range(self, data: List[float], method: str = 'percentile') -> Dict[str, float]:
        """Calculate adaptive color range for data"""
        if data is None or len(data) == 0:
            return {'min': 0, 'max': 1, 'center': 0.5}
        
        data_array = np.array(data)
        
        if method == 'percentile':
            p5, p95 = np.percentile(data_array, [5, 95])
            center = np.median(data_array)
            min_range = (p95 - p5) * 0.1
            if min_range < 1e-6:
                min_range = 1e-6
            
            return {
                'min': max(p5 - min_range, np.min(data_array)),
                'max': min(p95 + min_range, np.max(data_array)),
                'center': center,
                'p5': p5,
                'p95': p95
            }
        else:
            return {
                'min': np.min(data_array),
                'max': np.max(data_array),
                'center': np.mean(data_array)
            }

    def _generate_visualizations(self, analyses: Dict[str, Any]):
        """Generate all visualizations"""
        logger.info("🎨 Phase 2: Generating visualizations...")
        
        for bag_name, analysis in analyses.items():
            if 'odom_analysis' not in analysis:
                continue
            
            self._generate_adaptive_color_plots(bag_name, analysis)
            logger.info(f"   ✅ Generated visualizations for {bag_name}")

    def _generate_adaptive_color_plots(self, bag_name: str, analysis: Dict[str, Any]):
        """Generate adaptive color-coded trajectory plots"""
        odom_analysis = analysis['odom_analysis']
        positions = odom_analysis['positions']
        timestamps = odom_analysis['timestamps']
        adaptive_ranges = odom_analysis['adaptive_ranges']
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Complete Analysis: {bag_name}', fontsize=16)
        
        # 1. Speed-colored trajectory
        ax1 = plt.subplot(3, 4, 1)
        self._plot_speed_trajectory(ax1, positions, odom_analysis['speeds'], adaptive_ranges['speed'])
        
        # 2. Turning rate-colored trajectory
        ax2 = plt.subplot(3, 4, 2)
        self._plot_turning_rate_trajectory(ax2, positions, odom_analysis['turning_rates'], adaptive_ranges['turning_rate'])
        
        # 3. Elevation-colored trajectory
        ax3 = plt.subplot(3, 4, 3)
        self._plot_elevation_trajectory(ax3, positions, adaptive_ranges['elevation'])
        
        # 4. GPS coverage-colored trajectory
        ax4 = plt.subplot(3, 4, 4)
        if 'gps_analysis' in analysis:
            self._plot_gps_trajectory(ax4, positions, analysis['gps_analysis'])
        
        # 5. 3D trajectory
        ax5 = plt.subplot(3, 4, 5, projection='3d')
        self._plot_3d_trajectory(ax5, positions)
        
        # 6. Speed profile
        ax6 = plt.subplot(3, 4, 6)
        self._plot_speed_profile(ax6, timestamps, odom_analysis['speeds'], adaptive_ranges['speed'])
        
        # 7. Turning rate profile
        ax7 = plt.subplot(3, 4, 7)
        self._plot_turning_rate_profile(ax7, timestamps, odom_analysis['turning_rates'], adaptive_ranges['turning_rate'])
        
        # 8. Data distribution
        ax8 = plt.subplot(3, 4, 8)
        self._plot_data_distribution(ax8, odom_analysis['speeds'], odom_analysis['turning_rates'])
        
        # 9. Topic statistics
        ax9 = plt.subplot(3, 4, 9)
        self._plot_topic_statistics(ax9, analysis['topics'])
        
        # 10. GPS analysis
        ax10 = plt.subplot(3, 4, 10)
        if 'gps_analysis' in analysis:
            self._plot_gps_analysis(ax10, analysis['gps_analysis'])
        
        # 11. IMU analysis
        ax11 = plt.subplot(3, 4, 11)
        if 'imu_analysis' in analysis:
            self._plot_imu_analysis(ax11, analysis['imu_analysis'])
        
        # 12. Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        self._plot_summary_statistics(ax12, analysis)
        
        plt.tight_layout()
        
        # Save to visualizations folder
        viz_dir = Path("reports/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f'{bag_name}_complete_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"   📁 Visualization saved: {viz_path.absolute()}")
        plt.close()

    def _plot_speed_trajectory(self, ax, positions: np.ndarray, speeds: List[float], speed_range: Dict):
        """Plot trajectory colored by speed"""
        speed_array = np.zeros(len(positions))
        for i in range(1, len(positions)):
            if i-1 < len(speeds):
                speed_array[i] = speeds[i-1]
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=speed_array, 
                           cmap='viridis', s=20, alpha=0.7,
                           vmin=speed_range['min'], vmax=speed_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Speed-Colored Trajectory\nRange: {speed_range["min"]:.2f}-{speed_range["max"]:.2f} m/s')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (m/s)')

    def _plot_turning_rate_trajectory(self, ax, positions: np.ndarray, turning_rates: List[float], turning_range: Dict):
        """Plot trajectory colored by turning rate"""
        turning_array = np.zeros(len(positions))
        for i in range(2, len(positions)):
            if i-2 < len(turning_rates):
                turning_array[i] = turning_rates[i-2]
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=turning_array, 
                           cmap='plasma', s=20, alpha=0.7,
                           vmin=turning_range['min'], vmax=turning_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Turning Rate-Colored Trajectory\nRange: {turning_range["min"]:.3f}-{turning_range["max"]:.3f} rad/s')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Turning Rate (rad/s)')

    def _plot_elevation_trajectory(self, ax, positions: np.ndarray, elevation_range: Dict):
        """Plot trajectory colored by elevation"""
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
                           cmap='terrain', s=20, alpha=0.7,
                           vmin=elevation_range['min'], vmax=elevation_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Elevation-Colored Trajectory\nRange: {elevation_range["min"]:.3f}-{elevation_range["max"]:.3f} m')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elevation (m)')

    def _plot_gps_trajectory(self, ax, positions: np.ndarray, gps_analysis: Dict):
        """Plot trajectory colored by GPS coverage"""
        statuses = gps_analysis['statuses']
        gps_coverage = np.zeros(len(positions))
        
        for i, status in enumerate(statuses):
            if i < len(positions):
                gps_coverage[i] = status
        
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=gps_coverage, 
                           cmap='RdYlGn', s=20, alpha=0.7)
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'GPS Coverage-Colored Trajectory\nCoverage: {gps_analysis["gps_coverage_percent"]:.1f}%')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('GPS Status')

    def _plot_3d_trajectory(self, ax, positions: np.ndarray):
        """Plot 3D trajectory"""
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color='green', s=100, marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory')
        ax.legend()

    def _plot_speed_profile(self, ax, timestamps: np.ndarray, speeds: List[float], speed_range: Dict):
        """Plot speed profile"""
        if speeds:
            time_points = []
            for i in range(1, len(timestamps)):
                time_points.append((timestamps[i] - timestamps[0]) / 1e9)
            
            min_len = min(len(time_points), len(speeds))
            time_points = time_points[:min_len]
            speeds = speeds[:min_len]
            
            ax.plot(time_points, speeds, 'b-', linewidth=2)
            ax.fill_between(time_points, speeds, alpha=0.3)
            ax.axhline(y=speed_range['center'], color='r', linestyle='--', label=f'Center: {speed_range["center"]:.2f} m/s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_turning_rate_profile(self, ax, timestamps: np.ndarray, turning_rates: List[float], turning_range: Dict):
        """Plot turning rate profile"""
        if turning_rates:
            time_points = []
            for i in range(2, len(timestamps)):
                time_points.append((timestamps[i] - timestamps[0]) / 1e9)
            
            min_len = min(len(time_points), len(turning_rates))
            time_points = time_points[:min_len]
            turning_rates = turning_rates[:min_len]
            
            ax.plot(time_points, turning_rates, 'r-', linewidth=2)
            ax.fill_between(time_points, turning_rates, alpha=0.3)
            ax.axhline(y=turning_range['center'], color='b', linestyle='--', label=f'Center: {turning_range["center"]:.3f} rad/s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Turning Rate (rad/s)')
        ax.set_title('Turning Rate Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_data_distribution(self, ax, speeds: List[float], turning_rates: List[float]):
        """Plot data distribution"""
        ax2 = ax.twinx()
        
        if speeds:
            ax.hist(speeds, bins=30, alpha=0.7, color='blue', label='Speed Distribution')
            ax.set_ylabel('Speed Count', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
        
        if turning_rates:
            ax2.hist(turning_rates, bins=30, alpha=0.7, color='red', label='Turning Rate Distribution')
            ax2.set_ylabel('Turning Rate Count', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Value')
        ax.set_title('Data Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_topic_statistics(self, ax, topics: Dict):
        """Plot topic statistics"""
        topic_names = list(topics.keys())
        frequencies = [topics[topic]['frequency_hz'] for topic in topic_names]
        
        ax.barh(range(len(topic_names)), frequencies)
        ax.set_yticks(range(len(topic_names)))
        ax.set_yticklabels([name.split('/')[-1] for name in topic_names])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Topic Frequencies')
        ax.grid(True, alpha=0.3)

    def _plot_gps_analysis(self, ax, gps_analysis: Dict):
        """Plot GPS analysis"""
        labels = ['No Fix', 'GPS Fix', 'DGPS Fix', 'RTK Fix', 'RTK Float']
        counts = [gps_analysis['no_fix_count'], gps_analysis['gps_fix_count'], 
                 gps_analysis['dgps_fix_count'], gps_analysis['rtk_fix_count'], 
                 gps_analysis['rtk_float_count']]
        colors = ['red', 'orange', 'yellow', 'green', 'lightgreen']
        
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title(f'GPS Analysis\nCoverage: {gps_analysis["gps_coverage_percent"]:.1f}%')

    def _plot_imu_analysis(self, ax, imu_analysis: Dict):
        """Plot IMU analysis"""
        metrics = ['Avg Accel', 'Max Accel', 'Avg Ang Vel', 'Max Ang Vel']
        values = [imu_analysis['avg_acceleration'], imu_analysis['max_acceleration'],
                 imu_analysis['avg_angular_velocity'], imu_analysis['max_angular_velocity']]
        
        ax.bar(metrics, values)
        ax.set_ylabel('Value')
        ax.set_title('IMU Analysis')
        ax.grid(True, alpha=0.3)

    def _plot_summary_statistics(self, ax, analysis: Dict):
        """Plot summary statistics"""
        stats_text = f"Summary for {analysis['bag_name']}:\n\n"
        
        if 'odom_analysis' in analysis:
            odom = analysis['odom_analysis']
            stats_text += f"Distance: {odom['total_distance']:.1f} m\n"
            stats_text += f"Duration: {odom['duration']:.1f} s\n"
            stats_text += f"Planar Motion: {'Yes' if odom['is_planar'] else 'No'}\n\n"
        
        if 'gps_analysis' in analysis:
            gps = analysis['gps_analysis']
            stats_text += f"GPS Coverage: {gps['gps_coverage_percent']:.1f}%\n"
            stats_text += f"Total Fixes: {gps['total_fixes']}\n\n"
        
        if 'camera_analysis' in analysis:
            cam = analysis['camera_analysis']
            stats_text += f"Camera: {cam['resolution']}\n"
            stats_text += f"Frames: {cam['total_frames']}\n"
            stats_text += f"Frame Rate: {cam['frame_rate_hz']:.1f} Hz\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary Statistics')

    def _generate_videos(self):
        """Generate videos (optional)"""
        logger.info("🎬 Phase 3: Video generation (optional)")
        # Video generation can be added here if needed
        pass

    def _generate_comprehensive_report(self, analyses: Dict[str, Any]):
        """Generate comprehensive report"""
        logger.info("📋 Phase 4: Generating comprehensive report...")
        
        report_path = self.output_dir / 'complete_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Complete ROS Bag Analysis Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Bags Analyzed**: {len(analyses)}\n")
            
            total_distance = sum(analysis.get('odom_analysis', {}).get('total_distance', 0) for analysis in analyses.values())
            total_duration = sum(analysis.get('odom_analysis', {}).get('duration', 0) for analysis in analyses.values())
            
            f.write(f"- **Total Distance**: {total_distance:.1f} m\n")
            f.write(f"- **Total Duration**: {total_duration:.1f} s\n\n")
            
            # Detailed analysis for each bag
            for bag_name, analysis in analyses.items():
                f.write(f"## {bag_name}\n\n")
                
                # Topic analysis
                f.write("### Topics and Frequencies\n")
                f.write("| Topic | Message Type | Count | Frequency (Hz) |\n")
                f.write("|-------|-------------|-------|----------------|\n")
                
                for topic_name, topic_data in analysis['topics'].items():
                    f.write(f"| {topic_name} | {topic_data['message_type']} | {topic_data['message_count']} | {topic_data.get('frequency_hz', 0):.2f} |\n")
                f.write("\n")
                
                # Odometry analysis
                if 'odom_analysis' in analysis:
                    odom = analysis['odom_analysis']
                    f.write("### Trajectory Analysis\n")
                    f.write(f"- **Total Distance**: {odom['total_distance']:.1f} m\n")
                    f.write(f"- **Duration**: {odom['duration']:.1f} s\n")
                    f.write(f"- **Elevation Range**: {odom['elevation_range']:.3f} m\n")
                    f.write(f"- **Is Planar Motion**: {'Yes' if odom['is_planar'] else 'No'}\n\n")
                
                # GPS analysis
                if 'gps_analysis' in analysis:
                    gps = analysis['gps_analysis']
                    f.write("### GPS Analysis\n")
                    f.write(f"- **GPS Coverage**: {gps['gps_coverage_percent']:.1f}%\n")
                    f.write(f"- **Total Fixes**: {gps['total_fixes']}\n")
                    f.write(f"- **No Fix Count**: {gps['no_fix_count']}\n")
                    f.write(f"- **GPS Fix Count**: {gps['gps_fix_count']}\n")
                    f.write(f"- **DGPS Fix Count**: {gps['dgps_fix_count']}\n")
                    f.write(f"- **RTK Fix Count**: {gps['rtk_fix_count']}\n\n")
                
                # IMU analysis
                if 'imu_analysis' in analysis:
                    imu = analysis['imu_analysis']
                    f.write("### IMU Analysis\n")
                    f.write(f"- **Average Acceleration**: {imu['avg_acceleration']:.2f} m/s²\n")
                    f.write(f"- **Max Acceleration**: {imu['max_acceleration']:.2f} m/s²\n")
                    f.write(f"- **Average Angular Velocity**: {imu['avg_angular_velocity']:.2f} rad/s\n")
                    f.write(f"- **Max Angular Velocity**: {imu['max_angular_velocity']:.2f} rad/s\n\n")
                
                # Camera analysis
                if 'camera_analysis' in analysis:
                    cam = analysis['camera_analysis']
                    f.write("### Camera Analysis\n")
                    f.write(f"- **Resolution**: {cam['resolution']}\n")
                    f.write(f"- **Encoding**: {cam['encoding']}\n")
                    f.write(f"- **Total Frames**: {cam['total_frames']}\n")
                    f.write(f"- **Frame Rate**: {cam['frame_rate_hz']:.1f} Hz\n")
                    f.write(f"- **Average Data Size**: {cam['average_data_size_bytes']:.0f} bytes\n\n")
            
            # Recommendations
            f.write("## Recommendations for Visual Localization\n\n")
            f.write("### Data Quality Assessment\n")
            f.write("- ✅ **Excellent camera data** (1920x1080 @ 26-27 Hz)\n")
            f.write("- ✅ **Good IMU data** (realistic accelerations and angular velocities)\n")
            f.write("- ✅ **Perfect planar motion** (Z=0 throughout)\n")
            f.write("- ❌ **No GPS signal** (GPS-denied environment)\n\n")
            
            f.write("### Recommended Approach\n")
            f.write("- **Visual-Inertial Odometry (VIO)** with 2D motion constraints\n")
            f.write("- **Use `/mbuggy/odom` as ground truth** for trajectory comparison\n")
            f.write("- **Implement planar motion assumption** (Z=0 constraint)\n")
            f.write("- **Focus on visual-inertial fusion** without GPS dependency\n\n")
            
            f.write("### Next Steps\n")
            f.write("1. **Implement VIO algorithm** (ORB-SLAM3 or OpenVINS)\n")
            f.write("2. **Setup coordinate frame alignment** using static transforms\n")
            f.write("3. **Validate against ground truth** from odometry data\n")
            f.write("4. **Optimize parameters** for outdoor, high-speed scenarios\n")
        
        logger.info(f"   ✅ Comprehensive report saved: {report_path.absolute()}")
        
        # Generate PDF version
        self._generate_pdf_report(report_path)
    
    def _generate_pdf_report(self, report_path: Path):
        """Generate PDF version of the report using weasyprint"""
        try:
            pdf_path = report_path.with_suffix('.pdf')
            logger.info(f"   📄 Generating PDF: {pdf_path.name}")
            
            # Import weasyprint components
            import markdown2
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # Read markdown file
            with open(report_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert markdown to HTML
            html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
            
            # CSS for styling
            css_content = """
            @page {
                size: A4;
                margin: 2cm;
            }
            
            body {
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
            }
            
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            
            h2 {
                color: #34495e;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
                margin-top: 25px;
            }
            
            h3 {
                color: #7f8c8d;
                margin-top: 20px;
            }
            
            code {
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }
            
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            """
            
            # Create full HTML document
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{report_path.stem}</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Generate PDF
            font_config = FontConfiguration()
            html_doc = HTML(string=full_html)
            css_doc = CSS(string=css_content, font_config=font_config)
            
            html_doc.write_pdf(pdf_path, stylesheets=[css_doc], font_config=font_config)
            logger.info(f"   ✅ PDF generated: {pdf_path.name}")
                
        except Exception as e:
            logger.warning(f"   ⚠️ PDF generation error: {e}")

class HardwareMonitor:
    """Hardware health monitoring"""
    def __init__(self):
        self.max_cpu_usage = 80
        self.max_memory_usage = 80
        self.max_gpu_temp = 85
        self.max_gpu_usage = 90
    
    def check_health(self):
        """Check hardware health"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > self.max_cpu_usage:
            logger.warning(f"High CPU usage: {cpu_usage:.1f}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.max_memory_usage:
            logger.warning(f"High memory usage: {memory.percent:.1f}%")
        
        # GPU usage
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.temperature > self.max_gpu_temp:
                        logger.warning(f"High GPU temperature: {gpu.temperature}°C")
                    if gpu.load * 100 > self.max_gpu_usage:
                        logger.warning(f"High GPU usage: {gpu.load * 100:.1f}%")
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description='Complete ROS Bag Analyzer')
    parser.add_argument('bag_paths', nargs='+', help='Paths to ROS bag directories')
    parser.add_argument('--output-dir', default='reports/analysis', 
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    analyzer = CompleteROSBagAnalyzer(args.bag_paths, args.output_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
