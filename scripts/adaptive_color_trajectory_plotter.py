#!/usr/bin/env python3
"""
Adaptive Color-Coded Trajectory Plotter
- Uses adaptive color scaling to make data variations more discernable
- Focuses color range on actual data variation rather than full range
- Provides better visualization of subtle changes in trajectory metrics
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveColorTrajectoryPlotter:
    def __init__(self, bag_paths: List[str], output_dir: str = "reports/visualizations"):
        self.bag_paths = bag_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.typestore = get_typestore(Stores.ROS2_JAZZY)
        
        logger.info(f"🎨 Adaptive Color Trajectory Plotter initialized")
        logger.info(f"   📁 Bags: {len(bag_paths)} bags")
        logger.info(f"   📁 Output: {output_dir}")

    def analyze_and_plot(self):
        """Analyze data and generate adaptive color-coded plots"""
        all_analyses = {}
        
        for bag_path in self.bag_paths:
            bag_name = Path(bag_path).name
            logger.info(f"📊 Analyzing {bag_name} for adaptive color plotting...")
            
            analysis = self._extract_trajectory_data(bag_path, bag_name)
            analysis = self._calculate_adaptive_metrics(analysis)
            all_analyses[bag_name] = analysis
            
            logger.info(f"   ✅ Analysis completed for {bag_name}")
        
        # Generate adaptive color plots
        self._generate_adaptive_color_plots(all_analyses)
        
        return all_analyses

    def _extract_trajectory_data(self, bag_path: str, bag_name: str) -> Dict[str, Any]:
        """Extract trajectory data from bag"""
        analysis = {
            'bag_name': bag_name,
            'odom_data': [],
            'gps_data': [],
            'imu_data': []
        }
        
        try:
            with Reader(bag_path) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    topic_name = connection.topic
                    
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
                    
                    except Exception as e:
                        logger.warning(f"Failed to process message from {topic_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading bag {bag_name}: {e}")
        
        return analysis

    def _calculate_adaptive_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics with adaptive scaling information"""
        processed = analysis.copy()
        
        if analysis['odom_data']:
            processed['adaptive_metrics'] = self._calculate_odom_adaptive_metrics(analysis['odom_data'])
        
        if analysis['gps_data']:
            processed['gps_adaptive_metrics'] = self._calculate_gps_adaptive_metrics(analysis['gps_data'])
        
        if analysis['imu_data']:
            processed['imu_adaptive_metrics'] = self._calculate_imu_adaptive_metrics(analysis['imu_data'])
        
        return processed

    def _calculate_odom_adaptive_metrics(self, odom_data: List[Dict]) -> Dict[str, Any]:
        """Calculate odometry metrics with adaptive scaling"""
        positions = np.array([point['position'] for point in odom_data])
        orientations = np.array([point['orientation'] for point in odom_data])
        timestamps = np.array([point['timestamp'] for point in odom_data])
        
        # Calculate speeds
        speeds = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            dt = (timestamps[i] - timestamps[i-1]) / 1e9
            if dt > 0:
                speed = dist / dt
                speeds.append(speed)
        
        # Calculate turning rates
        turning_rates = []
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
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(speeds)):
            dt = (timestamps[i+1] - timestamps[i]) / 1e9
            if dt > 0:
                accel = (speeds[i] - speeds[i-1]) / dt
                accelerations.append(accel)
        
        # Calculate jerks
        jerks = []
        for i in range(1, len(accelerations)):
            dt = (timestamps[i+2] - timestamps[i+1]) / 1e9
            if dt > 0:
                jerk = (accelerations[i] - accelerations[i-1]) / dt
                jerks.append(jerk)
        
        # Calculate adaptive color ranges
        metrics = {
            'positions': positions,
            'timestamps': timestamps,
            'speeds': speeds,
            'turning_rates': turning_rates,
            'accelerations': accelerations,
            'jerks': jerks,
            'adaptive_ranges': {
                'speed': self._calculate_adaptive_range(speeds, method='percentile'),
                'turning_rate': self._calculate_adaptive_range(turning_rates, method='percentile'),
                'acceleration': self._calculate_adaptive_range(accelerations, method='percentile'),
                'jerk': self._calculate_adaptive_range(jerks, method='percentile'),
                'elevation': self._calculate_adaptive_range(positions[:, 2], method='full')
            }
        }
        
        return metrics

    def _calculate_gps_adaptive_metrics(self, gps_data: List[Dict]) -> Dict[str, Any]:
        """Calculate GPS metrics with adaptive scaling"""
        statuses = [point['status'] for point in gps_data]
        altitudes = [point['altitude'] for point in gps_data if point['altitude'] != 0]
        
        metrics = {
            'statuses': statuses,
            'altitudes': altitudes,
            'adaptive_ranges': {
                'gps_status': self._calculate_adaptive_range(statuses, method='discrete'),
                'altitude': self._calculate_adaptive_range(altitudes, method='percentile') if altitudes else None
            }
        }
        
        return metrics

    def _calculate_imu_adaptive_metrics(self, imu_data: List[Dict]) -> Dict[str, Any]:
        """Calculate IMU metrics with adaptive scaling"""
        linear_accels = np.array([point['linear_acceleration'] for point in imu_data])
        angular_vels = np.array([point['angular_velocity'] for point in imu_data])
        
        accel_magnitudes = np.linalg.norm(linear_accels, axis=1)
        angular_magnitudes = np.linalg.norm(angular_vels, axis=1)
        
        # Calculate jerks
        jerks = []
        for i in range(1, len(linear_accels)):
            jerk = np.linalg.norm(linear_accels[i] - linear_accels[i-1])
            jerks.append(jerk)
        
        metrics = {
            'accel_magnitudes': accel_magnitudes,
            'angular_magnitudes': angular_magnitudes,
            'jerks': jerks,
            'adaptive_ranges': {
                'acceleration': self._calculate_adaptive_range(accel_magnitudes, method='percentile'),
                'angular_velocity': self._calculate_adaptive_range(angular_magnitudes, method='percentile'),
                'jerk': self._calculate_adaptive_range(jerks, method='percentile')
            }
        }
        
        return metrics

    def _calculate_adaptive_range(self, data: List[float], method: str = 'percentile') -> Dict[str, float]:
        """Calculate adaptive color range for data"""
        if data is None or len(data) == 0:
            return {'min': 0, 'max': 1, 'center': 0.5}
        
        data_array = np.array(data)
        
        if method == 'percentile':
            # Use percentiles to focus on main data distribution
            p5, p95 = np.percentile(data_array, [5, 95])
            center = np.median(data_array)
            
            # Ensure minimum range for visualization
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
        
        elif method == 'full':
            # Use full range
            return {
                'min': np.min(data_array),
                'max': np.max(data_array),
                'center': np.mean(data_array)
            }
        
        elif method == 'discrete':
            # For discrete data (like GPS status)
            unique_values = np.unique(data_array)
            return {
                'min': np.min(unique_values),
                'max': np.max(unique_values),
                'center': np.mean(unique_values),
                'unique_values': unique_values.tolist()
            }
        
        else:
            # Default to full range
            return {
                'min': np.min(data_array),
                'max': np.max(data_array),
                'center': np.mean(data_array)
            }

    def _generate_adaptive_color_plots(self, analyses: Dict[str, Any]):
        """Generate adaptive color-coded trajectory plots"""
        logger.info("🎨 Generating adaptive color-coded trajectory plots...")
        
        for bag_name, analysis in analyses.items():
            if 'adaptive_metrics' not in analysis:
                continue
            
            # Create comprehensive adaptive plot
            fig = plt.figure(figsize=(24, 18))
            fig.suptitle(f'Adaptive Color-Coded Trajectory Analysis: {bag_name}', fontsize=16)
            
            # 1. Speed-colored trajectory (adaptive)
            ax1 = plt.subplot(3, 4, 1)
            self._plot_adaptive_speed_trajectory(ax1, analysis['adaptive_metrics'])
            
            # 2. Turning rate-colored trajectory (adaptive)
            ax2 = plt.subplot(3, 4, 2)
            self._plot_adaptive_turning_rate_trajectory(ax2, analysis['adaptive_metrics'])
            
            # 3. Elevation-colored trajectory (adaptive)
            ax3 = plt.subplot(3, 4, 3)
            self._plot_adaptive_elevation_trajectory(ax3, analysis['adaptive_metrics'])
            
            # 4. GPS coverage-colored trajectory (adaptive)
            ax4 = plt.subplot(3, 4, 4)
            if 'gps_adaptive_metrics' in analysis:
                self._plot_adaptive_gps_trajectory(ax4, analysis['adaptive_metrics'], analysis['gps_adaptive_metrics'])
            
            # 5. Acceleration-colored trajectory (adaptive)
            ax5 = plt.subplot(3, 4, 5)
            if 'imu_adaptive_metrics' in analysis:
                self._plot_adaptive_acceleration_trajectory(ax5, analysis['adaptive_metrics'], analysis['imu_adaptive_metrics'])
            
            # 6. Jerk-colored trajectory (adaptive)
            ax6 = plt.subplot(3, 4, 6)
            if 'imu_adaptive_metrics' in analysis:
                self._plot_adaptive_jerk_trajectory(ax6, analysis['adaptive_metrics'], analysis['imu_adaptive_metrics'])
            
            # 7. 3D trajectory with adaptive elevation
            ax7 = plt.subplot(3, 4, 7, projection='3d')
            self._plot_adaptive_3d_trajectory(ax7, analysis['adaptive_metrics'])
            
            # 8. Speed profile with adaptive scaling
            ax8 = plt.subplot(3, 4, 8)
            self._plot_adaptive_speed_profile(ax8, analysis['adaptive_metrics'])
            
            # 9. Turning rate profile with adaptive scaling
            ax9 = plt.subplot(3, 4, 9)
            self._plot_adaptive_turning_rate_profile(ax9, analysis['adaptive_metrics'])
            
            # 10. Data distribution analysis
            ax10 = plt.subplot(3, 4, 10)
            self._plot_data_distribution(ax10, analysis['adaptive_metrics'])
            
            # 11. Color range information
            ax11 = plt.subplot(3, 4, 11)
            self._plot_color_range_info(ax11, analysis['adaptive_metrics'])
            
            # 12. Statistics summary
            ax12 = plt.subplot(3, 4, 12)
            self._plot_adaptive_statistics_summary(ax12, analysis)
            
        plt.tight_layout()
        output_path = self.output_dir / f'{bag_name}_adaptive_color_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"   📁 Adaptive color plot saved: {output_path.absolute()}")
        plt.close()
            
            logger.info(f"   ✅ Adaptive color plots saved for {bag_name}")

    def _plot_adaptive_speed_trajectory(self, ax, metrics: Dict):
        """Plot trajectory colored by speed with adaptive scaling"""
        positions = metrics['positions']
        speeds = metrics['speeds']
        speed_range = metrics['adaptive_ranges']['speed']
        
        # Create speed array for each position
        speed_array = np.zeros(len(positions))
        for i in range(1, len(positions)):
            if i-1 < len(speeds):
                speed_array[i] = speeds[i-1]
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=speed_array, 
                           cmap='viridis', s=20, alpha=0.7,
                           vmin=speed_range['min'], vmax=speed_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Speed-Colored Trajectory\nRange: {speed_range["min"]:.2f}-{speed_range["max"]:.2f} m/s')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (m/s)')

    def _plot_adaptive_turning_rate_trajectory(self, ax, metrics: Dict):
        """Plot trajectory colored by turning rate with adaptive scaling"""
        positions = metrics['positions']
        turning_rates = metrics['turning_rates']
        turning_range = metrics['adaptive_ranges']['turning_rate']
        
        # Create turning rate array for each position
        turning_array = np.zeros(len(positions))
        for i in range(2, len(positions)):
            if i-2 < len(turning_rates):
                turning_array[i] = turning_rates[i-2]
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=turning_array, 
                           cmap='plasma', s=20, alpha=0.7,
                           vmin=turning_range['min'], vmax=turning_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Turning Rate-Colored Trajectory\nRange: {turning_range["min"]:.3f}-{turning_range["max"]:.3f} rad/s')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Turning Rate (rad/s)')

    def _plot_adaptive_elevation_trajectory(self, ax, metrics: Dict):
        """Plot trajectory colored by elevation with adaptive scaling"""
        positions = metrics['positions']
        elevation_range = metrics['adaptive_ranges']['elevation']
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
                           cmap='terrain', s=20, alpha=0.7,
                           vmin=elevation_range['min'], vmax=elevation_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Elevation-Colored Trajectory\nRange: {elevation_range["min"]:.3f}-{elevation_range["max"]:.3f} m')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elevation (m)')

    def _plot_adaptive_gps_trajectory(self, ax, odom_metrics: Dict, gps_metrics: Dict):
        """Plot trajectory colored by GPS coverage with adaptive scaling"""
        positions = odom_metrics['positions']
        statuses = gps_metrics['statuses']
        gps_range = gps_metrics['adaptive_ranges']['gps_status']
        
        # Create GPS coverage array
        gps_coverage = np.zeros(len(positions))
        for i, status in enumerate(statuses):
            if i < len(positions):
                gps_coverage[i] = status
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=gps_coverage, 
                           cmap='RdYlGn', s=20, alpha=0.7,
                           vmin=gps_range['min'], vmax=gps_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'GPS Coverage-Colored Trajectory\nRange: {gps_range["min"]:.0f}-{gps_range["max"]:.0f}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('GPS Status')

    def _plot_adaptive_acceleration_trajectory(self, ax, odom_metrics: Dict, imu_metrics: Dict):
        """Plot trajectory colored by acceleration with adaptive scaling"""
        positions = odom_metrics['positions']
        accel_magnitudes = imu_metrics['accel_magnitudes']
        accel_range = imu_metrics['adaptive_ranges']['acceleration']
        
        # Create acceleration array for each position
        accel_array = np.zeros(len(positions))
        for i in range(len(positions)):
            if i < len(accel_magnitudes):
                accel_array[i] = accel_magnitudes[i]
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=accel_array, 
                           cmap='hot', s=20, alpha=0.7,
                           vmin=accel_range['min'], vmax=accel_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Acceleration-Colored Trajectory\nRange: {accel_range["min"]:.2f}-{accel_range["max"]:.2f} m/s²')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Acceleration (m/s²)')

    def _plot_adaptive_jerk_trajectory(self, ax, odom_metrics: Dict, imu_metrics: Dict):
        """Plot trajectory colored by jerk with adaptive scaling"""
        positions = odom_metrics['positions']
        jerks = imu_metrics['jerks']
        jerk_range = imu_metrics['adaptive_ranges']['jerk']
        
        # Create jerk array for each position
        jerk_array = np.zeros(len(positions))
        for i in range(2, len(positions)):
            if i-2 < len(jerks):
                jerk_array[i] = jerks[i-2]
        
        # Plot with adaptive color coding
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=jerk_array, 
                           cmap='coolwarm', s=20, alpha=0.7,
                           vmin=jerk_range['min'], vmax=jerk_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Jerk-Colored Trajectory\nRange: {jerk_range["min"]:.2f}-{jerk_range["max"]:.2f} m/s³')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar with adaptive range
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Jerk (m/s³)')

    def _plot_adaptive_3d_trajectory(self, ax, metrics: Dict):
        """Plot 3D trajectory with adaptive elevation coloring"""
        positions = metrics['positions']
        elevation_range = metrics['adaptive_ranges']['elevation']
        
        # Plot 3D trajectory with adaptive elevation coloring
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c=positions[:, 2], cmap='terrain', s=20, alpha=0.7,
                           vmin=elevation_range['min'], vmax=elevation_range['max'])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory (Adaptive Elevation)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Elevation (m)')

    def _plot_adaptive_speed_profile(self, ax, metrics: Dict):
        """Plot speed profile with adaptive scaling"""
        timestamps = metrics['timestamps']
        speeds = metrics['speeds']
        speed_range = metrics['adaptive_ranges']['speed']
        
        if speeds:
            time_points = []
            for i in range(1, len(timestamps)):
                time_points.append((timestamps[i] - timestamps[0]) / 1e9)
            
            # Ensure arrays have same length
            min_len = min(len(time_points), len(speeds))
            time_points = time_points[:min_len]
            speeds = speeds[:min_len]
            
            ax.plot(time_points, speeds, 'b-', linewidth=2)
            ax.fill_between(time_points, speeds, alpha=0.3)
            ax.axhline(y=speed_range['center'], color='r', linestyle='--', label=f'Center: {speed_range["center"]:.2f} m/s')
            ax.axhline(y=speed_range['p5'], color='orange', linestyle=':', label=f'P5: {speed_range["p5"]:.2f} m/s')
            ax.axhline(y=speed_range['p95'], color='orange', linestyle=':', label=f'P95: {speed_range["p95"]:.2f} m/s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed Profile (Adaptive Range)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_adaptive_turning_rate_profile(self, ax, metrics: Dict):
        """Plot turning rate profile with adaptive scaling"""
        timestamps = metrics['timestamps']
        turning_rates = metrics['turning_rates']
        turning_range = metrics['adaptive_ranges']['turning_rate']
        
        if turning_rates:
            time_points = []
            for i in range(2, len(timestamps)):
                time_points.append((timestamps[i] - timestamps[0]) / 1e9)
            
            # Ensure arrays have same length
            min_len = min(len(time_points), len(turning_rates))
            time_points = time_points[:min_len]
            turning_rates = turning_rates[:min_len]
            
            ax.plot(time_points, turning_rates, 'r-', linewidth=2)
            ax.fill_between(time_points, turning_rates, alpha=0.3)
            ax.axhline(y=turning_range['center'], color='b', linestyle='--', label=f'Center: {turning_range["center"]:.3f} rad/s')
            ax.axhline(y=turning_range['p5'], color='orange', linestyle=':', label=f'P5: {turning_range["p5"]:.3f} rad/s')
            ax.axhline(y=turning_range['p95'], color='orange', linestyle=':', label=f'P95: {turning_range["p95"]:.3f} rad/s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Turning Rate (rad/s)')
        ax.set_title('Turning Rate Profile (Adaptive Range)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_data_distribution(self, ax, metrics: Dict):
        """Plot data distribution analysis"""
        speeds = metrics['speeds']
        turning_rates = metrics['turning_rates']
        
        # Create subplot for speed distribution
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
        ax.set_title('Data Distribution Analysis')
        ax.grid(True, alpha=0.3)

    def _plot_color_range_info(self, ax, metrics: Dict):
        """Plot color range information"""
        ranges = metrics['adaptive_ranges']
        
        info_text = "Adaptive Color Ranges:\n\n"
        for metric_name, range_info in ranges.items():
            if metric_name != 'elevation':  # Skip elevation for brevity
                info_text += f"{metric_name.upper()}:\n"
                info_text += f"  Min: {range_info['min']:.3f}\n"
                info_text += f"  Max: {range_info['max']:.3f}\n"
                info_text += f"  Center: {range_info['center']:.3f}\n"
                if 'p5' in range_info:
                    info_text += f"  P5: {range_info['p5']:.3f}\n"
                    info_text += f"  P95: {range_info['p95']:.3f}\n"
                info_text += "\n"
        
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Color Range Information')

    def _plot_adaptive_statistics_summary(self, ax, analysis: Dict):
        """Plot adaptive statistics summary"""
        if 'adaptive_metrics' not in analysis:
            return
        
        metrics = analysis['adaptive_metrics']
        ranges = metrics['adaptive_ranges']
        
        stats_text = f"Adaptive Statistics:\n\n"
        stats_text += f"Speed Range: {ranges['speed']['min']:.2f}-{ranges['speed']['max']:.2f} m/s\n"
        stats_text += f"Turning Rate Range: {ranges['turning_rate']['min']:.3f}-{ranges['turning_rate']['max']:.3f} rad/s\n"
        stats_text += f"Elevation Range: {ranges['elevation']['min']:.3f}-{ranges['elevation']['max']:.3f} m\n\n"
        
        if 'imu_adaptive_metrics' in analysis:
            imu_ranges = analysis['imu_adaptive_metrics']['adaptive_ranges']
            stats_text += f"Acceleration Range: {imu_ranges['acceleration']['min']:.2f}-{imu_ranges['acceleration']['max']:.2f} m/s²\n"
            stats_text += f"Jerk Range: {imu_ranges['jerk']['min']:.2f}-{imu_ranges['jerk']['max']:.2f} m/s³\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Adaptive Statistics Summary')

def main():
    parser = argparse.ArgumentParser(description='Adaptive Color-Coded Trajectory Plotter')
    parser.add_argument('bag_paths', nargs='+', help='Paths to ROS bag directories')
    parser.add_argument('--output-dir', default='reports/visualizations', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    plotter = AdaptiveColorTrajectoryPlotter(args.bag_paths, args.output_dir)
    analyses = plotter.analyze_and_plot()

if __name__ == "__main__":
    main()
