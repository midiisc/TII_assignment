#!/usr/bin/env python3
"""
Data synchronization validator for 2D VIO pipeline
Validates camera and IMU data synchronization from ROS bags
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from nav_msgs.msg import Odometry
import time
from collections import deque

class DataValidator(Node):
    def __init__(self):
        super().__init__('data_validator')
        
        # Data storage for synchronization analysis
        self.camera_timestamps = deque(maxlen=100)
        self.imu_timestamps = deque(maxlen=100)
        self.odom_timestamps = deque(maxlen=100)
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, 
            '/mbuggy/camera_front/image_rect', 
            self.camera_callback, 
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/mbuggy/camera_front/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu, 
            '/mbuggy/imu_ins', 
            self.imu_callback, 
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/mbuggy/odom', 
            self.odom_callback, 
            10
        )
        
        # Statistics tracking
        self.camera_count = 0
        self.imu_count = 0
        self.odom_count = 0
        self.camera_info_received = False
        
        # Timer for periodic reporting
        self.timer = self.create_timer(5.0, self.report_statistics)
        
        self.get_logger().info("🔍 Data Validator started - monitoring synchronization...")
        
    def camera_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.camera_timestamps.append(timestamp)
        self.camera_count += 1
        
        if self.camera_count <= 5:  # Log first few messages
            self.get_logger().info(
                f'📷 Camera [{self.camera_count}]: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} '
                f'| Frame: {msg.header.frame_id} | Size: {msg.width}x{msg.height}'
            )
    
    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.get_logger().info(f'📷 Camera Info - Frame: {msg.header.frame_id}')
            self.get_logger().info(f'   Resolution: {msg.width}x{msg.height}')
            self.get_logger().info(f'   Intrinsics: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}')
            self.get_logger().info(f'   Distortion: {[f"{d:.6f}" for d in msg.d[:4]]}')
            self.camera_info_received = True
    
    def imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_timestamps.append(timestamp)
        self.imu_count += 1
        
        if self.imu_count <= 5:  # Log first few messages
            self.get_logger().info(
                f'🧭 IMU [{self.imu_count}]: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} '
                f'| Frame: {msg.header.frame_id}'
            )
            self.get_logger().info(
                f'   Accel: [{msg.linear_acceleration.x:.3f}, {msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f}]'
            )
            self.get_logger().info(
                f'   Gyro:  [{msg.angular_velocity.x:.3f}, {msg.angular_velocity.y:.3f}, {msg.angular_velocity.z:.3f}]'
            )
    
    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.odom_timestamps.append(timestamp)
        self.odom_count += 1
        
        if self.odom_count <= 5:  # Log first few messages
            self.get_logger().info(
                f'🗺️  Odom [{self.odom_count}]: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} '
                f'| Frame: {msg.header.frame_id}'
            )
            self.get_logger().info(
                f'   Position: [{msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}, {msg.pose.pose.position.z:.3f}]'
            )
    
    def report_statistics(self):
        """Report data rates and synchronization statistics"""
        if not any([self.camera_timestamps, self.imu_timestamps, self.odom_timestamps]):
            self.get_logger().warn("⚠️  No data received yet...")
            return
            
        # Calculate frequencies
        camera_freq = self.calculate_frequency(self.camera_timestamps)
        imu_freq = self.calculate_frequency(self.imu_timestamps)
        odom_freq = self.calculate_frequency(self.odom_timestamps)
        
        self.get_logger().info("📊 === DATA STATISTICS ===")
        self.get_logger().info(f"📷 Camera: {self.camera_count} msgs @ {camera_freq:.1f} Hz")
        self.get_logger().info(f"🧭 IMU: {self.imu_count} msgs @ {imu_freq:.1f} Hz")
        self.get_logger().info(f"🗺️  Odometry: {self.odom_count} msgs @ {odom_freq:.1f} Hz")
        
        # Check synchronization
        if len(self.camera_timestamps) > 1 and len(self.imu_timestamps) > 1:
            sync_diff = abs(self.camera_timestamps[-1] - self.imu_timestamps[-1])
            if sync_diff < 0.1:  # Within 100ms
                self.get_logger().info(f"✅ Synchronization: GOOD ({sync_diff*1000:.1f}ms diff)")
            else:
                self.get_logger().warn(f"⚠️  Synchronization: POOR ({sync_diff*1000:.1f}ms diff)")
    
    def calculate_frequency(self, timestamps):
        """Calculate frequency from timestamp queue"""
        if len(timestamps) < 2:
            return 0.0
        
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return 0.0
            
        return (len(timestamps) - 1) / time_span

def main(args=None):
    rclpy.init(args=args)
    
    validator = DataValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.get_logger().info("🏁 Data validation complete")
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
