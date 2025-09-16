#!/usr/bin/env python3
"""
Camera Calibration Extractor for ORB-SLAM3
Extracts camera intrinsic parameters from ROS bags and creates ORB-SLAM3 configuration files
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraCalibrationExtractor:
    def __init__(self, bag_path: str, output_dir: str = "config"):
        self.bag_path = bag_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ROS message types
        self.typestore = get_typestore(Stores.LATEST)
        
    def extract_calibration(self) -> Optional[Dict]:
        """Extract camera calibration from ROS bag"""
        logger.info(f"🔍 Extracting camera calibration from {self.bag_path}")
        
        calibration_data = None
        
        with Reader(self.bag_path) as reader:
            for connection in reader.connections:
                if 'camera_info' in connection.topic:
                    logger.info(f"📷 Found camera info topic: {connection.topic}")
                    
                    # Read first camera_info message
                    for connection, timestamp, rawdata in reader.messages(connections=[connection]):
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        
                        calibration_data = {
                            'width': msg.width,
                            'height': msg.height,
                            'distortion_model': msg.distortion_model,
                            'camera_matrix': {
                                'fx': msg.k[0],  # K[0,0]
                                'fy': msg.k[4],  # K[1,1] 
                                'cx': msg.k[2],  # K[0,2]
                                'cy': msg.k[5],  # K[1,2]
                                'data': list(msg.k)
                            },
                            'distortion_coefficients': list(msg.d),
                            'rectification_matrix': list(msg.r),
                            'projection_matrix': list(msg.p),
                            'frame_id': msg.header.frame_id
                        }
                        
                        logger.info(f"   📊 Resolution: {msg.width}x{msg.height}")
                        logger.info(f"   📊 Distortion model: {msg.distortion_model}")
                        logger.info(f"   📊 fx: {calibration_data['camera_matrix']['fx']:.2f}")
                        logger.info(f"   📊 fy: {calibration_data['camera_matrix']['fy']:.2f}")
                        logger.info(f"   📊 cx: {calibration_data['camera_matrix']['cx']:.2f}")
                        logger.info(f"   📊 cy: {calibration_data['camera_matrix']['cy']:.2f}")
                        logger.info(f"   📊 Distortion coeffs: {len(msg.d)} parameters")
                        
                        break  # Only need first message
                    break
        
        if calibration_data is None:
            logger.error("❌ No camera_info topic found in bag!")
            return None
            
        return calibration_data
    
    def create_orb_slam3_config(self, calibration_data: Dict, bag_name: str) -> Path:
        """Create ORB-SLAM3 configuration file"""
        logger.info("📝 Creating ORB-SLAM3 configuration file...")
        
        # ORB-SLAM3 configuration template
        config = {
            # Camera calibration
            'Camera.fx': float(calibration_data['camera_matrix']['fx']),
            'Camera.fy': float(calibration_data['camera_matrix']['fy']),
            'Camera.cx': float(calibration_data['camera_matrix']['cx']),
            'Camera.cy': float(calibration_data['camera_matrix']['cy']),
            
            # Distortion parameters (assuming plumb_bob/radial-tangential model)
            'Camera.k1': float(calibration_data['distortion_coefficients'][0]) if len(calibration_data['distortion_coefficients']) > 0 else 0.0,
            'Camera.k2': float(calibration_data['distortion_coefficients'][1]) if len(calibration_data['distortion_coefficients']) > 1 else 0.0,
            'Camera.p1': float(calibration_data['distortion_coefficients'][2]) if len(calibration_data['distortion_coefficients']) > 2 else 0.0,
            'Camera.p2': float(calibration_data['distortion_coefficients'][3]) if len(calibration_data['distortion_coefficients']) > 3 else 0.0,
            'Camera.k3': float(calibration_data['distortion_coefficients'][4]) if len(calibration_data['distortion_coefficients']) > 4 else 0.0,
            
            # Camera resolution
            'Camera.width': calibration_data['width'],
            'Camera.height': calibration_data['height'],
            
            # Camera frames per second (will be updated based on actual data)
            'Camera.fps': 27.0,  # Based on analysis report
            
            # Color order (0: BGR, 1: RGB)
            'Camera.RGB': 0,  # BGR format
            
            # ORB Parameters
            'ORBextractor.nFeatures': 1000,
            'ORBextractor.scaleFactor': 1.2,
            'ORBextractor.nLevels': 8,
            'ORBextractor.iniThFAST': 20,
            'ORBextractor.minThFAST': 7,
            
            # Viewer Parameters
            'Viewer.KeyFrameSize': 0.05,
            'Viewer.KeyFrameLineWidth': 1,
            'Viewer.GraphLineWidth': 0.9,
            'Viewer.PointSize': 2,
            'Viewer.CameraSize': 0.08,
            'Viewer.CameraLineWidth': 3,
            'Viewer.ViewpointX': 0,
            'Viewer.ViewpointY': -0.7,
            'Viewer.ViewpointZ': -1.8,
            'Viewer.ViewpointF': 500,
            
            # System Parameters
            'System.LoadAtlasFromFile': 0,
            'System.SaveAtlasToFile': 0,
        }
        
        # Save as YAML file for ORB-SLAM3
        config_path = self.output_dir / f"{bag_name}_camera_config.yaml"
        with open(config_path, 'w') as f:
            f.write("%YAML:1.0\n")
            f.write("---\n")
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f"{key}: \"{value}\"\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"   ✅ ORB-SLAM3 config saved: {config_path}")
        return config_path
    
    def create_camera_info_summary(self, calibration_data: Dict, bag_name: str) -> Path:
        """Create human-readable camera calibration summary"""
        summary_path = self.output_dir / f"{bag_name}_camera_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write(f"# Camera Calibration Summary - {bag_name}\n\n")
            
            f.write("## Camera Parameters\n")
            f.write(f"- **Resolution**: {calibration_data['width']}x{calibration_data['height']}\n")
            f.write(f"- **Distortion Model**: {calibration_data['distortion_model']}\n")
            f.write(f"- **Frame ID**: {calibration_data['frame_id']}\n\n")
            
            f.write("## Intrinsic Matrix (K)\n")
            fx = calibration_data['camera_matrix']['fx']
            fy = calibration_data['camera_matrix']['fy']
            cx = calibration_data['camera_matrix']['cx']
            cy = calibration_data['camera_matrix']['cy']
            
            f.write("```\n")
            f.write(f"K = [ {fx:8.2f}    0.00   {cx:8.2f} ]\n")
            f.write(f"    [    0.00   {fy:8.2f}   {cy:8.2f} ]\n")
            f.write(f"    [    0.00      0.00      1.00 ]\n")
            f.write("```\n\n")
            
            f.write("## Distortion Coefficients\n")
            if calibration_data['distortion_coefficients']:
                d = calibration_data['distortion_coefficients']
                f.write(f"- **k1**: {d[0]:.6f}\n")
                f.write(f"- **k2**: {d[1]:.6f}\n")
                f.write(f"- **p1**: {d[2]:.6f}\n")
                f.write(f"- **p2**: {d[3]:.6f}\n")
                if len(d) > 4:
                    f.write(f"- **k3**: {d[4]:.6f}\n")
            else:
                f.write("- No distortion coefficients\n")
            
            f.write("\n## Field of View\n")
            fov_x = 2 * np.arctan(calibration_data['width'] / (2 * fx)) * 180 / np.pi
            fov_y = 2 * np.arctan(calibration_data['height'] / (2 * fy)) * 180 / np.pi
            f.write(f"- **Horizontal FOV**: {fov_x:.1f}°\n")
            f.write(f"- **Vertical FOV**: {fov_y:.1f}°\n\n")
            
            f.write("## ORB-SLAM3 Ready\n")
            f.write("✅ Camera calibration extracted and formatted for ORB-SLAM3\n")
            f.write("✅ Configuration file generated\n")
            f.write("✅ Ready for visual localization\n")
        
        logger.info(f"   ✅ Camera summary saved: {summary_path}")
        return summary_path
    
    def process_bag(self, bag_path: str) -> bool:
        """Process a single bag and extract calibration"""
        bag_name = Path(bag_path).name
        logger.info(f"🎬 Processing {bag_name}...")
        
        # Extract calibration data
        calibration_data = self.extract_calibration()
        if calibration_data is None:
            return False
        
        # Create ORB-SLAM3 config
        config_path = self.create_orb_slam3_config(calibration_data, bag_name)
        
        # Create summary
        summary_path = self.create_camera_info_summary(calibration_data, bag_name)
        
        logger.info(f"   ✅ Camera calibration extraction complete for {bag_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Extract camera calibration for ORB-SLAM3')
    parser.add_argument('bag_path', help='Path to ROS bag')
    parser.add_argument('--output-dir', default='config', help='Output directory for config files')
    
    args = parser.parse_args()
    
    if not Path(args.bag_path).exists():
        logger.error(f"❌ Bag path does not exist: {args.bag_path}")
        return 1
    
    extractor = CameraCalibrationExtractor(args.bag_path, args.output_dir)
    success = extractor.process_bag(args.bag_path)
    
    if success:
        logger.info("🎉 Camera calibration extraction completed successfully!")
        return 0
    else:
        logger.error("❌ Camera calibration extraction failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
