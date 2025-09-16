#!/usr/bin/env python3
"""
Simple Visual Localization Test
A minimal implementation to test visual localization concepts with your ROS bag data
before full ORB-SLAM3 integration
"""

import os
import sys
import cv2
import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVisualLocalizer:
    def __init__(self, bag_path: str, config_path: str, output_dir: str = "results"):
        self.bag_path = bag_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ROS message types
        self.typestore = get_typestore(Stores.LATEST)
        
        # Load camera configuration
        self.camera_config = self.load_camera_config()
        
        # Initialize ORB detector
        self.orb_detector = cv2.ORB_create(nfeatures=1000)
        
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Storage for tracking
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.trajectory_points = []
        
    def load_camera_config(self) -> Dict:
        """Load camera configuration from YAML file"""
        logger.info(f"📷 Loading camera config from {self.config_path}")
        
        # Read and parse OpenCV YAML format
        with open(self.config_path, 'r') as f:
            lines = f.readlines()
        
        # Skip OpenCV YAML header and parse key-value pairs
        config = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('---') and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert to appropriate type
                try:
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    config[key] = value
        
        # Extract camera matrix
        camera_matrix = np.array([
            [config['Camera.fx'], 0, config['Camera.cx']],
            [0, config['Camera.fy'], config['Camera.cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Extract distortion coefficients
        dist_coeffs = np.array([
            config['Camera.k1'],
            config['Camera.k2'],
            config['Camera.p1'],
            config['Camera.p2'],
            config['Camera.k3']
        ], dtype=np.float32)
        
        logger.info(f"   ✅ Camera matrix loaded: fx={config['Camera.fx']:.1f}, fy={config['Camera.fy']:.1f}")
        logger.info(f"   ✅ Principal point: cx={config['Camera.cx']:.1f}, cy={config['Camera.cy']:.1f}")
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'width': config['Camera.width'],
            'height': config['Camera.height']
        }
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect ORB features
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []
        
        # Match descriptors
        matches = self.matcher.match(desc1, desc2)
        
        # Sort by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    def estimate_motion(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        """Estimate camera motion between frames using essential matrix"""
        if len(matches) < 8:  # Need at least 8 points for essential matrix
            return None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find essential matrix
        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, 
            self.camera_config['camera_matrix'],
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if essential_matrix is None:
            return None
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(
            essential_matrix, pts1, pts2,
            self.camera_config['camera_matrix']
        )
        
        return R, t, len(matches), np.sum(mask)
    
    def process_rosbag(self, max_frames: int = 100) -> List[Dict]:
        """Process ROS bag and perform simple visual localization"""
        logger.info(f"🎬 Processing {self.bag_path} for visual localization test...")
        
        results = []
        frame_count = 0
        
        with Reader(self.bag_path) as reader:
            for connection in reader.connections:
                if 'image' in connection.topic and 'rect' in connection.topic:
                    logger.info(f"📷 Processing camera topic: {connection.topic}")
                    
                    for connection, timestamp, rawdata in reader.messages(connections=[connection]):
                        if frame_count >= max_frames:
                            break
                            
                        msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                        
                        # Decode image
                        img_data = np.frombuffer(msg.data, dtype=np.uint8)
                        image = img_data.reshape((msg.height, msg.width, 3))
                        
                        # Extract features
                        keypoints, descriptors = self.extract_features(image)
                        
                        result = {
                            'frame_id': frame_count,
                            'timestamp': int(timestamp),  # Convert to int for JSON
                            'num_features': len(keypoints),
                            'image_shape': list(image.shape)  # Convert to list for JSON
                        }
                        
                        # Track motion if we have a previous frame
                        if self.previous_descriptors is not None:
                            matches = self.match_features(self.previous_descriptors, descriptors)
                            
                            if len(matches) > 20:  # Good number of matches
                                motion_result = self.estimate_motion(
                                    self.previous_keypoints, keypoints, matches
                                )
                                
                                if motion_result is not None:
                                    R, t, total_matches, inlier_matches = motion_result
                                    result.update({
                                        'total_matches': int(total_matches),
                                        'inlier_matches': int(inlier_matches),
                                        'rotation_matrix': R.tolist(),
                                        'translation': t.flatten().tolist(),
                                        'tracking_quality': float(inlier_matches / total_matches if total_matches > 0 else 0.0)
                                    })
                        
                        # Store for next iteration
                        self.previous_frame = image
                        self.previous_keypoints = keypoints
                        self.previous_descriptors = descriptors
                        
                        results.append(result)
                        frame_count += 1
                        
                        if frame_count % 10 == 0:
                            logger.info(f"   📊 Processed {frame_count} frames...")
                    
                    break  # Only process first camera topic
        
        logger.info(f"   ✅ Processed {frame_count} frames total")
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze visual localization results"""
        logger.info("📊 Analyzing visual localization results...")
        
        # Calculate statistics
        total_frames = len(results)
        frames_with_tracking = sum(1 for r in results if 'tracking_quality' in r)
        
        if frames_with_tracking > 0:
            avg_features = np.mean([r['num_features'] for r in results])
            avg_matches = np.mean([r.get('total_matches', 0) for r in results if 'total_matches' in r])
            avg_quality = np.mean([r.get('tracking_quality', 0) for r in results if 'tracking_quality' in r])
        else:
            avg_features = avg_matches = avg_quality = 0
        
        analysis = {
            'total_frames': total_frames,
            'frames_with_tracking': frames_with_tracking,
            'tracking_success_rate': frames_with_tracking / total_frames * 100,
            'avg_features_per_frame': avg_features,
            'avg_matches_per_frame': avg_matches,
            'avg_tracking_quality': avg_quality
        }
        
        logger.info(f"   📊 Tracking success rate: {analysis['tracking_success_rate']:.1f}%")
        logger.info(f"   📊 Average features per frame: {analysis['avg_features_per_frame']:.0f}")
        logger.info(f"   📊 Average matches per frame: {analysis['avg_matches_per_frame']:.0f}")
        logger.info(f"   📊 Average tracking quality: {analysis['avg_tracking_quality']:.2f}")
        
        return analysis
    
    def save_results(self, results: List[Dict], analysis: Dict, bag_name: str):
        """Save results to files"""
        results_path = self.output_dir / f"{bag_name}_visual_localization_test.json"
        
        import json
        with open(results_path, 'w') as f:
            json.dump({
                'analysis': analysis,
                'frame_results': results
            }, f, indent=2)
        
        logger.info(f"   ✅ Results saved: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple Visual Localization Test')
    parser.add_argument('bag_path', help='Path to ROS bag')
    parser.add_argument('config_path', help='Path to camera config YAML file')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum frames to process')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.bag_path).exists():
        logger.error(f"❌ Bag path does not exist: {args.bag_path}")
        return 1
    
    if not Path(args.config_path).exists():
        logger.error(f"❌ Config path does not exist: {args.config_path}")
        return 1
    
    # Run visual localization test
    localizer = SimpleVisualLocalizer(args.bag_path, args.config_path, args.output_dir)
    results = localizer.process_rosbag(args.max_frames)
    
    # Analyze results
    analysis = localizer.analyze_results(results)
    
    # Save results
    bag_name = Path(args.bag_path).name
    localizer.save_results(results, analysis, bag_name)
    
    logger.info("🎉 Visual localization test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
