#!/usr/bin/env python3
"""
Process All ROS Bags Script
Sequentially processes all ROS bags to generate enhanced synchronized videos
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_video_generation(bag_path, output_dir="reports/visualizations"):
    """Run video generation for a single bag"""
    logger.info(f"🎬 Starting video generation for {bag_path}")
    
    try:
        # Build command (with test-frames=0 to process entire bag)
        cmd = [
            sys.executable,  # Use current Python interpreter
            "scripts/enhanced_synchronized_video_generator.py",
            str(bag_path),
            "--output-dir", output_dir,
            "--test-frames", "0"  # 0 means process entire bag
        ]
        
        logger.info(f"   📋 Command: {' '.join(cmd)}")
        
        # Run the command with real-time output
        start_time = time.time()
        logger.info("   📋 Starting video generation (showing full output):")
        logger.info("   " + "="*80)
        
        # Run with real-time output
        result = subprocess.run(cmd, cwd=Path.cwd())
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        logger.info("   " + "="*80)
        
        if result.returncode == 0:
            logger.info(f"   ✅ Video generation completed for {bag_path}")
            logger.info(f"   ⏱️  Processing time: {processing_time:.1f} seconds")
            return True
        else:
            logger.error(f"   ❌ Video generation failed for {bag_path}")
            logger.error(f"   📋 Return code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Exception during video generation for {bag_path}: {e}")
        return False

def main():
    """Main function to process all bags"""
    logger.info("🚀 Starting sequential processing of all ROS bags for ENTIRE LENGTH...")
    logger.info("📋 This will process ALL frames in each bag (not just test frames)")
    
    # Define bag paths
    bag_paths = [
        "data/log_0_ros2",
        "data/log_1_ros2"
    ]
    
    # Check if bags exist
    existing_bags = []
    for bag_path in bag_paths:
        if Path(bag_path).exists():
            existing_bags.append(bag_path)
            logger.info(f"   ✅ Found bag: {bag_path}")
        else:
            logger.warning(f"   ⚠️  Bag not found: {bag_path}")
    
    if not existing_bags:
        logger.error("❌ No ROS bags found to process!")
        return False
    
    # Process each bag sequentially
    successful = 0
    total = len(existing_bags)
    total_start_time = time.time()
    
    for i, bag_path in enumerate(existing_bags, 1):
        logger.info(f"📦 Processing bag {i}/{total}: {bag_path}")
        
        # Run video generation
        if run_video_generation(bag_path):
            successful += 1
        else:
            logger.error(f"❌ Failed to process {bag_path}")
        
        # Add spacing between bags
        if i < total:
            logger.info("")  # Add spacing
            logger.info("⏳ Waiting 2 seconds before next bag...")
            time.sleep(2)
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   📦 Total bags processed: {total}")
    logger.info(f"   ✅ Successfully processed: {successful}")
    logger.info(f"   ❌ Failed: {total - successful}")
    logger.info(f"   ⏱️  Total processing time: {total_processing_time:.1f} seconds")
    logger.info(f"   📁 Output directory: reports/visualizations")
    
    if successful == total:
        logger.info("🎉 All bags processed successfully!")
        return True
    else:
        logger.error("❌ Some bags failed to process. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
