#!/usr/bin/env python3
"""
测试改进的模拟数据生成
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_slam.datasets.dataset_factory import MockStereoDataset

def test_improved_mock_data():
    """测试改进的模拟数据生成"""
    print("Testing improved mock data generation...")
    
    # 创建模拟数据集
    dataset = MockStereoDataset(num_frames=10, resolution=(640, 480))
    
    # 创建输出目录
    output_dir = Path("test_improved_mock_output")
    output_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    for stereo_frame in dataset:
        if frame_count >= 5:  # 只测试前5帧
            break
            
        print(f"Frame {frame_count}:")
        print(f"  Timestamp: {stereo_frame.timestamp:.3f}")
        print(f"  Left image shape: {stereo_frame.left_image.shape}")
        print(f"  Right image shape: {stereo_frame.right_image.shape}")
        print(f"  Depth map shape: {stereo_frame.depth_map.shape if stereo_frame.depth_map is not None else 'None'}")
        print(f"  Number of features: {stereo_frame.num_features}")
        print(f"  Baseline: {stereo_frame.baseline}")
        
        # 保存图像
        cv2.imwrite(str(output_dir / f"left_{frame_count:03d}.png"), stereo_frame.left_image)
        cv2.imwrite(str(output_dir / f"right_{frame_count:03d}.png"), stereo_frame.right_image)
        
        # 保存深度图（如果存在）
        if stereo_frame.depth_map is not None:
            depth_normalized = cv2.normalize(stereo_frame.depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(str(output_dir / f"depth_{frame_count:03d}.png"), depth_colored)
        
        frame_count += 1
    
    print(f"Generated {frame_count} test frames in {output_dir}")
    print("Test completed successfully!")
    
    dataset.close()

if __name__ == "__main__":
    test_improved_mock_data()