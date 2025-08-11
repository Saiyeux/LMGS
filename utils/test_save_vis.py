#!/usr/bin/env python3
"""
保存可视化图像而不是实时显示
"""

import cv2
import numpy as np
from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset

def test_save_visualization():
    """保存可视化图像"""
    print("Creating mock dataset...")
    dataset = create_mock_stereo_dataset(num_frames=5, resolution=(640, 480))
    
    print("Generating visualization images...")
    
    frame_count = 0
    for stereo_frame in dataset:
        if frame_count >= 5:
            break
        
        print(f"Processing frame {frame_count + 1}")
        
        # 创建统一画布
        canvas = np.zeros((600, 1200, 3), dtype=np.uint8)
        
        # 调整图像大小
        left_resized = cv2.resize(stereo_frame.left_image, (300, 240))
        right_resized = cv2.resize(stereo_frame.right_image, (300, 240))
        
        # 放置图像
        canvas[10:250, 10:310] = left_resized      # 左上角
        canvas[10:250, 320:620] = right_resized     # 右上角
        
        # 添加标签
        cv2.putText(canvas, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "Right Camera", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, f"Mock Frame {frame_count + 1}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "SLAM Visualization Test", (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 绘制分割线
        cv2.line(canvas, (310, 0), (310, 600), (100, 100, 100), 2)
        cv2.line(canvas, (0, 260), (1200, 260), (100, 100, 100), 2)
        
        # 保存图像
        filename = f"visualization_frame_{frame_count+1:03d}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved {filename}")
        
        frame_count += 1
    
    print("All visualization images saved!")
    print("Generated files:", [f"visualization_frame_{i+1:03d}.png" for i in range(frame_count)])

if __name__ == "__main__":
    test_save_visualization()