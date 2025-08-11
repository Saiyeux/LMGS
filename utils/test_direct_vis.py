#!/usr/bin/env python3
"""
直接用OpenCV显示模拟数据
"""

import cv2
import numpy as np
from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset

def test_direct_visualization():
    """直接显示模拟数据"""
    print("Creating mock dataset...")
    dataset = create_mock_stereo_dataset(num_frames=30, resolution=(640, 480))
    
    print("Starting direct visualization...")
    cv2.namedWindow('Mock Data Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mock Data Visualization', 1200, 600)
    
    try:
        frame_count = 0
        for stereo_frame in dataset:
            if frame_count > 20:
                break
            
            print(f"Frame {frame_count + 1}: Left={stereo_frame.left_image.shape}, Right={stereo_frame.right_image.shape}")
            
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
            cv2.putText(canvas, f"Frame {frame_count + 1}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, "Mock Data Test", (650, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 绘制分割线
            cv2.line(canvas, (310, 0), (310, 600), (100, 100, 100), 2)
            cv2.line(canvas, (0, 260), (1200, 260), (100, 100, 100), 2)
            
            # 显示
            cv2.imshow('Mock Data Visualization', canvas)
            
            # 等待
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                print("User quit")
                break
            
            frame_count += 1
        
        print("Test completed successfully!")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        dataset.close()

if __name__ == "__main__":
    test_direct_visualization()