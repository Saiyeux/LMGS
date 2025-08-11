#!/usr/bin/env python3
"""
最简单的可视化器
"""

import cv2
import numpy as np

class SimpleVisualizer:
    def __init__(self):
        print("Creating simple visualizer...")
        cv2.namedWindow('Simple Visualization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Simple Visualization', 1200, 600)
        print("Visualizer ready!")
        
    def show_frame(self, left_image, right_image):
        """显示一帧数据"""
        # 创建画布
        canvas = np.zeros((600, 1200, 3), dtype=np.uint8)
        
        if left_image is not None:
            left_resized = cv2.resize(left_image, (300, 240))
            canvas[10:250, 10:310] = left_resized
            cv2.putText(canvas, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if right_image is not None:
            right_resized = cv2.resize(right_image, (300, 240))
            canvas[10:250, 320:620] = right_resized
            cv2.putText(canvas, "Right Camera", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Simple Visualization', canvas)
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        cv2.destroyAllWindows()

def test_simple_visualization():
    """测试简单可视化"""
    print("Testing simple visualization...")
    
    try:
        # 创建可视化器
        vis = SimpleVisualizer()
        
        # 从模拟数据集获取数据
        from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset
        dataset = create_mock_stereo_dataset(num_frames=10, resolution=(640, 480))
        
        count = 0
        for stereo_frame in dataset:
            print(f"Showing frame {count + 1}")
            
            key = vis.show_frame(stereo_frame.left_image, stereo_frame.right_image)
            if key == ord('q'):
                break
            
            count += 1
            if count >= 10:
                break
        
        print("Test completed! Press any key to exit...")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        vis.close()

if __name__ == "__main__":
    test_simple_visualization()