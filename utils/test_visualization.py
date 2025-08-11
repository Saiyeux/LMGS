#!/usr/bin/env python3
"""
简单的可视化测试脚本
"""

import cv2
import numpy as np
import time
from hybrid_slam.utils.visualization import RealtimeVisualizer

def test_visualization():
    """测试可视化系统"""
    print("Testing visualization system...")
    
    # 创建可视化器
    visualizer = RealtimeVisualizer(
        window_size=(1200, 600),
        show_trajectory=True,
        show_pointcloud=False,
        max_trajectory_points=100
    )
    
    # 生成测试数据
    width, height = 640, 480
    
    try:
        for i in range(10):  # 生成10帧测试数据
            print(f"Generating test frame {i+1}/10...")
            
            # 创建测试图像
            left_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            depth_map = np.random.rand(height, width) * 10  # 0-10米深度
            
            # 创建简单的轨迹
            pose = np.eye(4)
            pose[0, 3] = i * 0.1  # X方向移动
            pose[2, 3] = i * 0.05  # Z方向移动
            
            # 更新可视化数据
            vis_data = {
                'left_image': left_img,
                'right_image': right_img,
                'depth_map': depth_map,
                'current_pose': pose,
                'tracking_info': {
                    'method': 'test',
                    'confidence': 0.8,
                    'num_matches': 100 + i * 10,
                    'processing_time': 33.3
                }
            }
            
            visualizer.update(vis_data)
            
            # 渲染
            for _ in range(5):  # 每帧渲染5次
                visualizer.render()
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    print("User pressed 'q' to quit")
                    return
            
            time.sleep(0.1)
        
        print("Test completed! Press any key in the visualization window to exit...")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_visualization()