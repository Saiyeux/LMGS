#!/usr/bin/env python3
"""
直接测试模拟数据的可视化
"""

import cv2
import numpy as np
import time
from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset
from hybrid_slam.utils.visualization import RealtimeVisualizer

def test_mock_visualization():
    """测试模拟数据可视化"""
    print("Creating mock dataset...")
    
    # 创建模拟数据集
    dataset = create_mock_stereo_dataset(num_frames=100, resolution=(640, 480))
    
    print("Creating visualizer...")
    # 创建可视化器
    visualizer = RealtimeVisualizer(
        window_size=(1200, 600),
        show_trajectory=True,
        show_pointcloud=False
    )
    
    print("Starting visualization loop...")
    try:
        frame_count = 0
        for stereo_frame in dataset:
            if frame_count > 50:  # 只显示50帧
                break
                
            print(f"Processing frame {frame_count + 1}")
            
            # 创建假的跟踪结果
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = frame_count * 0.1  # X方向移动
            pose[2, 3] = frame_count * 0.05  # Z方向移动
            
            # 创建可视化数据
            vis_data = {
                'left_image': stereo_frame.left_image,
                'right_image': stereo_frame.right_image,
                'depth_map': None,  # 暂时不添加深度图
                'current_pose': pose,
                'tracking_info': {
                    'method': 'mock',
                    'confidence': 0.9,
                    'num_matches': 150,
                    'processing_time': 33.3
                }
            }
            
            # 更新并渲染
            visualizer.update(vis_data)
            visualizer.render()
            
            # 检查退出
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("User quit")
                break
            
            frame_count += 1
        
        print("Visualization test completed!")
        print("Press any key in visualization window to exit...")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        visualizer.close()
        dataset.close()

if __name__ == "__main__":
    test_mock_visualization()