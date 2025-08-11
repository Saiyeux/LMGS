#!/usr/bin/env python3
import cv2
import numpy as np
import torch
from hybrid_slam.core.slam_system import HybridSLAMSystem

def simple_3d_test():
    print("Starting simple 3D reconstruction test...")
    
    config = {
        'input': {
            'source': 'camera',
            'camera': {
                'left_device': 0,
                'right_device': 1,
                'resolution': [640, 480]
            }
        },
        'frontend': {
            'loftr_config': {'device': 'cpu', 'model_path': None},
            'pnp_solver': {}
        },
        'monogs': {
            'cam': {'H': 480, 'W': 640, 'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
        },
        'visualization': True,
        'performance_targets': {'target_fps': 8}
    }
    
    try:
        slam_system = HybridSLAMSystem(config, save_dir="simple_test")
        print(f"MonoGS enabled: {slam_system.use_monogs}")
        
        # 手动添加测试数据
        if slam_system.monogs_backend:
            test_points = np.random.rand(200, 3) * 5
            test_colors = np.random.randint(50, 255, (200, 3))
            slam_system.monogs_backend.add_points(test_points, test_colors)
            print("Added test 3D points")
        
        # 运行几秒钟
        print("Running for 10 seconds...")
        import time
        start_time = time.time()
        
        while time.time() - start_time < 10:
            try:
                if slam_system.dataset:
                    frame = next(iter(slam_system.dataset))
                    result = slam_system._process_stereo_frame(frame)
                    
                    if slam_system.visualizer:
                        # 获取3D重建数据
                        reconstruction = slam_system.get_3d_reconstruction()
                        
                        vis_data = {
                            'left_image': frame.left_image,
                            'right_image': frame.right_image,
                            'depth_map': frame.depth_map,
                            'current_pose': result.pose,
                            'trajectory': slam_system.trajectory,
                            '3d_reconstruction': reconstruction,
                            'tracking_info': {
                                'method': result.tracking_method,
                                'confidence': result.confidence,
                                'num_matches': result.num_matches,
                                'processing_time': result.processing_time
                            }
                        }
                        
                        slam_system.visualizer.update(vis_data)
                        slam_system.visualizer.render()
                        
                        if reconstruction:
                            points = reconstruction.get('points', [])
                            print(f"Frame processed - 3D points: {len(points)}")
                
                time.sleep(0.1)
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Frame error: {e}")
                continue
        
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_3d_test()
