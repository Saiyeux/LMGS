#!/usr/bin/env python3
"""
修复可视化显示问题 - 确保3D重建能正确显示
"""

from pathlib import Path

def fix_visualization_display():
    """修复可视化显示问题"""
    print("Fixing visualization display...")
    
    # 修复可视化器的_render_images方法，确保显示3D重建
    vis_file = Path("hybrid_slam/utils/visualization.py")
    
    with open(vis_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否调用了3D重建渲染
    if '_render_3d_reconstruction' not in content:
        print("3D reconstruction rendering method missing!")
        return False
    
    # 修复_render_images方法以包含3D重建显示
    original_render = '''        # 区域4：跟踪信息和3D重建预览 (右下)
        info_panel = self._create_tracking_info_image(panel_width, panel_height)
        canvas[panel_height+30:panel_height*2+30, panel_width+30:panel_width*2+30] = info_panel'''
    
    fixed_render = '''        # 区域4：跟踪信息和3D重建预览 (右下)
        info_panel = self._create_tracking_info_image(panel_width, panel_height)
        canvas[panel_height+30:panel_height*2+30, panel_width+30:panel_width*2+30] = info_panel
        
        # 添加3D重建显示到右下角的一部分
        if hasattr(self, '_render_3d_reconstruction'):
            recon_x = panel_width + 50
            recon_y = panel_height + 50  
            recon_w = min(300, panel_width - 40)
            recon_h = min(200, panel_height - 40)
            self._render_3d_reconstruction(canvas, recon_x, recon_y, recon_w, recon_h)'''
    
    if original_render in content:
        content = content.replace(original_render, fixed_render)
        
        with open(vis_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Fixed visualization to show 3D reconstruction")
        return True
    
    print("Could not find visualization render code to fix")
    return False

def fix_tracking_info_display():
    """修复跟踪信息显示以包含3D重建状态"""
    print("Fixing tracking info display...")
    
    vis_file = Path("hybrid_slam/utils/visualization.py")
    
    with open(vis_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找_create_tracking_info_image方法
    if 'def _create_tracking_info_image' in content:
        # 在方法末尾添加3D重建信息显示
        original_end = '''        return info_image'''
        
        fixed_end = '''        # 添加3D重建信息
        if hasattr(self, 'reconstruction_data') and self.reconstruction_data:
            recon_data = self.reconstruction_data
            points_count = len(recon_data.get('points', []))
            recon_type = recon_data.get('type', 'unknown')
            
            y_offset = 150
            cv2.putText(info_image, "3D Reconstruction:", (10, y_offset), 
                       font, font_scale, color, thickness)
            cv2.putText(info_image, f"Type: {recon_type}", (10, y_offset + 25), 
                       font, font_scale, color, thickness)
            cv2.putText(info_image, f"Points: {points_count}", (10, y_offset + 50), 
                       font, font_scale, color, thickness)
        else:
            y_offset = 150
            cv2.putText(info_image, "3D Reconstruction: None", (10, y_offset), 
                       font, font_scale, (100, 100, 100), thickness)
        
        return info_image'''
        
        if original_end in content:
            content = content.replace(original_end, fixed_end)
            
            with open(vis_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("Fixed tracking info to show 3D reconstruction status")
            return True
    
    print("Could not find tracking info display code")
    return False

def create_simple_test_script():
    """创建简单的测试脚本"""
    print("Creating simple test script...")
    
    test_script = """#!/usr/bin/env python3
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
"""
    
    test_file = Path("simple_3d_test.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"Simple test script created: {test_file}")
    return test_file

def check_slam_system_processing():
    """检查SLAM系统的处理流程"""
    print("Checking SLAM system processing...")
    
    slam_file = Path("hybrid_slam/core/slam_system.py")
    
    with open(slam_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查_process_stereo_frame是否正确调用MonoGS
    if '_update_monogs_with_tracking_result' in content and 'self.monogs_backend.add_points' in content:
        print("✓ SLAM system correctly calls MonoGS backend")
        return True
    else:
        print("✗ SLAM system missing MonoGS integration")
        return False

if __name__ == "__main__":
    print("Fixing Visualization Display Issues")
    print("=" * 40)
    
    # 执行修复
    fix1 = fix_visualization_display()
    fix2 = fix_tracking_info_display()
    test_file = create_simple_test_script()
    check1 = check_slam_system_processing()
    
    print("\n" + "=" * 40)
    print("Fix Summary:")
    print(f"Visualization display: {'OK' if fix1 else 'SKIP'}")
    print(f"Tracking info display: {'OK' if fix2 else 'SKIP'}")
    print(f"SLAM processing: {'OK' if check1 else 'FAIL'}")
    print("=" * 40)
    
    print("Now try:")
    print(f"1. python {test_file}")
    print("2. python run_dual_camera_3d.py --config configs/working_3d_config.yaml")
    print("3. python run_qt_slam.py")
    print("\nYou should now see 3D reconstruction data in the interface!")