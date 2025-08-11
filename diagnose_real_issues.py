#!/usr/bin/env python3
"""
诊断实际的3D重建问题
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

def test_basic_system():
    """测试基础系统是否正常工作"""
    print("=" * 50)
    print("Testing Basic System")
    print("=" * 50)
    
    try:
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        
        # 最简单的mock配置
        config = {
            'input': {
                'source': 'mock',
                'mock': {'num_frames': 5, 'image_size': [640, 480]}
            },
            'frontend': {
                'loftr_config': {'device': 'cpu', 'model_path': None},
                'pnp_solver': {}
            },
            'visualization': False,  # 关闭可视化避免卡死
            'monogs': {
                'cam': {'H': 480, 'W': 640, 'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
            }
        }
        
        print("Creating SLAM system...")
        slam_system = HybridSLAMSystem(config, save_dir="diagnose_test")
        
        print(f"System created successfully")
        print(f"MonoGS enabled: {slam_system.use_monogs}")
        print(f"MonoGS backend: {slam_system.monogs_backend is not None}")
        
        # 测试3D重建接口
        print("Testing 3D reconstruction...")
        reconstruction = slam_system.get_3d_reconstruction()
        print(f"Initial reconstruction: {reconstruction}")
        
        # 手动添加一些测试数据到MonoGS后端
        if slam_system.use_monogs and slam_system.monogs_backend:
            test_points = np.random.rand(100, 3) * 5  # 100个随机3D点
            test_colors = np.random.randint(0, 255, (100, 3))
            slam_system.monogs_backend.add_points(test_points, test_colors)
            print("Added test points to MonoGS backend")
            
            # 再次检查重建
            reconstruction = slam_system.get_3d_reconstruction()
            if reconstruction:
                points = reconstruction.get('points')
                print(f"Reconstruction now has {len(points)} points")
                print(f"Reconstruction type: {reconstruction.get('type')}")
                return True
            else:
                print("Still no reconstruction data")
                return False
        else:
            print("MonoGS backend not available")
            return False
            
    except Exception as e:
        print(f"System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera_access():
    """测试摄像头访问"""
    print("\n" + "=" * 50)
    print("Testing Camera Access")
    print("=" * 50)
    
    import cv2
    
    # 测试左摄像头
    print("Testing left camera (device 0)...")
    cap0 = cv2.VideoCapture(0)
    if cap0.isOpened():
        ret, frame = cap0.read()
        if ret:
            print(f"Left camera OK - frame shape: {frame.shape}")
            cap0.release()
            left_ok = True
        else:
            print("Left camera cannot read frame")
            cap0.release()
            left_ok = False
    else:
        print("Cannot open left camera")
        left_ok = False
    
    # 测试右摄像头
    print("Testing right camera (device 1)...")
    cap1 = cv2.VideoCapture(1)
    if cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            print(f"Right camera OK - frame shape: {frame.shape}")
            cap1.release()
            right_ok = True
        else:
            print("Right camera cannot read frame")
            cap1.release()
            right_ok = False
    else:
        print("Cannot open right camera")
        right_ok = False
    
    return left_ok and right_ok

def check_qt_slam_config():
    """检查Qt SLAM配置"""
    print("\n" + "=" * 50)
    print("Checking Qt SLAM Configuration")
    print("=" * 50)
    
    config_file = Path("config/qt_slam_config.json")
    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("Current Qt config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        if not config.get('enable_mono_gs', False):
            print("WARNING: MonoGS is disabled in Qt config")
            
            # 启用MonoGS
            config['enable_mono_gs'] = True
            config['enable_3d_reconstruction'] = True
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("Updated Qt config to enable MonoGS")
            
        return True
    else:
        print("Qt config file not found")
        return False

def create_working_config():
    """创建一个确实能工作的配置"""
    print("\n" + "=" * 50)
    print("Creating Working Configuration")
    print("=" * 50)
    
    # 创建一个非常简单但能工作的配置
    working_config = {
        'input': {
            'source': 'camera',
            'camera': {
                'left_device': 0,
                'right_device': 1,
                'resolution': [640, 480],
                'fps': 15
            }
        },
        'frontend': {
            'loftr_config': {
                'device': 'cpu',  # 强制CPU避免GPU问题
                'model_path': None,
                'resize_to': [640, 480],
                'match_threshold': 0.5  # 更宽松的阈值
            },
            'pnp_solver': {
                'method': 'SOLVEPNP_ITERATIVE',
                'confidence': 0.95,
                'reprojection_threshold': 3.0
            }
        },
        'monogs': {
            'cam': {
                'H': 480, 'W': 640,
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            },
            'mapping': {
                'every_frame': 1,  # 每帧都进行3D重建
                'new_points': 1000
            }
        },
        'visualization': True,
        'visualization_config': {
            'window_size': [1200, 700],
            'show_trajectory': True,
            'show_pointcloud': True,
            'save_mode': False
        },
        'performance_targets': {
            'target_fps': 10,  # 降低FPS确保稳定性
            'max_memory_gb': 4,
            'max_gpu_memory_gb': 2
        },
        'output': {
            'save_trajectory': True,
            'save_intermediate': True,
            'save_reconstruction': True
        }
    }
    
    import yaml
    config_file = Path("configs/working_3d_config.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(working_config, f, default_flow_style=False)
    
    print(f"Working config saved: {config_file}")
    return config_file

def check_visualization_update():
    """检查可视化更新是否正确"""
    print("\n" + "=" * 50) 
    print("Checking Visualization Update")
    print("=" * 50)
    
    try:
        from hybrid_slam.utils.visualization import RealtimeVisualizerExtended
        
        # 创建测试可视化器
        vis = RealtimeVisualizerExtended(
            window_size=(800, 600),
            show_trajectory=True,
            show_pointcloud=True
        )
        
        # 模拟3D重建数据
        test_reconstruction = {
            'points': np.random.rand(500, 3) * 10,  # 500个随机3D点
            'colors': np.random.randint(0, 255, (500, 3)),
            'type': 'gaussian_splatting'
        }
        
        # 模拟vis_data
        vis_data = {
            'left_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'right_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'depth_map': np.random.rand(480, 640) * 10,
            'current_pose': torch.eye(4),
            'trajectory': [],
            '3d_reconstruction': test_reconstruction,
            'tracking_info': {
                'method': 'hybrid',
                'confidence': 0.8,
                'num_matches': 200,
                'processing_time': 50.0
            }
        }
        
        print("Updating visualizer with test data...")
        vis.update(vis_data)
        
        # 检查重建数据是否正确更新
        if hasattr(vis, 'reconstruction_data') and vis.reconstruction_data:
            print("✓ Visualization has reconstruction data")
            print(f"  Points: {len(vis.reconstruction_data['points'])}")
            print(f"  Type: {vis.reconstruction_data['type']}")
            
            # 测试3D预览生成
            preview = vis._create_3d_reconstruction_preview(400, 300)
            if preview is not None:
                print("✓ 3D reconstruction preview generated successfully")
                return True
            else:
                print("✗ Failed to generate 3D reconstruction preview")
                return False
        else:
            print("✗ Visualization missing reconstruction data")
            return False
            
    except Exception as e:
        print(f"Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Real Issues Diagnosis")
    print("=" * 50)
    
    # 运行所有测试
    test1 = test_basic_system()
    test2 = test_camera_access() 
    test3 = check_qt_slam_config()
    config_file = create_working_config()
    test4 = check_visualization_update()
    
    print("\n" + "=" * 50)
    print("Diagnosis Summary:")
    print(f"Basic system: {'✓ OK' if test1 else '✗ FAIL'}")
    print(f"Camera access: {'✓ OK' if test2 else '✗ FAIL'}")
    print(f"Qt config: {'✓ OK' if test3 else '✗ FAIL'}")
    print(f"Visualization: {'✓ OK' if test4 else '✗ FAIL'}")
    print("=" * 50)
    
    if test1 and test4:
        print("✓ 3D reconstruction should work now!")
        print("Try:")
        print(f"  python run_dual_camera_3d.py --config {config_file}")
    elif test2:
        print("⚠ Cameras work but system has issues")
        print("  Check the error messages above")
    else:
        print("✗ Multiple issues found - check cameras and dependencies")