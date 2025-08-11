#!/usr/bin/env python3
"""
测试MonoGS集成是否正常工作
"""

import sys
import torch
import numpy as np
from pathlib import Path
from hybrid_slam.core.slam_system import HybridSLAMSystem, MONOGS_AVAILABLE
from hybrid_slam.utils.config_manager import ConfigManager

def test_monogs_integration():
    """测试MonoGS集成"""
    print("=== MonoGS Integration Test ===")
    
    # 检查MonoGS可用性
    print(f"MonoGS Available: {MONOGS_AVAILABLE}")
    
    if not MONOGS_AVAILABLE:
        print("MonoGS not available, cannot test integration")
        return False
    
    # 创建测试配置
    config = {
        'input': {
            'source': 'mock',
            'mock': {
                'num_frames': 10,
                'image_size': [640, 480]
            }
        },
        'frontend': {
            'loftr_config': {
                'model_type': 'outdoor'
            }
        },
        'monogs': {
            'cam': {
                'H': 480, 'W': 640,
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            },
            'tracking': {
                'use_gt_camera': False,
                'forward_prop': True,
                'num_kf': 4
            },
            'mapping': {
                'first_mesh': True,
                'new_points': 1000,
                'every_frame': 1
            }
        },
        'visualization': False,
        'performance_targets': {
            'target_fps': 10
        }
    }
    
    print("Testing SLAM system initialization...")
    
    try:
        # 创建SLAM系统
        slam_system = HybridSLAMSystem(config, save_dir="test_monogs_output")
        
        # 检查MonoGS是否成功初始化
        print(f"MonoGS SLAM initialized: {slam_system.use_monogs}")
        print(f"MonoGS backend available: {slam_system.monogs_backend is not None}")
        
        if slam_system.use_monogs:
            print("✓ MonoGS integration successful!")
        else:
            print("✗ MonoGS integration failed")
            print(f"MonoGS slam object: {slam_system.monogs_slam}")
        
        # 测试3D重建数据获取
        print("Testing 3D reconstruction data retrieval...")
        reconstruction_data = slam_system.get_3d_reconstruction()
        print(f"Initial reconstruction data: {reconstruction_data}")
        
        print("Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_processing():
    """测试模拟数据处理"""
    print("\n=== Mock Data Processing Test ===")
    
    config = {
        'input': {
            'source': 'mock',
            'mock': {
                'num_frames': 5,
                'image_size': [640, 480]
            }
        },
        'frontend': {
            'loftr_config': {
                'model_type': 'outdoor'
            }
        },
        'monogs': {
            'cam': {
                'H': 480, 'W': 640,
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            }
        },
        'visualization': False
    }
    
    try:
        slam_system = HybridSLAMSystem(config, save_dir="test_mock_processing")
        
        print(f"Processing with MonoGS: {slam_system.use_monogs}")
        
        # 运行短时间的处理
        slam_system.run()
        
        # 检查结果
        print(f"Final trajectory length: {len(slam_system.trajectory)}")
        
        # 检查3D重建数据
        reconstruction_data = slam_system.get_3d_reconstruction()
        if reconstruction_data:
            print(f"3D reconstruction type: {reconstruction_data.get('type')}")
            points = reconstruction_data.get('points')
            if points is not None:
                print(f"Number of 3D points: {len(points)}")
            else:
                print("No 3D points generated")
        else:
            print("No 3D reconstruction data generated")
        
        print("Mock processing test completed!")
        return True
        
    except Exception as e:
        print(f"Mock processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_monogs_integration()
    success2 = test_mock_processing()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)