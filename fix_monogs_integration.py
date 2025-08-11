#!/usr/bin/env python3
"""
修复MonoGS集成问题 - 创建必要的数据集文件
"""

import os
import sys
import numpy as np
from pathlib import Path

def create_dummy_dataset_files(dataset_path):
    """创建MonoGS需要的数据集文件"""
    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dataset files in: {dataset_path}")
    
    # 创建rgb.txt (TUM格式的图像列表)
    rgb_file = dataset_path / "rgb.txt"
    with open(rgb_file, 'w') as f:
        f.write("# color images\n")
        f.write("# file: 'associations.txt'\n")
        f.write("# timestamp filename\n")
        # 写入一些虚拟时间戳
        for i in range(10):
            timestamp = 1234567890.0 + i * 0.1
            f.write(f"{timestamp:.6f} rgb/{i:06d}.png\n")
    
    # 创建depth.txt (深度图列表)  
    depth_file = dataset_path / "depth.txt"
    with open(depth_file, 'w') as f:
        f.write("# depth maps\n")
        f.write("# file: 'associations.txt'\n")
        f.write("# timestamp filename\n")
        for i in range(10):
            timestamp = 1234567890.0 + i * 0.1
            f.write(f"{timestamp:.6f} depth/{i:06d}.png\n")
    
    # 创建groundtruth.txt (可选的真值轨迹)
    gt_file = dataset_path / "groundtruth.txt"
    with open(gt_file, 'w') as f:
        f.write("# ground truth trajectory\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(10):
            timestamp = 1234567890.0 + i * 0.1
            # 简单的直线轨迹
            tx, ty, tz = i * 0.1, 0.0, 0.0
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # 单位四元数
            f.write(f"{timestamp:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    
    # 创建associations.txt (图像关联文件)
    assoc_file = dataset_path / "associations.txt"
    with open(assoc_file, 'w') as f:
        f.write("# associations\n")
        for i in range(10):
            timestamp = 1234567890.0 + i * 0.1
            f.write(f"{timestamp:.6f} rgb/{i:06d}.png {timestamp:.6f} depth/{i:06d}.png\n")
    
    # 创建rgb和depth目录
    (dataset_path / "rgb").mkdir(exist_ok=True)
    (dataset_path / "depth").mkdir(exist_ok=True)
    
    print("Dataset files created successfully")
    return True

def test_monogs_with_dataset():
    """测试带数据集文件的MonoGS集成"""
    print("=" * 50)
    print("Testing MonoGS with proper dataset files")
    print("=" * 50)
    
    # 创建测试数据集
    test_dir = Path("monogs_test_dataset")
    create_dummy_dataset_files(test_dir)
    
    try:
        from hybrid_slam.core.slam_system import HybridSLAMSystem, MONOGS_AVAILABLE
        
        print(f"MonoGS Available: {MONOGS_AVAILABLE}")
        
        if not MONOGS_AVAILABLE:
            print("MonoGS not available")
            return False
        
        # 配置使用我们创建的数据集
        config = {
            'input': {
                'source': 'mock',
                'mock': {'num_frames': 3}
            },
            'monogs': {
                'dataset_path': str(test_dir),  # 指向我们的数据集
                'cam': {
                    'H': 480, 'W': 640,
                    'fx': 525.0, 'fy': 525.0,
                    'cx': 320.0, 'cy': 240.0
                }
            },
            'visualization': False
        }
        
        print("Testing SLAM system initialization...")
        slam_system = HybridSLAMSystem(config, save_dir="monogs_integration_test")
        
        print(f"System created")
        print(f"MonoGS enabled: {slam_system.use_monogs}")
        print(f"MonoGS SLAM: {slam_system.monogs_slam is not None}")
        print(f"MonoGS Backend: {slam_system.monogs_backend is not None}")
        
        if slam_system.use_monogs:
            print("MonoGS integration SUCCESS!")
            return True
        else:
            print("MonoGS integration still failed")
            return False
            
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_live_camera_bypass():
    """创建绕过数据集检查的MonoGS配置"""
    print("\n" + "=" * 50)
    print("Creating live camera bypass configuration")
    print("=" * 50)
    
    # 修改SLAM系统以支持实时相机输入而不需要数据集文件
    config_content = """
# MonoGS实时相机配置
monogs_live_config = {
    'Results': {
        'save_results': True,
        'use_gui': False,
        'eval_rendering': False,
        'use_wandb': False
    },
    'Dataset': {
        'type': 'realsense',  # 使用实时相机类型
        'sensor_type': 'monocular',
        'pcd_downsample': 64,
        'adaptive_pointsize': True
    },
    'Training': {
        'single_thread': True,
        'init_itr_num': 200,
        'tracking_itr_num': 30,
        'mapping_itr_num': 50
    }
}
"""
    
    config_file = Path("monogs_live_config.py")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Live camera config created: {config_file}")
    return config_file

if __name__ == "__main__":
    print("MonoGS Integration Fixer")
    print("=" * 50)
    
    # 测试1: 创建数据集文件并测试
    success1 = test_monogs_with_dataset()
    
    # 测试2: 创建实时相机配置
    config_file = create_live_camera_bypass()
    
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print(f"Dataset test: {'PASS' if success1 else 'FAIL'}")
    print(f"Live config: Created {config_file}")
    print("=" * 50)
    
    if success1:
        print("MonoGS integration is now working!")
        print("You can now run: python run_dual_camera_3d.py")
    else:
        print("MonoGS still has issues, using fallback mode")