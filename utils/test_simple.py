#!/usr/bin/env python3
"""
简单测试Hybrid SLAM系统
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """简单测试"""
    print("Testing Hybrid SLAM System...")
    
    try:
        # 测试基本导入
        from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset
        print("OK: Mock dataset import successful")
        
        # 创建模拟数据
        dataset = create_mock_stereo_dataset(num_frames=5, resolution=(320, 240))
        print("OK: Mock dataset created")
        
        # 测试迭代
        frame_count = 0
        for frame in dataset:
            frame_count += 1
            print(f"Frame {frame_count}: {frame.left_image.shape}, baseline={frame.baseline}")
            if frame_count >= 3:
                break
        
        print("OK: Basic functionality working")
        
        # 测试配置管理
        test_config = {
            'project_name': 'Test',
            'input': {'source': 'mock', 'camera': {'resolution': [320, 240]}},
            'output': {'save_trajectory': False},
            'frontend': {'min_matches': 5},
            'visualization': False,
            'device': 'cpu'
        }
        
        # 测试核心系统初始化
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        print("OK: Core system import successful")
        
        slam = HybridSLAMSystem(test_config, save_dir="test_results")
        print("OK: SLAM system initialized")
        
        # 清理
        slam._shutdown()
        
        print("\nSUCCESS: All tests passed!")
        return 0
        
    except Exception as e:
        print(f"ERROR: Test failed - {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())