#!/usr/bin/env python3
"""
最简单的系统测试
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """最简单的测试"""
    print("Testing basic imports...")
    
    try:
        # 测试数据结构
        from hybrid_slam.utils.data_structures import StereoFrame
        print("OK StereoFrame import")
        
        # 测试数据源
        from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset
        print("OK Dataset factory import")
        
        # 创建测试数据
        dataset = create_mock_stereo_dataset(num_frames=3, resolution=(100, 100))
        print("OK Mock dataset created")
        
        # 测试迭代一帧
        frame = next(iter(dataset))
        print(f"OK Frame generated: {frame.left_image.shape}")
        
        print("\nSUCCESS: Basic functionality working!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())