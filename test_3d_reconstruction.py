#!/usr/bin/env python3
"""
测试3D重建功能
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_slam.utils.monogs_adapter import MonoGSAdapter
from hybrid_slam.utils.data_structures import StereoFrame
from hybrid_slam.frontend.hybrid_frontend import TrackingResult

def test_monogs_adapter():
    """测试MonoGS适配器功能"""
    print("Testing MonoGS Adapter...")
    
    # 创建适配器
    config = {
        'save_intermediate': True,
        'max_frames': 100
    }
    
    adapter = MonoGSAdapter(config)
    
    # 模拟一些帧数据
    for i in range(20):
        # 创建模拟立体帧
        left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map = np.random.uniform(0.5, 5.0, (480, 640))
        
        # 相机参数
        K_left = np.array([[525.0, 0, 320.0],
                           [0, 525.0, 240.0],
                           [0, 0, 1.0]])
        K_right = K_left.copy()
        
        stereo_frame = StereoFrame(
            frame_id=i,
            timestamp=i * 0.1,
            left_image=left_img,
            right_image=right_img,
            camera_matrices=(K_left, K_right),
            baseline=0.1,
            depth_map=depth_map
        )
        
        # 创建跟踪结果
        pose = torch.eye(4)
        pose[0, 3] = i * 0.1  # 沿x轴移动
        pose[2, 3] = i * 0.05  # 沿z轴移动
        
        tracking_result = TrackingResult(
            success=True,
            pose=pose,
            tracking_method='feature',
            num_matches=100 + i * 5,
            num_inliers=80 + i * 3,
            reprojection_error=1.0 - i * 0.01,
            processing_time=50.0,
            confidence=0.8 + i * 0.01
        )
        
        # 添加到适配器
        adapter.add_frame(stereo_frame, tracking_result)
        
        if i % 5 == 0:
            print(f"Processed frame {i}")
    
    # 获取统计信息
    stats = adapter.get_statistics()
    print(f"Statistics: {stats}")
    
    # 保存重建数据
    save_dir = Path("test_reconstruction_output")
    adapter.save_reconstruction_data(save_dir)
    
    # 导出可视化数据
    vis_files = adapter.export_for_visualization(save_dir)
    print(f"Visualization files: {vis_files}")
    
    print("MonoGS Adapter test completed successfully!")
    return save_dir

def main():
    """主测试函数"""
    print("=" * 60)
    print("3D Reconstruction Test")
    print("=" * 60)
    
    try:
        save_dir = test_monogs_adapter()
        
        # 检查生成的文件
        print(f"\nGenerated files in {save_dir}:")
        if save_dir.exists():
            for file_path in save_dir.rglob("*"):
                if file_path.is_file():
                    print(f"  {file_path.relative_to(save_dir)} ({file_path.stat().st_size} bytes)")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())