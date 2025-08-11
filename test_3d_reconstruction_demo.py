#!/usr/bin/env python3
"""
3D重建演示测试
快速验证MonoGS集成是否工作
"""

import sys
import time
import numpy as np
from pathlib import Path

from hybrid_slam.core.slam_system import HybridSLAMSystem, MONOGS_AVAILABLE

def test_3d_reconstruction():
    """测试3D重建功能"""
    print("=" * 50)
    print("3D Reconstruction Demo Test")
    print("=" * 50)
    
    # 检查MonoGS可用性
    print(f"MonoGS Available: {MONOGS_AVAILABLE}")
    
    if not MONOGS_AVAILABLE:
        print("❌ MonoGS not available - running basic stereo mode")
    else:
        print("✅ MonoGS ready for 3D reconstruction")
    
    # 创建演示配置
    config = {
        'input': {
            'source': 'camera',
            'camera': {
                'left_device': 0,
                'right_device': 1,
                'resolution': [640, 480]
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
                'every_frame': 3  # 每3帧重建一次
            }
        },
        'frontend': {
            'loftr_config': {
                'model_type': 'outdoor'
            }
        },
        'visualization': True,
        'visualization_config': {
            'window_size': [1200, 800],
            'show_trajectory': True,
            'show_pointcloud': True
        },
        'performance_targets': {
            'target_fps': 15  # 降低FPS以确保稳定性
        }
    }
    
    save_dir = "test_3d_demo_output"
    
    try:
        print(f"\n🚀 Initializing system...")
        slam_system = HybridSLAMSystem(config, save_dir=save_dir)
        
        print(f"MonoGS Enabled: {slam_system.use_monogs}")
        print(f"MonoGS Backend: {slam_system.monogs_backend is not None}")
        print(f"Visualizer: {slam_system.visualizer is not None}")
        
        if slam_system.use_monogs:
            print("🎯 MonoGS 3D reconstruction will be active!")
        else:
            print("⚠️  Using fallback stereo reconstruction")
        
        print(f"\n📹 Starting camera capture...")
        print("👀 Watch the visualization window for 3D reconstruction")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        # 运行系统
        slam_system.run()
        
    except KeyboardInterrupt:
        print("\n✋ Stopping system...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查结果
    print("\n" + "=" * 50)
    print("📊 Results Summary")
    print("=" * 50)
    
    try:
        # 轨迹信息
        trajectory_len = len(slam_system.trajectory)
        print(f"📍 Trajectory points: {trajectory_len}")
        
        # 3D重建信息
        reconstruction = slam_system.get_3d_reconstruction()
        if reconstruction:
            recon_type = reconstruction.get('type', 'unknown')
            points = reconstruction.get('points')
            colors = reconstruction.get('colors')
            
            print(f"🎨 3D Reconstruction Type: {recon_type}")
            
            if points is not None:
                print(f"🔺 3D Points: {len(points)}")
                if colors is not None:
                    print(f"🌈 With Colors: {len(colors)}")
                
                # 保存3D重建结果
                if slam_system.visualizer and hasattr(slam_system.visualizer, 'save_3d_reconstruction'):
                    output_path = Path(save_dir) / "reconstruction.ply"
                    slam_system.visualizer.save_3d_reconstruction(str(output_path))
                    print(f"💾 3D model saved: {output_path}")
                
            else:
                print("⚠️  No 3D points generated")
        else:
            print("❌ No 3D reconstruction data")
            
        print(f"📁 All results in: {save_dir}")
        
        return reconstruction is not None
        
    except Exception as e:
        print(f"⚠️  Could not analyze results: {e}")
        return False

def main():
    success = test_3d_reconstruction()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 3D Reconstruction Test PASSED")
        print("🎉 Your system is working with MonoGS!")
    else:
        print("❌ 3D Reconstruction Test FAILED")
        print("🔧 Check camera connections and MonoGS setup")
    print("=" * 50)

if __name__ == "__main__":
    main()