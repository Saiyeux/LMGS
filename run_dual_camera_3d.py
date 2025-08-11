#!/usr/bin/env python3
"""
双目摄像头3D重建系统启动脚本
启用MonoGS进行真正的3D场景重建
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 确保项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_slam.core.slam_system import HybridSLAMSystem
from hybrid_slam.utils.config_manager import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="Dual Camera 3D Reconstruction with MonoGS")
    parser.add_argument("--config", default="configs/dual_camera_monogs.yaml", 
                       help="Configuration file path")
    parser.add_argument("--save-dir", default="dual_camera_3d_results", 
                       help="Results save directory")
    parser.add_argument("--left-cam", type=int, default=0, 
                       help="Left camera device ID")
    parser.add_argument("--right-cam", type=int, default=1, 
                       help="Right camera device ID")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480],
                       help="Camera resolution [width height]")
    parser.add_argument("--enable-3d", action="store_true", default=True,
                       help="Enable MonoGS 3D reconstruction")
    parser.add_argument("--no-vis", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        print("Debug logging enabled")
    
    print("=" * 60)
    print("Dual Camera 3D Reconstruction System")
    print("MonoGS Integration Enabled")
    print("=" * 60)
    
    # 加载配置
    try:
        config = ConfigManager.load_config(args.config)
        print(f"✓ Configuration loaded: {args.config}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1
    
    # 更新相机配置
    if 'input' not in config:
        config['input'] = {}
    if 'camera' not in config['input']:
        config['input']['camera'] = {}
    
    config['input']['camera'].update({
        'left_device': args.left_cam,
        'right_device': args.right_cam,
        'resolution': args.resolution
    })
    
    # 强制启用MonoGS
    if args.enable_3d:
        if 'monogs' not in config:
            config['monogs'] = {}
        
        # 确保MonoGS配置存在
        config['monogs'].update({
            'cam': {
                'H': args.resolution[1],
                'W': args.resolution[0],
                'fx': 525.0,  # 可能需要根据实际相机校准调整
                'fy': 525.0,
                'cx': args.resolution[0] / 2.0,
                'cy': args.resolution[1] / 2.0
            },
            'tracking': {
                'use_gt_camera': False,
                'forward_prop': True,
                'num_kf': 4
            },
            'mapping': {
                'first_mesh': True,
                'new_points': 2000,
                'every_frame': 2,  # 每2帧进行重建，提高性能
                'no_vis_on_first_frame': False
            }
        })
        print("✓ MonoGS 3D reconstruction enabled")
    
    # 禁用可视化（如果指定）
    if args.no_vis:
        config['visualization'] = False
        print("✓ Visualization disabled for better performance")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results will be saved to: {save_dir}")
    
    try:
        # 创建SLAM系统
        print("\nInitializing Hybrid SLAM System...")
        slam_system = HybridSLAMSystem(config, save_dir=str(save_dir))
        
        # 检查MonoGS状态
        print(f"MonoGS Status: {'✓ Enabled' if slam_system.use_monogs else '✗ Disabled'}")
        if slam_system.use_monogs:
            print("  - Backend initialized:", slam_system.monogs_backend is not None)
            print("  - SLAM object:", slam_system.monogs_slam is not None)
        
        # 启动系统
        print("\n" + "=" * 60)
        print("Starting 3D Reconstruction...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        slam_system.run()
        
    except KeyboardInterrupt:
        print("\n✓ Shutting down system...")
    except Exception as e:
        print(f"\n✗ System error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("3D Reconstruction completed!")
    
    # 显示结果统计
    try:
        if slam_system.trajectory:
            print(f"✓ Trajectory points: {len(slam_system.trajectory)}")
        
        # 检查3D重建数据
        reconstruction = slam_system.get_3d_reconstruction()
        if reconstruction:
            points = reconstruction.get('points')
            recon_type = reconstruction.get('type', 'unknown')
            print(f"✓ 3D Reconstruction: {recon_type}")
            if points is not None:
                print(f"  - 3D Points: {len(points)}")
            else:
                print("  - No 3D points generated")
        else:
            print("✗ No 3D reconstruction data available")
            
        print(f"✓ Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"Warning: Could not display results: {e}")
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())