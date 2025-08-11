#!/usr/bin/env python3
"""
Hybrid SLAM 启动脚本
支持双摄像头实时重建

使用方法:
python run_hybrid_slam.py --config configs/stereo_camera_config.yaml
python run_hybrid_slam.py --mock  # 使用模拟数据测试
python run_hybrid_slam.py --help  # 显示帮助信息
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid SLAM - 双摄像头实时重建系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用实际双摄像头
  python run_hybrid_slam.py --config configs/stereo_camera_config.yaml
  
  # 使用模拟数据测试
  python run_hybrid_slam.py --mock
  
  # 自定义保存目录
  python run_hybrid_slam.py --config configs/stereo_camera_config.yaml --save-dir results/test1
  
  # 关闭可视化（提高性能）
  python run_hybrid_slam.py --config configs/stereo_camera_config.yaml --no-vis
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/stereo_camera_config.yaml',
        help='配置文件路径 (默认: configs/stereo_camera_config.yaml)'
    )
    
    parser.add_argument(
        '--mock', '-m',
        action='store_true',
        help='使用模拟数据进行测试'
    )
    
    parser.add_argument(
        '--save-dir', '-s',
        type=str,
        default=None,
        help='结果保存目录'
    )
    
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='关闭可视化界面'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='计算设备 (cpu/cuda)'
    )
    
    return parser.parse_args()

def create_mock_config():
    """创建模拟数据配置"""
    return {
        'device': 'cpu',
        'input': {
            'source': 'mock',
            'camera': {
                'resolution': [640, 480],
                'fps': 20
            }
        },
        'frontend': {
            'matcher_type': 'loftr',
            'tracking_method': 'pnp'
        },
        'EfficientLoFTR': {
            'model_path': 'thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt',
            'resize_to': [640, 480],
            'match_threshold': 0.2,
            'max_keypoints': 2048
        },
        'performance_targets': {
            'target_fps': 20,
            'max_memory_gb': 8,
            'max_gpu_memory_gb': 6
        },
        'visualization': True,
        'visualization_config': {
            'save_mode': True,
            'save_dir': 'mock_visualization',
            'save_frequency': 3,
            'window_size': [1200, 600],
            'show_trajectory': True
        },
        'output': {
            'save_trajectory': True,
            'save_keyframes': False,
            'formats': {
                'trajectory': ['tum', 'kitti']
            }
        }
    }

def check_system_requirements():
    """检查系统要求"""
    print("检查系统要求...")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("  GPU: 未检测到CUDA设备，将使用CPU")
    except:
        print("  GPU: PyTorch未安装或不支持CUDA")
    
    # 检查摄像头（如果使用实际摄像头）
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
    except:
        print("  ERROR: OpenCV未安装")
        return False
    
    print("系统检查完成")
    return True

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Hybrid SLAM - 双摄像头实时重建系统")
    print("=" * 60)
    
    # 检查系统要求
    if not check_system_requirements():
        print("错误: 系统要求不满足")
        return 1
    
    try:
        # 导入核心系统
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        from hybrid_slam.utils.config_manager import ConfigManager
        
        # 加载配置
        if args.mock:
            print("\n使用模拟数据模式")
            config = create_mock_config()
        else:
            if not Path(args.config).exists():
                print(f"错误: 配置文件不存在: {args.config}")
                return 1
            
            print(f"\n加载配置文件: {args.config}")
            config = ConfigManager.load_config(args.config)
        
        # 应用命令行参数覆盖
        if args.device:
            config['device'] = args.device
            print(f"设备设置为: {args.device}")
        
        if args.no_vis:
            config['visualization'] = False
            print("可视化已关闭")
        
        # 显示主要配置
        print("\n主要配置:")
        print(f"  输入源: {config.get('input', {}).get('source', 'unknown')}")
        print(f"  计算设备: {config.get('device', 'unknown')}")
        print(f"  目标帧率: {config.get('performance_targets', {}).get('target_fps', 'unknown')}")
        print(f"  可视化: {config.get('visualization', False)}")
        
        # 创建保存目录
        if args.save_dir:
            save_dir = args.save_dir
        else:
            save_dir = "hybrid_slam_results"
        
        # 创建并运行SLAM系统
        print(f"\n初始化SLAM系统，保存至: {save_dir}")
        slam_system = HybridSLAMSystem(config, save_dir=save_dir)
        
        print("\n" + "=" * 60)
        print("开始SLAM处理...")
        print("按 Ctrl+C 停止系统")
        print("=" * 60)
        
        # 运行系统
        slam_system.run()
        
    except KeyboardInterrupt:
        print("\n\n用户中断，正在关闭系统...")
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n系统正常退出")
    return 0

if __name__ == "__main__":
    exit(main())