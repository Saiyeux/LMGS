#!/usr/bin/env python3
"""
Hybrid SLAM 主运行脚本
EfficientLoFTR + OpenCV PnP + MonoGS 融合系统
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hybrid SLAM System')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('--eval', action='store_true',
                       help='Run in evaluation mode')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Results save directory')
    parser.add_argument('--no-gui', action='store_true',
                       help='Disable GUI')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("Hybrid SLAM System")
    print("EfficientLoFTR + OpenCV PnP + MonoGS Integration")
    print("Dual Camera Real-time 3D Reconstruction")
    print("="*60)
    
    try:
        print(f"Loading config from: {args.config}")
        
        # 导入HybridSLAMSystem
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        
        # 创建并运行SLAM系统
        slam = HybridSLAMSystem.from_config_file(args.config, args.save_dir)
        
        print(f"System initialized, starting SLAM...")
        print(f"Device: {args.device}")
        print(f"Save directory: {slam.save_dir}")
        print(f"Visualization: {'Enabled' if not args.no_gui else 'Disabled'}")
        
        # 运行系统
        slam.run()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Please make sure the config file path is correct.")
        return 1
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("SLAM completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())