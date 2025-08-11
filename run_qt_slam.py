#!/usr/bin/env python3
"""
Qt SLAM系统启动脚本
启动新的Qt+OpenCV+AI架构的Hybrid SLAM系统
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def check_requirements():
    """检查依赖项"""
    print("正在检查系统依赖...")
    
    missing_deps = []
    
    # 检查PyQt5
    try:
        import PyQt5
        print("OK PyQt5 可用")
    except ImportError:
        missing_deps.append("PyQt5")
    
    # 检查OpenCV
    try:
        import cv2
        print(f"OK OpenCV 可用 (版本: {cv2.__version__})")
    except ImportError:
        missing_deps.append("opencv-python")
    
    # 检查NumPy
    try:
        import numpy
        print(f"OK NumPy 可用 (版本: {numpy.__version__})")
    except ImportError:
        missing_deps.append("numpy")
    
    # 检查可选依赖
    try:
        import torch
        print(f"OK PyTorch 可用 (版本: {torch.__version__})")
    except ImportError:
        print("WARN PyTorch 不可用，某些AI功能可能无法使用")
    
    if missing_deps:
        print("\nERROR 缺少必要依赖:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\n请安装: pip install {' '.join(missing_deps)}")
        return False
    
    print("OK 所有必要依赖都可用")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qt SLAM系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_qt_slam.py                    # 启动GUI界面
  python run_qt_slam.py --test            # 运行测试
  python run_qt_slam.py --check-deps      # 检查依赖
  python run_qt_slam.py --left-cam 0 --right-cam 1  # 指定摄像头设备
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='运行系统测试而不启动GUI'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='检查系统依赖'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='运行快速系统测试'
    )
    
    parser.add_argument(
        '--left-cam',
        type=int,
        default=0,
        help='左摄像头设备ID (默认: 0)'
    )
    
    parser.add_argument(
        '--right-cam',
        type=int,
        default=1,
        help='右摄像头设备ID (默认: 1)'
    )
    
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='禁用AI功能，仅显示视频流'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        choices=range(10, 61),
        metavar="[10-60]",
        help='目标FPS (默认: 30)'
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps or not check_requirements():
        return 1 if not check_requirements() else 0
    
    # 运行测试
    if args.test:
        print("运行系统测试...")
        try:
            from test_qt_slam import main as test_main
            return test_main()
        except ImportError as e:
            print(f"无法导入测试模块: {e}")
            return 1
    
    # 运行快速测试
    if args.quick_test:
        print("运行快速测试...")
        try:
            from quick_test import main as quick_test_main
            return quick_test_main()
        except ImportError as e:
            print(f"无法导入快速测试模块: {e}")
            return 1
    
    # 启动GUI系统
    print("启动 Hybrid SLAM Qt系统...")
    print(f"配置: 左摄像头={args.left_cam}, 右摄像头={args.right_cam}, FPS={args.fps}")
    
    if args.no_ai:
        print("WARN AI功能已禁用")
    
    try:
        from hybrid_slam.gui.main_window import main as gui_main
        
        # 可以在这里预设配置
        import json
        config_override = {
            'left_device': args.left_cam,
            'right_device': args.right_cam,
            'target_fps': args.fps,
            'enable_loftr': not args.no_ai,
            'enable_pnp': not args.no_ai,
            'enable_mono_gs': True   # 启用MonoGS 3D重建
        }
        
        # 保存配置覆盖（如果需要）
        config_file = Path("config/qt_slam_config.json")
        config_file.parent.mkdir(exist_ok=True)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
                existing_config.update(config_override)
        else:
            existing_config = config_override
        
        with open(config_file, 'w') as f:
            json.dump(existing_config, f, indent=2)
        
        print("配置已更新，启动GUI...")
        
        # 启动GUI
        return gui_main()
        
    except ImportError as e:
        print(f"无法启动GUI: {e}")
        print("请确保已安装 PyQt5: pip install PyQt5")
        return 1
    except KeyboardInterrupt:
        print("\n用户中断，退出系统")
        return 0
    except Exception as e:
        print(f"系统启动失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())