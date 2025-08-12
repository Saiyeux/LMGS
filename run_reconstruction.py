#!/usr/bin/env python3
"""
LMGS 3D Reconstruction System - Main Entry Point
模块化3D重建系统主启动脚本
"""

import time
import argparse
from pathlib import Path

from lmgs_reconstruction import (
    SmartCameraManager,
    HybridAdvanced3DReconstructor,
    UltimateVisualization,
    dependencies
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LMGS 3D Reconstruction System')
    
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                       help='计算设备 (默认: cuda)')
    parser.add_argument('--max-cameras', type=int, default=5,
                       help='最大搜索相机数量 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录 (默认: output)')
    parser.add_argument('--fps-limit', type=float, default=30.0,
                       help='帧率限制 (默认: 30.0)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='保存间隔帧数 (默认: 100)')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行')
    parser.add_argument('--window-size', nargs=2, type=int, default=[1600, 1000],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='窗口尺寸 (默认: 1600 1000)')
    parser.add_argument('--camera-config', type=str, default='camera_calibration.yaml',
                       help='相机参数配置文件路径 (默认: camera_calibration.yaml)')
    
    return parser.parse_args()


def setup_output_directory(output_dir):
    """设置输出目录"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path


def print_system_info(args, camera_manager, reconstructor):
    """打印系统信息"""
    print("=" * 70)
    print("LMGS 3D重建系统 - 模块化版本")
    print("=" * 70)
    
    # 系统配置
    print(f"计算设备: {args.device}")
    print(f"最大相机数: {args.max_cameras}")
    print(f"帧率限制: {args.fps_limit} FPS")
    print(f"输出目录: {args.output_dir}")
    print(f"窗口尺寸: {args.window_size[0]}x{args.window_size[1]}")
    print(f"运行模式: {'无头模式' if args.headless else 'GUI模式'}")
    print(f"相机配置: {args.camera_config}")
    
    # 相机状态
    is_mock = camera_manager.use_mock
    is_stereo = camera_manager.is_stereo_mode()
    camera_count = camera_manager.get_camera_count()
    
    print(f"相机模式: {'模拟' if is_mock else '真实'}")
    print(f"重建模式: {'立体' if is_stereo else '单目'}")
    print(f"相机数量: {camera_count}")
    
    # 算法状态
    components = []
    if hasattr(reconstructor, 'loftr_processor') and reconstructor.loftr_processor:
        components.append("EfficientLoFTR")
    components.append("传统OpenCV")
    
    print(f"算法组件: {', '.join(components)}")
    print()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 检查依赖项
    print("检查系统依赖项...")
    dependencies.check_dependencies()
    print()
    
    # 设置输出目录
    output_path = setup_output_directory(args.output_dir)
    
    # 初始化系统组件
    print("初始化系统组件...")
    camera_manager = SmartCameraManager(max_cameras=args.max_cameras)
    reconstructor = HybridAdvanced3DReconstructor(device=args.device, camera_config_path=args.camera_config)
    
    # 初始化可视化系统
    if args.headless:
        # 简化的无头模式可视化
        visualizer = None
        print("无头模式：可视化已禁用")
    else:
        visualizer = UltimateVisualization(
            canvas_size=(args.window_size[0], args.window_size[1])
        )
    
    # 初始化相机
    if not camera_manager.initialize():
        print("系统初始化失败！")
        return 1
    
    # 打印系统信息
    print_system_info(args, camera_manager, reconstructor)
    
    # 开始重建循环
    return run_reconstruction_loop(
        camera_manager, reconstructor, visualizer, 
        args, output_path
    )


def run_reconstruction_loop(camera_manager, reconstructor, visualizer, args, output_path):
    """运行重建循环"""
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    last_save = 0
    
    is_mock = camera_manager.use_mock
    is_stereo = camera_manager.is_stereo_mode()
    
    print("开始3D重建...")
    print("按 'q' 退出系统")
    print()
    
    try:
        while True:
            frame_count += 1
            
            # 获取帧数据
            frames = camera_manager.get_frames()
            
            if frames:
                # 3D重建处理
                reconstructor.process_frames(frames, is_mock)
                
                # 获取重建数据
                reconstruction_data = reconstructor.get_reconstruction_data()
                
                # 更新显示
                if visualizer:
                    if not visualizer.display(frames, reconstruction_data, is_stereo, is_mock):
                        break
                elif args.headless:
                    # 无头模式：定期打印状态
                    if reconstruction_data and frame_count % 60 == 0:
                        print(f"处理帧数: {frame_count}, 3D点数: {reconstruction_data['count']}")
                
                # 定期保存结果
                if frame_count - last_save >= args.save_interval:
                    save_reconstruction_result(reconstructor, output_path, frame_count)
                    last_save = frame_count
                
                # 性能报告
                current_time = time.time()
                if current_time - last_report > 5.0:  # 每5秒报告一次
                    report_performance(frame_count, start_time, current_time, reconstruction_data)
                    last_report = current_time
            
            # 控制帧率
            time.sleep(1.0 / args.fps_limit)
            
    except KeyboardInterrupt:
        print("\n用户中断...")
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理资源
        cleanup_system(camera_manager, visualizer, reconstructor, output_path, frame_count)
    
    return 0


def save_reconstruction_result(reconstructor, output_path, frame_count):
    """保存重建结果"""
    try:
        save_file = output_path / f"reconstruction_frame_{frame_count:06d}.npz"
        if reconstructor.save_reconstruction(save_file):
            print(f"已保存重建结果: {save_file}")
    except Exception as e:
        print(f"保存失败: {e}")


def report_performance(frame_count, start_time, current_time, reconstruction_data):
    """报告性能数据"""
    elapsed = current_time - start_time
    fps = frame_count / elapsed
    points_count = reconstruction_data['count'] if reconstruction_data else 0
    
    print(f"性能报告 - FPS: {fps:.1f}, 处理帧数: {frame_count}, 3D点数: {points_count}")


def cleanup_system(camera_manager, visualizer, reconstructor, output_path, frame_count):
    """清理系统资源"""
    print("\n清理系统资源...")
    
    # 清理相机
    camera_manager.cleanup()
    
    # 清理可视化
    if visualizer:
        visualizer.close()
    
    # 保存最终结果
    reconstruction_data = reconstructor.get_reconstruction_data()
    if reconstruction_data:
        final_save_path = output_path / "final_reconstruction.npz"
        if reconstructor.save_reconstruction(final_save_path):
            print(f"最终3D重建结果已保存: {final_save_path}")
            print(f"总点数: {reconstruction_data['count']}, 处理帧数: {frame_count}")
    
    print("LMGS 3D重建系统已关闭")


if __name__ == "__main__":
    exit(main())