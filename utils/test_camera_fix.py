#!/usr/bin/env python3
"""
测试修复后的摄像头功能
"""

import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def signal_handler(sig, frame):
    """信号处理，优雅退出"""
    print("\n接收到中断信号，正在退出...")
    sys.exit(0)

def test_video_manager_basic():
    """测试基础视频管理器功能"""
    print("=== 测试基础VideoStreamManager ===")
    
    try:
        from hybrid_slam.core.video_stream_manager import SimpleVideoStreamManager
        
        print("1. 创建视频管理器...")
        manager = SimpleVideoStreamManager(left_device=0, right_device=1)
        
        print("2. 启动摄像头采集...")
        if not manager.start_capture():
            print("FAIL 摄像头启动失败")
            return False
        
        print("3. 等待几秒钟收集帧...")
        time.sleep(3)
        
        print("4. 检查统计信息...")
        stats = manager.get_stats()
        print(f"   - 总帧数: {stats['total_frames']}")
        print(f"   - 丢失帧数: {stats['dropped_frames']}")
        print(f"   - 当前FPS: {stats['fps']:.1f}")
        
        print("5. 尝试获取最新帧...")
        frame = manager.get_latest_frame()
        if frame:
            print(f"   - 获取到帧ID: {frame.frame_id}")
            print(f"   - 左图形状: {frame.left_image.shape}")
            print(f"   - 右图形状: {frame.right_image.shape}")
        else:
            print("   - 没有可用帧")
        
        print("6. 停止采集...")
        manager.stop_capture()
        
        # 检查是否成功收集到帧
        success = stats['total_frames'] > 0
        print(f"测试结果: {'OK 成功' if success else 'FAIL 失败'}")
        
        return success
        
    except Exception as e:
        print(f"FAIL 测试失败: {e}")
        return False

def test_video_manager_qt():
    """测试Qt版本的视频管理器"""
    print("\n=== 测试Qt VideoStreamManager ===")
    
    try:
        # 检查Qt可用性
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer
            app = QApplication([])
            qt_available = True
        except ImportError:
            print("WARN PyQt5不可用，跳过Qt测试")
            return True
        
        from hybrid_slam.core.video_stream_manager import VideoStreamManager
        
        print("1. 创建Qt视频管理器...")
        manager = VideoStreamManager(left_device=0, right_device=1, target_fps=15)
        
        frame_count = 0
        
        def on_frame_ready(stereo_frame):
            nonlocal frame_count
            frame_count += 1
            if frame_count <= 5:  # 只显示前5帧信息
                print(f"   - 接收到帧 {stereo_frame.frame_id}: {stereo_frame.left_image.shape}")
        
        # 连接信号
        manager.frame_ready.connect(on_frame_ready)
        
        print("2. 启动Qt摄像头采集...")
        if not manager.start_capture():
            print("FAIL Qt摄像头启动失败")
            app.quit()
            return False
        
        # 创建定时器来停止测试
        def stop_test():
            print("3. 停止Qt采集...")
            manager.stop_capture()
            app.quit()
        
        timer = QTimer()
        timer.timeout.connect(stop_test)
        timer.setSingleShot(True)
        timer.start(3000)  # 3秒后停止
        
        print("   等待3秒钟...")
        app.exec_()
        
        success = frame_count > 0
        print(f"Qt测试结果: {'OK 成功' if success else 'FAIL 失败'} (收到 {frame_count} 帧)")
        
        return success
        
    except Exception as e:
        print(f"FAIL Qt测试失败: {e}")
        try:
            app.quit()
        except:
            pass
        return False

def main():
    """主测试函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    print("摄像头修复功能测试")
    print("=" * 40)
    
    # 测试1: 基础功能
    test1_success = test_video_manager_basic()
    
    # 测试2: Qt功能
    test2_success = test_video_manager_qt()
    
    print("\n" + "=" * 40)
    print("测试总结:")
    print(f"基础功能: {'OK 通过' if test1_success else 'FAIL 失败'}")
    print(f"Qt功能: {'OK 通过' if test2_success else 'FAIL 失败'}")
    
    if test1_success and test2_success:
        print("\nSUCCESS 摄像头修复成功！可以正常启动系统了")
        print("建议运行: python run_qt_slam.py --no-ai --left-cam 0 --right-cam 1")
        return 0
    else:
        print("\nWARN 部分测试失败，请检查摄像头连接")
        return 1

if __name__ == "__main__":
    sys.exit(main())