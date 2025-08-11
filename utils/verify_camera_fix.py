#!/usr/bin/env python3
"""
验证摄像头修复是否成功
快速测试VideoStreamManager是否可以正常工作
"""

import sys
import time
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from hybrid_slam.core.video_stream_manager import VideoStreamManager
    from hybrid_slam.utils.data_structures import StereoFrame
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_video_stream_manager():
    """测试VideoStreamManager的功能"""
    
    # 读取检测到的摄像头配置
    config_file = Path("camera_fallback_config.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        cam_config = config['camera_config']
        left_device = cam_config['left_device']
        right_device = cam_config['right_device']
        
        print(f"使用检测到的摄像头配置: 左={left_device}, 右={right_device}")
    else:
        # 使用默认配置
        left_device = 0
        right_device = 1
        print("使用默认摄像头配置: 左=0, 右=1")
    
    print("=== 测试VideoStreamManager ===")
    
    # 创建视频流管理器
    stream_manager = VideoStreamManager(
        left_device=left_device,
        right_device=right_device,
        target_fps=20,
        buffer_size=5
    )
    
    try:
        print("初始化摄像头...")
        if not stream_manager.initialize_cameras():
            print("[FAIL] 摄像头初始化失败")
            return False
        
        print("[OK] 摄像头初始化成功")
        
        print("开始视频采集...")
        if not stream_manager.start_capture():
            print("[FAIL] 无法开始视频采集")
            return False
        
        print("[OK] 视频采集已启动")
        
        # 等待几秒钟获取一些帧
        print("等待帧数据...")
        time.sleep(3)
        
        # 检查是否能获取到帧
        frames_received = 0
        for i in range(10):
            frame = stream_manager.get_latest_frame()
            if frame:
                frames_received += 1
                if frames_received == 1:
                    print(f"[OK] 收到第一帧: ID={frame.frame_id}")
                    print(f"     左图形状: {frame.left_image.shape}")
                    print(f"     右图形状: {frame.right_image.shape}")
            time.sleep(0.1)
        
        # 获取统计信息
        stats = stream_manager.get_stats()
        print(f"\n=== 统计信息 ===")
        print(f"总帧数: {stats['total_frames']}")
        print(f"丢帧数: {stats['dropped_frames']}")
        print(f"同步错误: {stats['sync_errors']}")
        print(f"FPS: {stats['fps']:.1f}")
        
        # 判断是否成功
        success = frames_received > 0 and stats['total_frames'] > 0
        
        if success:
            print(f"\n[SUCCESS] 视频流管理器工作正常!")
            print(f"          收到 {frames_received} 帧数据")
            print(f"          总计 {stats['total_frames']} 帧")
        else:
            print(f"\n[FAIL] 视频流管理器无法正常获取帧")
            print(f"       收到帧数: {frames_received}")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] 测试过程中发生错误: {e}")
        return False
        
    finally:
        print("停止视频采集...")
        stream_manager.stop_capture()
        print("[OK] 资源已清理")

def main():
    """主函数"""
    print("VideoStreamManager验证测试")
    print("=" * 40)
    
    success = test_video_stream_manager()
    
    if success:
        print("\n🎉 摄像头修复成功!")
        print("   现在可以正常使用Qt SLAM应用")
        print("\n推荐运行:")
        print("   python run_qt_slam.py --no-ai --left-cam 0 --right-cam 1")
    else:
        print("\n❌ 摄像头仍有问题")
        print("   建议使用模拟模式:")
        print("   python run_qt_slam.py --mock")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)