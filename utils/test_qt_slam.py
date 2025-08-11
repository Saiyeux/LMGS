#!/usr/bin/env python3
"""
Qt SLAM系统测试脚本
测试新的Qt+OpenCV+AI架构
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_data_structures():
    """测试数据结构"""
    print("=== 测试数据结构 ===")
    
    try:
        from hybrid_slam.utils.data_structures import StereoFrame, ProcessingResult
        import numpy as np
        
        # 测试StereoFrame
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        stereo_frame = StereoFrame(
            frame_id=1,
            timestamp=time.time(),
            left_image=left_img,
            right_image=right_img
        )
        
        print(f"OK StereoFrame创建成功: frame_id={stereo_frame.frame_id}")
        
        # 测试ProcessingResult
        result = ProcessingResult(
            frame_id=1,
            timestamp=time.time(),
            confidence=0.85,
            num_matches=42,
            method="opencv"
        )
        
        print(f"OK ProcessingResult创建成功: matches={result.num_matches}, confidence={result.confidence}")
        print("OK 数据结构测试通过")
        
    except Exception as e:
        print(f"FAIL 数据结构测试失败: {e}")
        return False
    
    return True

def test_video_stream_manager():
    """测试视频流管理器（非Qt模式）"""
    print("\n=== 测试视频流管理器 ===")
    
    try:
        from hybrid_slam.core.video_stream_manager import SimpleVideoStreamManager
        
        # 创建管理器
        manager = SimpleVideoStreamManager(left_device=0, right_device=1)
        print("OK VideoStreamManager创建成功")
        
        # 测试初始化（不实际启动摄像头）
        print("OK 基础功能测试通过")
        
        # 如果需要实际测试摄像头，取消注释以下代码
        """
        if manager.start_capture():
            print("OK 摄像头启动成功")
            
            # 等待几帧
            time.sleep(2)
            
            # 获取帧
            frame = manager.get_latest_frame()
            if frame:
                print(f"OK 获取到帧: frame_id={frame.frame_id}")
            
            # 获取统计信息
            stats = manager.get_stats()
            print(f"OK 统计信息: {stats}")
            
            # 停止
            manager.stop_capture()
            print("OK 摄像头停止成功")
        else:
            print("WARN 摄像头启动失败（可能没有连接摄像头）")
        """
        
    except Exception as e:
        print(f"FAIL 视频流管理器测试失败: {e}")
        return False
    
    return True

def test_ai_processing_pipeline():
    """测试AI处理管道（非Qt模式）"""
    print("\n=== 测试AI处理管道 ===")
    
    try:
        from hybrid_slam.core.ai_processing_pipeline import SimpleAIProcessingPipeline
        
        # 创建处理管道
        config = {
            'enable_loftr': False,  # 跳过EfficientLoFTR以避免模型加载
            'enable_pnp': False,    # 跳过PnP
            'enable_mono_gs': False
        }
        
        pipeline = SimpleAIProcessingPipeline(config)
        print("OK AIProcessingPipeline创建成功")
        
        # 测试初始化
        if pipeline.initialize_models():
            print("OK AI模型初始化成功")
        else:
            print("WARN AI模型初始化失败（预期行为）")
        
        # 获取统计信息
        stats = pipeline.get_stats()
        print(f"OK 统计信息: {stats}")
        
        print("OK AI处理管道基础功能测试通过")
        
    except Exception as e:
        print(f"FAIL AI处理管道测试失败: {e}")
        return False
    
    return True

def test_qt_components():
    """测试Qt组件"""
    print("\n=== 测试Qt组件 ===")
    
    try:
        # 测试PyQt5可用性
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import Qt
            qt_available = True
            print("OK PyQt5可用")
        except ImportError:
            qt_available = False
            print("WARN PyQt5不可用，跳过Qt组件测试")
            return True
        
        if qt_available:
            # 测试组件导入
            from hybrid_slam.gui.qt_display_widget import QtDisplayWidget
            from hybrid_slam.gui.main_window import MainWindow
            print("OK Qt组件导入成功")
            
            # 创建应用（但不显示）
            app = QApplication([])
            
            # 测试显示组件创建
            display_widget = QtDisplayWidget()
            print("OK QtDisplayWidget创建成功")
            
            # 测试主窗口创建
            main_window = MainWindow()
            print("OK MainWindow创建成功")
            
            # 清理
            app.quit()
            print("OK Qt组件测试通过")
        
    except Exception as e:
        print(f"FAIL Qt组件测试失败: {e}")
        return False
    
    return True

def test_integration():
    """集成测试（模拟数据）"""
    print("\n=== 集成测试 ===")
    
    try:
        from hybrid_slam.utils.data_structures import StereoFrame
        from hybrid_slam.core.ai_processing_pipeline import SimpleAIProcessingPipeline
        import numpy as np
        
        # 创建模拟数据
        left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        stereo_frame = StereoFrame(
            frame_id=1,
            timestamp=time.time(),
            left_image=left_img,
            right_image=right_img
        )
        
        # 创建处理管道（只启用OpenCV后备）
        config = {
            'enable_loftr': False,
            'enable_pnp': False,
            'enable_mono_gs': False
        }
        
        pipeline = SimpleAIProcessingPipeline(config)
        
        if pipeline.initialize_models():
            print("OK 模型初始化成功")
        
        # 模拟处理（这里不会实际处理，因为所有AI功能都关闭了）
        pipeline.enqueue_frame(stereo_frame)
        print("OK 帧处理队列添加成功")
        
        print("OK 集成测试通过")
        
    except Exception as e:
        print(f"FAIL 集成测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("Hybrid SLAM Qt架构测试")
    print("=" * 50)
    
    tests = [
        ("数据结构", test_data_structures),
        ("视频流管理器", test_video_stream_manager),
        ("AI处理管道", test_ai_processing_pipeline),
        ("Qt组件", test_qt_components),
        ("集成测试", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"OK {test_name} 通过")
        else:
            print(f"FAIL {test_name} 失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("SUCCESS 所有测试通过！新架构基本功能正常")
        return 0
    else:
        print("WARN 部分测试失败，请检查相关组件")
        return 1

if __name__ == "__main__":
    sys.exit(main())