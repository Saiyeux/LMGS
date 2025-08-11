#!/usr/bin/env python3
"""
调试Qt界面匹配问题
"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_complete_pipeline():
    """测试完整的处理管道"""
    try:
        print("=== 测试完整处理管道 ===")
        
        # 导入依赖
        from hybrid_slam.core.ai_processing_pipeline import AIProcessingPipeline
        from hybrid_slam.utils.data_structures import StereoFrame
        
        # 创建AI处理配置
        ai_config = {
            'enable_loftr': True,
            'enable_pnp': True,
            'enable_mono_gs': False,
            'confidence_threshold': 0.3,  # 降低阈值
            'match_threshold': 0.1,
            'use_gpu': False  # 使用CPU避免CUDA问题
        }
        
        print(f"AI配置: {ai_config}")
        
        # 创建处理管道
        pipeline = AIProcessingPipeline(ai_config)
        
        # 初始化模型
        if not pipeline.initialize_models():
            print("ERROR: 模型初始化失败")
            return False
        
        # 启动处理线程
        if not pipeline.start_processing():
            print("ERROR: 处理线程启动失败")
            return False
        
        print("处理管道启动成功")
        
        # 创建测试立体帧
        def create_test_stereo_frame(frame_id):
            # 创建有纹理的测试图像
            left_img = np.zeros((480, 640, 3), dtype=np.uint8)
            right_img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加棋盘格模式
            for i in range(0, 480, 40):
                for j in range(0, 640, 40):
                    if (i//40 + j//40) % 2 == 0:
                        left_img[i:i+40, j:j+40] = [180, 180, 180]
                        right_img[i:i+40, j:j+40] = [180, 180, 180]
                    else:
                        left_img[i:i+40, j:j+40] = [60, 60, 60]
                        right_img[i:i+40, j:j+40] = [60, 60, 60]
            
            # 添加一些特征点
            cv2.circle(left_img, (200, 200), 20, (0, 255, 0), -1)
            cv2.circle(right_img, (180, 200), 20, (0, 255, 0), -1)  # 模拟视差
            
            cv2.circle(left_img, (400, 150), 15, (255, 0, 0), -1)
            cv2.circle(right_img, (385, 150), 15, (255, 0, 0), -1)  # 模拟视差
            
            cv2.circle(left_img, (320, 350), 25, (0, 0, 255), -1)
            cv2.circle(right_img, (295, 350), 25, (0, 0, 255), -1)  # 模拟视差
            
            return StereoFrame(
                frame_id=frame_id,
                timestamp=time.time(),
                left_image=left_img,
                right_image=right_img
            )
        
        # 测试多帧处理
        results = []
        for i in range(3):
            print(f"\n--- 处理第{i+1}帧 ---")
            
            # 创建测试帧
            stereo_frame = create_test_stereo_frame(i+1)
            
            # 加入处理队列
            pipeline.enqueue_frame(stereo_frame)
            
            # 等待处理完成
            time.sleep(2.0)
            
            # 获取统计信息
            stats = pipeline.get_stats()
            print(f"统计信息: {stats}")
            
            results.append(stats)
        
        # 停止处理
        pipeline.stop_processing()
        
        # 分析结果
        print(f"\n=== 处理结果分析 ===")
        for i, stats in enumerate(results):
            print(f"帧{i+1}: 总处理数={stats['total_processed']}, 成功匹配={stats['successful_matches']}, 失败匹配={stats['failed_matches']}")
        
        # 如果成功匹配数大于0，说明处理管道工作正常
        total_successful = sum(stats['successful_matches'] for stats in results)
        print(f"总成功匹配: {total_successful}")
        
        return total_successful > 0
        
    except Exception as e:
        print(f"ERROR: 完整管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_camera_input():
    """测试实际摄像头输入"""
    try:
        print("\n=== 测试摄像头输入 ===")
        
        # 尝试打开摄像头
        cap_left = cv2.VideoCapture(0)
        cap_right = cv2.VideoCapture(1)
        
        if not cap_left.isOpened():
            print("ERROR: 无法打开左摄像头 (设备0)")
            return False
            
        if not cap_right.isOpened():
            print("WARNING: 无法打开右摄像头 (设备1)")
            # 使用左摄像头的不同时刻作为立体对
            cap_right = None
        
        # 设置摄像头参数
        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_left.set(cv2.CAP_PROP_FPS, 30)
        
        if cap_right:
            cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap_right.set(cv2.CAP_PROP_FPS, 30)
        
        # 等待摄像头稳定
        time.sleep(1.0)
        
        # 捕获几帧
        for i in range(5):
            ret_left, left_frame = cap_left.read()
            if not ret_left:
                print(f"ERROR: 无法从左摄像头读取第{i+1}帧")
                continue
            
            if cap_right:
                ret_right, right_frame = cap_right.read()
                if not ret_right:
                    print(f"ERROR: 无法从右摄像头读取第{i+1}帧")
                    continue
            else:
                # 延迟一点读取，模拟右摄像头
                time.sleep(0.1)
                ret_right, right_frame = cap_left.read()
                if not ret_right:
                    print(f"ERROR: 无法获取模拟右帧第{i+1}帧")
                    continue
            
            print(f"成功捕获第{i+1}帧: 左图{left_frame.shape}, 右图{right_frame.shape}")
            
            # 保存样本帧用于调试
            if i == 2:  # 保存中间帧
                cv2.imwrite(f'camera_left_sample.jpg', left_frame)
                cv2.imwrite(f'camera_right_sample.jpg', right_frame)
                print("摄像头样本帧已保存: camera_left_sample.jpg, camera_right_sample.jpg")
                
                # 测试此帧的匹配效果
                from hybrid_slam.core.ai_processing_pipeline import AIProcessingPipeline
                
                pipeline = AIProcessingPipeline({
                    'enable_loftr': True,
                    'confidence_threshold': 0.3,
                    'use_gpu': False
                })
                
                if pipeline.initialize_models():
                    matches, confidence = pipeline._perform_feature_matching(left_frame, right_frame)
                    print(f"摄像头帧匹配结果: {len(matches) if matches else 0}个匹配, 置信度: {confidence:.3f}")
                    
                    if matches and len(matches) > 0:
                        # 创建可视化
                        combined = cv2.hconcat([left_frame, right_frame])
                        
                        # 绘制前20个匹配点
                        for i, (pt1, pt2) in enumerate(matches[:20]):
                            if len(pt1) >= 2 and len(pt2) >= 2:
                                pt1_int = (int(pt1[0]), int(pt1[1]))
                                pt2_int = (int(pt2[0] + 640), int(pt2[1]))
                                
                                cv2.circle(combined, pt1_int, 3, (0, 255, 0), -1)
                                cv2.circle(combined, pt2_int, 3, (0, 255, 0), -1)
                                cv2.line(combined, pt1_int, pt2_int, (0, 255, 255), 1)
                        
                        cv2.imwrite('camera_matches.jpg', combined)
                        print("摄像头匹配可视化已保存: camera_matches.jpg")
                else:
                    print("ERROR: 无法初始化处理管道测试摄像头帧")
        
        # 释放摄像头
        cap_left.release()
        if cap_right:
            cap_right.release()
        
        print("摄像头测试完成")
        return True
        
    except Exception as e:
        print(f"ERROR: 摄像头测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始Qt界面匹配调试...")
    
    # 测试完整处理管道
    pipeline_ok = test_complete_pipeline()
    
    # 测试摄像头输入
    camera_ok = test_camera_input()
    
    print(f"\n=== 调试总结 ===")
    print(f"处理管道: {'OK' if pipeline_ok else 'FAILED'}")
    print(f"摄像头输入: {'OK' if camera_ok else 'FAILED'}")
    
    if pipeline_ok and camera_ok:
        print("处理管道和摄像头都正常，Qt界面问题可能是:")
        print("1. 信号连接问题")
        print("2. 显示组件更新问题")  
        print("3. 线程同步问题")
    elif pipeline_ok:
        print("处理管道正常，但摄像头有问题，检查摄像头设备")
    else:
        print("处理管道有问题，需要进一步调试")