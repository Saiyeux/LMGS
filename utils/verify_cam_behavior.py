#!/usr/bin/env python3
"""
验证cam.py的真实行为
检查它是否真的同时使用两个摄像头
"""

import cv2
import time

def monitor_cam_behavior():
    """监控cam.py式的摄像头行为"""
    print("=== 验证cam.py行为 ===")
    
    # 完全模仿cam.py的初始化
    camera_indices = [0, 1]
    cameras = []
    valid_cameras = []
    
    print("初始化摄像头（模仿cam.py）...")
    
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # 设置1080p分辨率（cam.py的设置）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cameras.append(cap)
            valid_cameras.append(idx)
            print(f"成功打开摄像头 {idx}")
            
            # 检查实际分辨率
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  实际分辨率: {actual_width}x{actual_height}")
        else:
            print(f"无法打开摄像头 {idx}")
            cap.release()
    
    if not cameras:
        print("没有可用的摄像头")
        return False
    
    print(f"总共打开了 {len(cameras)} 个摄像头: {valid_cameras}")
    
    # 测试cam.py的读取循环
    frame_count = 0
    success_count = 0
    fail_count = 0
    
    print("开始cam.py式读取测试（10秒）...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10.0:
            frames = []
            all_success = True
            
            # 从所有摄像头读取帧（cam.py的方式）
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"无法读取摄像头 {valid_cameras[i]} 的帧")
                    all_success = False
                    break
            
            # 如果所有摄像头都成功读取帧
            if all_success and frames:
                success_count += 1
                if success_count == 1:
                    print(f"首次成功！帧形状: {[f.shape for f in frames]}")
                elif success_count % 50 == 0:
                    print(f"已成功读取 {success_count} 组帧")
                
                # 每隔100帧显示一次（模仿cam.py，但不实际显示）
                if success_count % 100 == 0:
                    print(f"模拟显示帧 #{success_count}")
            else:
                fail_count += 1
                # 不要在每次失败时都打印，避免spam
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("用户中断")
    
    finally:
        # 释放所有摄像头资源（cam.py方式）
        for i, cap in enumerate(cameras):
            cap.release()
            print(f"释放摄像头 {valid_cameras[i]}")
        cv2.destroyAllWindows()
    
    # 结果分析
    elapsed = time.time() - start_time
    fps = success_count / elapsed if elapsed > 0 else 0
    
    print(f"\n=== 测试结果 ===")
    print(f"运行时间: {elapsed:.1f} 秒")
    print(f"总帧循环: {frame_count}")
    print(f"成功组帧: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"成功帧率: {fps:.1f} FPS")
    print(f"成功率: {success_count/frame_count*100:.1f}%")
    
    # 判断是否真的像cam.py一样工作
    if success_count > 50 and fps > 5.0:
        print("\n✅ 确认：可以模仿cam.py的行为")
        return True
    else:
        print("\n❌ 无法重现cam.py的成功行为")
        return False

def test_single_camera_detailed():
    """详细测试单个摄像头"""
    print("\n=== 单摄像头详细测试 ===")
    
    for cam_id in [0, 1]:
        print(f"\n--- 测试摄像头 {cam_id} ---")
        cap = cv2.VideoCapture(cam_id)
        
        if cap.isOpened():
            # 获取摄像头属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            backend = int(cap.get(cv2.CAP_PROP_BACKEND))
            
            print(f"  分辨率: {width}x{height}")
            print(f"  帧率: {fps}")
            print(f"  编码: {fourcc}")
            print(f"  后端: {backend}")
            
            # 测试连续读取
            success = 0
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    success += 1
                time.sleep(0.1)
            
            print(f"  读取成功率: {success}/10")
            cap.release()
        else:
            print(f"  无法打开摄像头 {cam_id}")

if __name__ == "__main__":
    print("摄像头行为验证工具")
    print("=" * 40)
    
    # 首先进行详细的单摄像头测试
    test_single_camera_detailed()
    
    # 然后验证cam.py行为
    success = monitor_cam_behavior()
    
    if success:
        print("\n🎉 找到了cam.py成功的原因！")
        print("建议：将这个工作模式集成到hybrid_slam中")
    else:
        print("\n🔍 cam.py的成功可能依赖于特殊条件")
        print("建议：使用模拟数据进行系统测试")