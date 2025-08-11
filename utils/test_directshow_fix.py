#!/usr/bin/env python3
"""
专门测试DirectShow后端的摄像头修复方案
"""

import cv2
import time

def test_directshow_cameras():
    """专门测试DirectShow后端"""
    print("=== 使用DirectShow后端测试双摄像头 ===")
    
    # 强制使用DirectShow后端
    backend_id = cv2.CAP_DSHOW
    
    cameras = []
    
    # 初始化摄像头
    for idx in [0, 1]:
        print(f"正在初始化摄像头 {idx} (DirectShow)...")
        cap = cv2.VideoCapture(idx, backend_id)
        
        if cap.isOpened():
            # 设置基本参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区
            
            # 测试读取第一帧
            ret, frame = cap.read()
            if ret:
                print(f"  摄像头 {idx}: [OK] 成功初始化 (帧形状: {frame.shape})")
                cameras.append(cap)
            else:
                print(f"  摄像头 {idx}: [FAIL] 无法读取帧")
                cap.release()
        else:
            print(f"  摄像头 {idx}: [FAIL] 无法打开")
            cap.release()
    
    if len(cameras) != 2:
        print(f"错误: 只初始化了 {len(cameras)} 个摄像头")
        return False
    
    # 测试连续读取
    print("\n开始连续读取测试...")
    successful_reads = 0
    failed_reads = 0
    test_frames = 20  # 测试20帧
    
    for i in range(test_frames):
        frames = []
        all_success = True
        
        # 依次读取每个摄像头
        for cam_idx, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append(frame)
                if i == 0:  # 第一次成功时显示信息
                    print(f"  摄像头 {cam_idx}: [OK] 读取成功 (形状: {frame.shape})")
            else:
                all_success = False
                if i == 0:
                    print(f"  摄像头 {cam_idx}: [FAIL] 读取失败")
                break
        
        if all_success and len(frames) == 2:
            successful_reads += 1
        else:
            failed_reads += 1
        
        # 小延迟以避免资源竞争
        time.sleep(0.05)
    
    # 清理资源
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    
    # 结果统计
    success_rate = (successful_reads / test_frames) * 100
    print(f"\n=== 测试结果 ===")
    print(f"成功读取: {successful_reads}/{test_frames} 帧")
    print(f"失败次数: {failed_reads}")
    print(f"成功率: {success_rate:.1f}%")
    
    if successful_reads >= test_frames * 0.8:  # 80%成功率
        print("[SUCCESS] DirectShow后端可以稳定工作！")
        
        # 生成修复配置
        print("\n=== 修复方案 ===")
        print("在Qt应用中使用以下配置:")
        print(f"backend_id = cv2.CAP_DSHOW  # {cv2.CAP_DSHOW}")
        print("cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)")
        print("cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)")
        
        return True
    else:
        print("[FAIL] DirectShow后端仍然不稳定")
        return False

def main():
    print("DirectShow后端专项测试")
    print("=" * 40)
    
    # 显示OpenCV信息
    print(f"OpenCV版本: {cv2.__version__}")
    
    success = test_directshow_cameras()
    
    if not success:
        print("\n建议的解决方案:")
        print("1. 关闭可能占用摄像头的其他应用程序")
        print("2. 重新插拔USB摄像头")
        print("3. 重启计算机")
        print("4. 使用模拟数据: --mock")
    
    return success

if __name__ == "__main__":
    main()