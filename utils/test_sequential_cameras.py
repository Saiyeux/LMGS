#!/usr/bin/env python3
"""
测试摄像头顺序访问策略
基于cam.py的成功经验但简化版本
"""

import cv2
import time

def test_cameras_sequential():
    """测试顺序摄像头访问"""
    print("=== 简化版双摄像头顺序测试 ===")
    
    # 摄像头索引列表（模仿cam.py）
    camera_indices = [0, 1]
    
    # 初始化摄像头列表
    cameras = []
    valid_cameras = []
    
    print("初始化摄像头...")
    
    # 遍历摄像头索引列表并初始化（完全模仿cam.py）
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # 设置参数（模仿cam.py，但用较小分辨率）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cameras.append(cap)
            valid_cameras.append(idx)
            print(f"成功打开摄像头 {idx}")
        else:
            print(f"无法打开摄像头 {idx}")
            cap.release()
    
    # 检查是否有可用的摄像头
    if not cameras:
        print("没有可用的摄像头")
        return False
    
    print(f"总共打开了 {len(cameras)} 个摄像头: {valid_cameras}")
    
    if len(cameras) < 2:
        print("需要至少2个摄像头")
        return False
    
    # 测试连续读取（模仿cam.py的主循环）
    successful_reads = 0
    failed_reads = 0
    test_duration = 5.0
    start_time = time.time()
    
    print("开始顺序读取测试...")
    
    try:
        while time.time() - start_time < test_duration:
            frames = []
            all_success = True
            
            # 从所有摄像头读取帧（完全模仿cam.py）
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"无法读取摄像头 {valid_cameras[i]} 的帧")
                    all_success = False
                    break
            
            # 如果所有摄像头都成功读取帧（模仿cam.py逻辑）
            if all_success and frames:
                successful_reads += 1
                if successful_reads == 1:
                    print(f"首次成功读取，帧形状: {[f.shape for f in frames]}")
            else:
                failed_reads += 1
                time.sleep(0.01)  # 短暂等待
    
    except KeyboardInterrupt:
        print("程序被用户终止")
    
    finally:
        # 释放所有摄像头资源（模仿cam.py）
        for i, cap in enumerate(cameras):
            cap.release()
            print(f"释放摄像头 {valid_cameras[i]}")
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
    
    # 结果统计
    elapsed = time.time() - start_time
    fps = successful_reads / elapsed if elapsed > 0 else 0
    
    print(f"\n测试结果:")
    print(f"  成功读取: {successful_reads} 组帧")
    print(f"  失败读取: {failed_reads} 次")
    print(f"  平均帧率: {fps:.1f} FPS")
    print(f"  测试时长: {elapsed:.1f} 秒")
    print(f"  成功率: {successful_reads/(successful_reads+failed_reads)*100:.1f}%")
    
    success = successful_reads > 10 and fps > 5.0
    print(f"  最终状态: {'SUCCESS' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    print("简化版摄像头测试")
    print("=" * 50)
    
    result = test_cameras_sequential()
    
    if result:
        print("\n✅ 测试成功！摄像头可以正常顺序读取")
        print("建议配置:")
        print("  左摄像头: 0")
        print("  右摄像头: 1") 
        print("  分辨率: 640x480")
        print("  使用顺序读取策略")
    else:
        print("\n❌ 测试失败！")
        print("建议:")
        print("1. 检查摄像头是否被其他程序占用")
        print("2. 尝试重新插拔摄像头")
        print("3. 重启计算机")
        print("4. 使用模拟数据测试系统：python run_hybrid_slam.py --mock")