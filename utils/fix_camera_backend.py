#!/usr/bin/env python3
"""
修复摄像头后端问题
Windows MSMF驱动问题的解决方案
"""

import cv2
import time
import os

def test_different_backends():
    """测试不同的OpenCV后端"""
    print("=== 测试不同的摄像头后端 ===")
    
    # 可用的后端列表
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation (MSMF)"),
        (cv2.CAP_V4L2, "Video4Linux2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    working_backend = None
    
    for backend_id, backend_name in backends:
        print(f"\n--- 测试 {backend_name} 后端 ---")
        try:
            # 测试摄像头0
            cap0 = cv2.VideoCapture(0, backend_id)
            if cap0.isOpened():
                ret0, frame0 = cap0.read()
                cap0.release()
                
                if ret0:
                    print(f"  摄像头0: [OK] 成功 (帧形状: {frame0.shape})")
                    
                    # 测试摄像头1
                    cap1 = cv2.VideoCapture(1, backend_id)
                    if cap1.isOpened():
                        ret1, frame1 = cap1.read()
                        cap1.release()
                        
                        if ret1:
                            print(f"  摄像头1: [OK] 成功 (帧形状: {frame1.shape})")
                            print(f"  [SUCCESS] {backend_name} 后端可以正常工作！")
                            working_backend = (backend_id, backend_name)
                        else:
                            print(f"  摄像头1: [FAIL] 无法读取帧")
                    else:
                        print(f"  摄像头1: [FAIL] 无法打开")
                else:
                    print(f"  摄像头0: [FAIL] 无法读取帧")
            else:
                print(f"  摄像头0: [FAIL] 无法打开")
                
        except Exception as e:
            print(f"  错误: {e}")
    
    return working_backend

def test_sequential_with_backend(backend_id, backend_name):
    """使用指定后端测试顺序读取"""
    print(f"\n=== 使用 {backend_name} 后端测试双摄像头 ===")
    
    cameras = []
    valid_cameras = []
    
    # 初始化摄像头
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, backend_id)
        if cap.isOpened():
            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            cameras.append(cap)
            valid_cameras.append(idx)
            print(f"成功打开摄像头 {idx}")
        else:
            print(f"无法打开摄像头 {idx}")
            cap.release()
    
    if len(cameras) != 2:
        print(f"错误: 只打开了 {len(cameras)} 个摄像头")
        return False
    
    # 测试连续读取
    successful_reads = 0
    failed_reads = 0
    test_duration = 3.0
    start_time = time.time()
    
    print("开始顺序读取测试...")
    
    try:
        while time.time() - start_time < test_duration:
            frames = []
            all_success = True
            
            # 顺序读取所有摄像头
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    all_success = False
                    break
            
            if all_success and len(frames) == 2:
                successful_reads += 1
                if successful_reads == 1:
                    print(f"首次成功! 帧形状: {[f.shape for f in frames]}")
            else:
                failed_reads += 1
                time.sleep(0.01)
    
    finally:
        # 释放摄像头
        for i, cap in enumerate(cameras):
            cap.release()
        cv2.destroyAllWindows()
    
    # 结果
    elapsed = time.time() - start_time
    fps = successful_reads / elapsed if elapsed > 0 else 0
    
    print(f"\n{backend_name} 测试结果:")
    print(f"  成功读取: {successful_reads} 组帧")
    print(f"  失败读取: {failed_reads} 次") 
    print(f"  平均帧率: {fps:.1f} FPS")
    print(f"  成功率: {successful_reads/(successful_reads+failed_reads)*100:.1f}%")
    
    success = successful_reads > 5 and fps > 2.0
    print(f"  状态: {'[SUCCESS]' if success else '[FAILED]'}")
    
    return success

def create_camera_config(backend_id, backend_name):
    """创建推荐的摄像头配置"""
    config = f"""
# 推荐的摄像头配置
# 使用 {backend_name} 后端 (backend_id = {backend_id})

input:
  source: camera
  camera:
    left_device: 0
    right_device: 1
    resolution: [640, 480]
    fps: 20
    backend: {backend_id}  # {backend_name}
    buffer_size: 1
    
# OpenCV后端说明:
# cv2.CAP_DSHOW = 700     # DirectShow (Windows推荐)
# cv2.CAP_MSMF = 1400     # Media Foundation (可能有问题)
# cv2.CAP_ANY = 0         # 自动选择
"""
    
    with open("camera_backend_config.txt", "w", encoding="utf-8") as f:
        f.write(config)
    
    print(f"配置已保存到: camera_backend_config.txt")

def main():
    """主函数"""
    print("OpenCV 摄像头后端诊断工具")
    print("=" * 50)
    print("正在诊断 Windows MSMF 驱动问题...")
    
    # 显示OpenCV信息
    print(f"\nOpenCV版本: {cv2.__version__}")
    print(f"可用后端: {cv2.videoio_registry.getCameraBackends()}")
    
    # 测试不同后端
    working_backend = test_different_backends()
    
    if working_backend:
        backend_id, backend_name = working_backend
        print(f"\n[FOUND] 找到可工作的后端: {backend_name}")
        
        # 进行详细测试
        if test_sequential_with_backend(backend_id, backend_name):
            print(f"\n[SUCCESS] 解决方案找到!")
            print(f"使用 {backend_name} 后端可以正常工作")
            
            # 创建配置文件
            create_camera_config(backend_id, backend_name)
            
            print(f"\n[NEXT] 下一步:")
            print(f"1. 修改 hybrid_slam 使用 backend_id = {backend_id}")
            print(f"2. 在 VideoCapture 初始化时指定后端:")
            print(f"   cap = cv2.VideoCapture(device_id, {backend_id})")
            print(f"3. 测试修复后的系统")
            
        else:
            print(f"\n[WARN] {backend_name} 后端仍有问题，继续寻找解决方案...")
    
    else:
        print(f"\n[FAIL] 没有找到可工作的后端")
        print(f"建议:")
        print(f"1. 更新摄像头驱动程序")
        print(f"2. 重启计算机") 
        print(f"3. 尝试使用 DirectShow 后端")
        print(f"4. 使用模拟数据: python run_hybrid_slam.py --mock")
    
    return working_backend is not None

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)