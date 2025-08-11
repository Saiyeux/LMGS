#!/usr/bin/env python3
"""
简单的摄像头后端测试
解决Windows MSMF驱动问题
"""

import cv2
import time

def test_backends():
    """测试不同的OpenCV后端"""
    print("=== Testing Camera Backends ===")
    
    # 后端列表（使用数字ID避免编码问题）
    backends = [
        (700, "DirectShow"),
        (1400, "Media Foundation"), 
        (0, "Auto"),
    ]
    
    working_backends = []
    
    for backend_id, backend_name in backends:
        print(f"\n--- Testing {backend_name} (ID: {backend_id}) ---")
        try:
            # 测试摄像头0
            cap0 = cv2.VideoCapture(0, backend_id)
            if cap0.isOpened():
                ret0, frame0 = cap0.read()
                if ret0:
                    print(f"  Camera 0: SUCCESS - Shape: {frame0.shape}")
                    
                    # 测试摄像头1
                    cap1 = cv2.VideoCapture(1, backend_id)
                    if cap1.isOpened():
                        ret1, frame1 = cap1.read()
                        if ret1:
                            print(f"  Camera 1: SUCCESS - Shape: {frame1.shape}")
                            print(f"  ** {backend_name} WORKS! **")
                            working_backends.append((backend_id, backend_name))
                        else:
                            print(f"  Camera 1: FAILED to read frame")
                        cap1.release()
                    else:
                        print(f"  Camera 1: FAILED to open")
                else:
                    print(f"  Camera 0: FAILED to read frame")
                cap0.release()
            else:
                print(f"  Camera 0: FAILED to open")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return working_backends

def test_dual_camera(backend_id, backend_name):
    """测试双摄像头连续读取"""
    print(f"\n=== Testing Dual Cameras with {backend_name} ===")
    
    # 初始化摄像头
    cap0 = cv2.VideoCapture(0, backend_id)
    cap1 = cv2.VideoCapture(1, backend_id)
    
    if not (cap0.isOpened() and cap1.isOpened()):
        print("FAILED: Cannot open both cameras")
        return False
    
    # 设置参数
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Testing sequential reading for 3 seconds...")
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < 3.0:
            # 顺序读取
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if ret0 and ret1:
                success_count += 1
                if success_count == 1:
                    print(f"First success! Shapes: {frame0.shape}, {frame1.shape}")
            else:
                fail_count += 1
                time.sleep(0.01)
    
    finally:
        cap0.release()
        cap1.release()
    
    elapsed = time.time() - start_time
    fps = success_count / elapsed
    total = success_count + fail_count
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    print(f"Results:")
    print(f"  Success: {success_count} frames")
    print(f"  Failed: {fail_count} times")
    print(f"  FPS: {fps:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    return success_count > 10 and fps > 2.0

def main():
    """主函数"""
    print("OpenCV Camera Backend Diagnostic")
    print("=" * 40)
    print(f"OpenCV Version: {cv2.__version__}")
    
    # 测试所有后端
    working_backends = test_backends()
    
    if working_backends:
        print(f"\nFound {len(working_backends)} working backend(s)!")
        
        # 测试最佳后端
        for backend_id, backend_name in working_backends:
            if test_dual_camera(backend_id, backend_name):
                print(f"\n*** SOLUTION FOUND! ***")
                print(f"Use backend: {backend_name} (ID: {backend_id})")
                print(f"Code to use:")
                print(f"  cap = cv2.VideoCapture(device_id, {backend_id})")
                
                # 保存配置
                with open("working_backend.txt", "w") as f:
                    f.write(f"Backend ID: {backend_id}\n")
                    f.write(f"Backend Name: {backend_name}\n")
                    f.write(f"Code: cv2.VideoCapture(device_id, {backend_id})\n")
                
                return True
        
        print("\nNo backend passed the dual camera test")
        return False
    
    else:
        print("\nNo working backends found")
        print("Suggestions:")
        print("1. Update camera drivers")
        print("2. Restart computer")
        print("3. Use mock data: python run_hybrid_slam.py --mock")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)