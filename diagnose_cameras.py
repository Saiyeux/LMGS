#!/usr/bin/env python3
"""
摄像头诊断工具
检测双摄像头的状态和性能
"""

import cv2
import time
import threading
from queue import Queue, Empty

def test_single_camera(device_id, duration=5):
    """测试单个摄像头"""
    print(f"\n=== 测试摄像头 {device_id} ===")
    
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"ERROR: 无法打开摄像头 {device_id}")
        return False
    
    # 获取摄像头信息
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"摄像头 {device_id} 信息:")
    print(f"  分辨率: {int(width)}x{int(height)}")
    print(f"  帧率: {fps}")
    
    # 测试实际读取性能
    frame_count = 0
    start_time = time.time()
    failed_reads = 0
    
    print(f"开始读取测试 ({duration}秒)...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
        else:
            failed_reads += 1
            time.sleep(0.01)  # 短暂等待
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    
    print(f"测试结果:")
    print(f"  成功读取: {frame_count} 帧")
    print(f"  失败读取: {failed_reads} 次")
    print(f"  实际帧率: {actual_fps:.1f} FPS")
    print(f"  测试时长: {elapsed:.1f} 秒")
    
    # 显示最后一帧
    if ret:
        print(f"  最后一帧形状: {frame.shape}")
        cv2.imshow(f"Camera {device_id}", frame)
        print(f"  显示窗口，按任意键继续...")
        cv2.waitKey(0)  # 等待用户按键
        cv2.destroyAllWindows()
    
    cap.release()
    
    success = frame_count > 0 and actual_fps > 1.0
    print(f"  状态: {'OK' if success else 'FAILED'}")
    
    return success

def test_dual_camera_sequential(left_device=0, right_device=1, duration=5):
    """测试双摄像头顺序读取（仿cam.py策略）"""
    print(f"\n=== 测试双摄像头顺序读取 ({left_device}, {right_device}) ===")
    
    # 初始化摄像头列表 - 模仿cam.py的方式
    cameras = []
    valid_cameras = []
    
    # 按照cam.py的方式初始化摄像头
    for idx in [left_device, right_device]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # 设置分辨率（跟cam.py一样，但使用更适合测试的分辨率）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)  # 使用cam.py的帧率
            cameras.append(cap)
            valid_cameras.append(idx)
            print(f"成功打开摄像头 {idx}")
        else:
            print(f"无法打开摄像头 {idx}")
            cap.release()
    
    if len(cameras) != 2:
        print("ERROR: 无法打开两个摄像头")
        for cap in cameras:
            cap.release()
        return False
    
    print(f"总共打开了 {len(cameras)} 个摄像头: {valid_cameras}")
    
    # 使用cam.py的顺序读取策略
    successful_frames = 0
    failed_reads = 0
    start_time = time.time()
    last_left_frame = None
    last_right_frame = None
    
    print("开始顺序读取测试（仿cam.py模式）...")
    
    while time.time() - start_time < duration:
        frames = []
        all_success = True
        
        # 从所有摄像头顺序读取帧 - 完全模仿cam.py
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"无法读取摄像头 {valid_cameras[i]} 的帧")
                all_success = False
                break
        
        # 如果所有摄像头都成功读取帧 - 模仿cam.py逻辑
        if all_success and len(frames) == 2:
            successful_frames += 1
            last_left_frame = frames[0]
            last_right_frame = frames[1]
            
            if successful_frames == 1:
                print(f"  左帧形状: {frames[0].shape}")
                print(f"  右帧形状: {frames[1].shape}")
        else:
            failed_reads += 1
            time.sleep(0.01)  # 短暂等待
    
    elapsed = time.time() - start_time
    sequential_fps = successful_frames / elapsed if elapsed > 0 else 0
    
    print(f"顺序读取测试结果:")
    print(f"  成功读取: {successful_frames} 组帧")
    print(f"  失败读取: {failed_reads} 次")
    print(f"  顺序帧率: {sequential_fps:.1f} FPS")
    print(f"  成功率: {successful_frames/(successful_frames+failed_reads)*100:.1f}%")
    
    # 显示最后成功的帧组合
    if last_left_frame is not None and last_right_frame is not None:
        combined = cv2.hconcat([last_left_frame, last_right_frame])
        cv2.imshow("Dual Camera Sequential", combined)
        print("  显示双摄像头画面，按任意键继续...")
        cv2.waitKey(0)  # 等待用户按键
        cv2.destroyAllWindows()
    
    # 释放所有摄像头资源 - 模仿cam.py
    for i, cap in enumerate(cameras):
        cap.release()
        print(f"释放摄像头 {valid_cameras[i]}")
    
    success = successful_frames > 10 and sequential_fps > 5.0
    print(f"  状态: {'OK' if success else 'FAILED'}")
    
    return success

def test_threaded_capture(left_device=0, right_device=1, duration=5):
    """测试多线程采集（模拟实际系统）"""
    print(f"\n=== 测试多线程采集 ===")
    
    frame_buffer = Queue(maxsize=10)
    stop_event = threading.Event()
    
    def capture_thread():
        left_cap = cv2.VideoCapture(left_device)
        right_cap = cv2.VideoCapture(right_device)
        
        if not (left_cap.isOpened() and right_cap.isOpened()):
            print("ERROR: 摄像头初始化失败")
            return
        
        # 设置参数
        for cap in [left_cap, right_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 使用cam.py的顺序读取策略
        cameras = [left_cap, right_cap]
        valid_cameras = [left_device, right_device]
        
        frame_id = 0
        while not stop_event.is_set():
            frames = []
            all_success = True
            
            # 顺序读取所有摄像头 - 模仿cam.py
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    all_success = False
                    break
            
            # 如果所有摄像头都成功读取帧
            if all_success and len(frames) == 2:
                try:
                    stereo_frame = {
                        'frame_id': frame_id,
                        'timestamp': time.time(),
                        'left': frames[0],   # 顺序：第一个是左，第二个是右
                        'right': frames[1]
                    }
                    frame_buffer.put(stereo_frame, timeout=0.1)
                    frame_id += 1
                except:
                    # 缓冲区满，丢弃旧帧
                    try:
                        frame_buffer.get(block=False)
                        frame_buffer.put(stereo_frame, block=False)
                    except:
                        pass
            else:
                time.sleep(0.01)
        
        left_cap.release()
        right_cap.release()
        print("采集线程已停止")
    
    # 启动采集线程
    capture_t = threading.Thread(target=capture_thread)
    capture_t.daemon = True
    capture_t.start()
    
    # 主线程读取测试
    received_frames = 0
    timeout_count = 0
    start_time = time.time()
    
    print("开始多线程读取测试...")
    
    while time.time() - start_time < duration:
        try:
            frame_data = frame_buffer.get(timeout=1.0)
            received_frames += 1
            if received_frames == 1:
                print(f"  首帧ID: {frame_data['frame_id']}")
                print(f"  缓冲区大小: {frame_buffer.qsize()}")
        except Empty:
            timeout_count += 1
            print(f"  超时 #{timeout_count}")
            if timeout_count > 3:
                print("  多次超时，停止测试")
                break
    
    # 停止采集
    stop_event.set()
    capture_t.join(timeout=2)
    
    elapsed = time.time() - start_time
    threaded_fps = received_frames / elapsed
    
    print(f"多线程测试结果:")
    print(f"  接收帧数: {received_frames}")
    print(f"  超时次数: {timeout_count}")
    print(f"  处理帧率: {threaded_fps:.1f} FPS")
    print(f"  缓冲区剩余: {frame_buffer.qsize()}")
    
    success = received_frames > 10 and timeout_count < 3
    print(f"  状态: {'OK' if success else 'FAILED'}")
    
    return success

def main():
    """主诊断流程"""
    print("="*60)
    print("摄像头诊断工具")
    print("="*60)
    
    # 检测可用摄像头
    print("\n检测可用摄像头...")
    available_cameras = []
    
    for i in range(4):  # 检测前4个设备
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            print(f"  摄像头 {i}: 可用")
            cap.release()
        else:
            print(f"  摄像头 {i}: 不可用")
    
    if len(available_cameras) < 2:
        print(f"\nERROR: 只检测到 {len(available_cameras)} 个摄像头")
        print("需要至少2个摄像头进行立体SLAM")
        return False
    
    print(f"\n找到 {len(available_cameras)} 个可用摄像头: {available_cameras}")
    
    # 逐个测试摄像头
    left_device = available_cameras[0]
    right_device = available_cameras[1]
    
    success1 = test_single_camera(left_device)
    success2 = test_single_camera(right_device)
    
    if not (success1 and success2):
        print("\nERROR: 单摄像头测试失败")
        return False
    
    # 测试双摄像头顺序读取（cam.py策略）
    success3 = test_dual_camera_sequential(left_device, right_device)
    
    # 测试多线程采集
    success4 = test_threaded_capture(left_device, right_device)
    
    # 总结
    print("\n" + "="*60)
    if success1 and success2 and success3 and success4:
        print("SUCCESS: 所有摄像头测试通过")
        print(f"建议使用摄像头: 左={left_device}, 右={right_device}")
        print("\n可以运行以下命令:")
        print(f"python run_hybrid_slam.py --config configs/stereo_camera_config.yaml")
        return True
    else:
        print("FAILED: 摄像头测试失败")
        print("请检查摄像头连接和驱动")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1)