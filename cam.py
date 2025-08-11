import cv2
import os
import datetime

# 摄像头索引列表
camera_indices = [0, 1]  # 可以根据需要添加更多摄像头索引

# 创建保存目录
save_dir = "captures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化摄像头列表
cameras = []
valid_cameras = []
video_writers = []
is_recording = False

# 遍历摄像头索引列表并初始化
for idx in camera_indices:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        # 设置1080p分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cameras.append(cap)
        valid_cameras.append(idx)
        video_writers.append(None)  # 初始化录像写入器为None
        print(f"成功打开摄像头 {idx}")
    else:
        print(f"无法打开摄像头 {idx}")
        cap.release()

# 检查是否有可用的摄像头
if not cameras:
    print("没有可用的摄像头")
    exit()

print(f"总共打开了 {len(cameras)} 个摄像头: {valid_cameras}")

# 获取视频参数
fps = 20.0
frame_width = int(cameras[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cameras[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n控制说明:")
print("按 's' 键截图")
print("按 'r' 键开始/停止录像")
print("按 'q' 键退出")

try:
    while True:
        frames = []
        all_success = True
        
        # 从所有摄像头读取帧
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
            # 显示所有摄像头的帧
            for i, frame in enumerate(frames):
                # 在帧上添加录像状态指示
                if is_recording:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # 红色圆点
                    cv2.putText(frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(f'Camera {valid_cameras[i]}', frame)
                
                # 如果正在录像，写入视频文件
                if is_recording and video_writers[i] is not None:
                    video_writers[i].write(frame)
        else:
            print("部分摄像头读取失败")
            break
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # 退出程序
            break
        elif key == ord('s'):
            # 截图
            if frames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, frame in enumerate(frames):
                    filename = os.path.join(save_dir, f"screenshot_cam{valid_cameras[i]}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"截图保存: {filename}")
        elif key == ord('r'):
            # 开始/停止录像
            if not is_recording:
                # 开始录像
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for i in range(len(cameras)):
                    filename = os.path.join(save_dir, f"video_cam{valid_cameras[i]}_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
                    video_writers[i] = writer
                    print(f"开始录像: {filename}")
                is_recording = True
            else:
                # 停止录像
                for i, writer in enumerate(video_writers):
                    if writer is not None:
                        writer.release()
                        video_writers[i] = None
                        print(f"停止录像摄像头 {valid_cameras[i]}")
                is_recording = False
            
except KeyboardInterrupt:
    print("程序被用户终止")
    
finally:
    # 停止录像
    if is_recording:
        for i, writer in enumerate(video_writers):
            if writer is not None:
                writer.release()
                print(f"停止录像摄像头 {valid_cameras[i]}")
    
    # 释放所有摄像头资源
    for i, cap in enumerate(cameras):
        cap.release()
        print(f"释放摄像头 {valid_cameras[i]}")
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()