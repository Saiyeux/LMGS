#!/usr/bin/env python3
"""
基于真实相机的3D重建 - 解决相机访问问题
"""

import cv2
import time
import numpy as np
from pathlib import Path
import threading
from queue import Queue, Empty

class RobustCameraAccess:
    """强健的相机访问类 - 解决Windows相机问题"""
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = None
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.capture_thread = None
        
    def _capture_frames(self):
        """后台捕获帧的线程"""
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # 清空旧帧，保持最新
                        try:
                            while not self.frame_queue.empty():
                                self.frame_queue.get_nowait()
                        except Empty:
                            pass
                        
                        try:
                            self.frame_queue.put_nowait(frame)
                        except:
                            pass
                time.sleep(0.033)  # ~30fps
            except Exception as e:
                print(f"Capture thread error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """启动相机"""
        try:
            # 尝试不同的相机后端
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                print(f"尝试相机后端: {backend}")
                self.cap = cv2.VideoCapture(self.device_id, backend)
                
                if self.cap.isOpened():
                    # 设置相机参数
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
                    
                    # 测试读取
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"相机{self.device_id}启动成功，后端: {backend}")
                        
                        # 启动后台捕获线程
                        self.running = True
                        self.capture_thread = threading.Thread(target=self._capture_frames)
                        self.capture_thread.daemon = True
                        self.capture_thread.start()
                        return True
                
                self.cap.release()
                self.cap = None
            
            print(f"无法启动相机{self.device_id}")
            return False
            
        except Exception as e:
            print(f"相机启动异常: {e}")
            return False
    
    def get_frame(self):
        """获取最新帧"""
        try:
            if not self.running:
                return None
            return self.frame_queue.get(timeout=0.1)
        except Empty:
            return None
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None
    
    def stop(self):
        """停止相机"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

class FlexibleStereoReconstructor:
    """灵活的立体重建器 - 支持单目和双目"""
    def __init__(self):
        self.points_3d = []
        self.colors_3d = []
        self.frame_count = 0
        self.use_stereo = True
        
        # 立体视觉参数
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 光流跟踪器（单目模式）
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_points = None
        
        # 相机参数
        self.fx = 525.0
        self.fy = 525.0  
        self.cx = 320.0
        self.cy = 240.0
        self.baseline = 0.12
        
    def process_stereo_frames(self, left_img, right_img):
        """处理立体帧"""
        try:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 计算视差
            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # 生成3D点
            points_3d, colors_3d = self._disparity_to_3d(left_img, disparity)
            
            if len(points_3d) > 0:
                self.points_3d.extend(points_3d)
                self.colors_3d.extend(colors_3d)
                
                # 限制点云大小
                if len(self.points_3d) > 3000:
                    self.points_3d = self.points_3d[-3000:]
                    self.colors_3d = self.colors_3d[-3000:]
                
                return True
                
        except Exception as e:
            print(f"立体重建失败: {e}")
        
        return False
    
    def process_mono_frame(self, img):
        """处理单目帧（使用光流估计深度）"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is not None and self.prev_points is not None:
                # 光流跟踪
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None, **self.lk_params
                )
                
                # 过滤有效点
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                if len(good_new) > 10:
                    # 简单的深度估计（基于光流幅度）
                    flow_magnitude = np.linalg.norm(good_new - good_old, axis=1)
                    depths = 5.0 / (flow_magnitude + 0.1)  # 简化深度估计
                    
                    # 生成3D点
                    points_3d = []
                    colors_3d = []
                    
                    for i, (point, depth) in enumerate(zip(good_new, depths)):
                        if 0.5 < depth < 20.0:
                            x, y = point.ravel()
                            X = (x - self.cx) * depth / self.fx
                            Y = (y - self.cy) * depth / self.fy
                            Z = depth
                            
                            points_3d.append([X, Y, Z])
                            
                            # 获取颜色
                            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                                color = img[int(y), int(x)]
                                colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
                    
                    if len(points_3d) > 0:
                        self.points_3d.extend(points_3d)
                        self.colors_3d.extend(colors_3d)
                        
                        # 限制点云大小
                        if len(self.points_3d) > 2000:
                            self.points_3d = self.points_3d[-2000:]
                            self.colors_3d = self.colors_3d[-2000:]
                        
                        return True
            
            # 检测新特征点
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if corners is not None:
                self.prev_points = corners
            
            self.prev_gray = gray.copy()
            
        except Exception as e:
            print(f"单目重建失败: {e}")
        
        return False
    
    def _disparity_to_3d(self, color_img, disparity):
        """从视差图生成3D点"""
        points_3d = []
        colors_3d = []
        
        h, w = disparity.shape
        step = 8
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = disparity[y, x]
                
                if d > 0:
                    Z = (self.fx * self.baseline) / d
                    if 0.5 < Z < 10.0:
                        X = (x - self.cx) * Z / self.fx
                        Y = (y - self.cy) * Z / self.fy
                        
                        points_3d.append([X, Y, Z])
                        
                        color = color_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d
    
    def get_reconstruction_data(self):
        """获取重建数据"""
        if len(self.points_3d) > 0:
            return {
                'points': np.array(self.points_3d),
                'colors': np.array(self.colors_3d),
                'type': 'stereo_reconstruction' if self.use_stereo else 'mono_reconstruction',
                'count': len(self.points_3d)
            }
        return None

class AdaptiveVisualization:
    """自适应可视化"""
    def __init__(self):
        self.window_name = "Adaptive Camera 3D Reconstruction"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
    
    def display(self, left_img, right_img, reconstruction_data, is_stereo):
        """显示当前状态"""
        try:
            h, w = 400, 600
            if left_img is not None:
                h, w = left_img.shape[:2]
            
            canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            
            # 左上：左相机或主相机
            if left_img is not None:
                canvas[0:h, 0:w] = cv2.resize(left_img, (w, h))
                cv2.putText(canvas, "Left/Main Camera", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 右上：右相机或状态信息
            if right_img is not None and is_stereo:
                canvas[0:h, w:w*2] = cv2.resize(right_img, (w, h))
                cv2.putText(canvas, "Right Camera", (w+10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # 单目模式信息
                info_panel = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(info_panel, "MONO MODE", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.putText(info_panel, "Using Optical Flow", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                canvas[0:h, w:w*2] = info_panel
            
            # 下方：3D重建信息和点云显示
            recon_display = self._create_reconstruction_display(w*2, h, reconstruction_data, is_stereo)
            canvas[h:h*2, 0:w*2] = recon_display
            
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q')
            
        except Exception as e:
            print(f"显示错误: {e}")
            return True
    
    def _create_reconstruction_display(self, width, height, reconstruction_data, is_stereo):
        """创建重建显示"""
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 左半部分：统计信息
        cv2.putText(display, "3D Reconstruction Status", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if reconstruction_data:
            points_count = reconstruction_data['count']
            recon_type = reconstruction_data['type']
            
            cv2.putText(display, f"Points: {points_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display, f"Mode: {'STEREO' if is_stereo else 'MONO'}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display, f"Type: {recon_type}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display, "Status: ACTIVE", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Status: INITIALIZING", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 右半部分：点云可视化
        if reconstruction_data:
            self._draw_pointcloud(display, width//2, 0, width//2, height, reconstruction_data)
        
        return display
    
    def _draw_pointcloud(self, canvas, x_offset, y_offset, width, height, reconstruction_data):
        """绘制点云"""
        try:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                # 投影到XZ平面
                x_coords = points[:, 0]
                z_coords = points[:, 2]
                
                if len(x_coords) > 0:
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    z_min, z_max = np.min(z_coords), np.max(z_coords)
                    
                    if x_max > x_min and z_max > z_min:
                        x_norm = ((x_coords - x_min) / (x_max - x_min) * (width - 40) + x_offset + 20).astype(int)
                        z_norm = ((z_coords - z_min) / (z_max - z_min) * (height - 40) + y_offset + 20).astype(int)
                        
                        # 绘制点
                        sample_step = max(1, len(points) // 500)
                        for i in range(0, len(x_norm), sample_step):
                            if x_offset <= x_norm[i] < x_offset + width and y_offset <= z_norm[i] < y_offset + height:
                                color = colors[i] if i < len(colors) else [255, 255, 255]
                                cv2.circle(canvas, (x_norm[i], z_norm[i]), 2,
                                          (int(color[0]), int(color[1]), int(color[2])), -1)
                
                # 标题
                cv2.putText(canvas, "3D Point Cloud (Top View)", (x_offset + 10, y_offset + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"点云绘制错误: {e}")
    
    def close(self):
        """关闭显示"""
        cv2.destroyWindow(self.window_name)

def main():
    """主函数"""
    print("=" * 60)
    print("自适应相机3D重建系统")
    print("=" * 60)
    print("按 'q' 退出")
    print("自动检测单目/双目模式")
    print()
    
    # 初始化相机
    left_camera = RobustCameraAccess(0)
    right_camera = RobustCameraAccess(1)
    
    print("启动左相机...")
    left_ok = left_camera.start()
    
    print("尝试启动右相机...")
    right_ok = right_camera.start()
    
    if not left_ok:
        print("无法启动任何相机！")
        return
    
    # 确定工作模式
    is_stereo = left_ok and right_ok
    mode_text = "立体模式" if is_stereo else "单目模式"
    print(f"工作模式: {mode_text}")
    
    # 初始化重建器和显示器
    reconstructor = FlexibleStereoReconstructor()
    reconstructor.use_stereo = is_stereo
    visualizer = AdaptiveVisualization()
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            
            # 获取图像
            left_img = left_camera.get_frame()
            right_img = right_camera.get_frame() if is_stereo else None
            
            if left_img is not None:
                # 3D重建
                if is_stereo and right_img is not None:
                    success = reconstructor.process_stereo_frames(left_img, right_img)
                else:
                    success = reconstructor.process_mono_frame(left_img)
                
                if success and frame_count % 10 == 0:
                    reconstruction_data = reconstructor.get_reconstruction_data()
                    points_count = reconstruction_data['count'] if reconstruction_data else 0
                    print(f"Frame {frame_count}: 重建成功，点数: {points_count}")
                
                # 显示
                reconstruction_data = reconstructor.get_reconstruction_data()
                if not visualizer.display(left_img, right_img, reconstruction_data, is_stereo):
                    break
            
            time.sleep(0.01)  # 控制帧率
            
    except KeyboardInterrupt:
        print("\n用户中断...")
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        left_camera.stop()
        if is_stereo:
            right_camera.stop()
        visualizer.close()
        
        # 保存结果
        reconstruction_data = reconstructor.get_reconstruction_data()
        if reconstruction_data:
            save_path = Path("adaptive_3d_reconstruction.npz")
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            print(f"3D重建结果已保存: {save_path}")
        
        print("系统已关闭")

if __name__ == "__main__":
    main()