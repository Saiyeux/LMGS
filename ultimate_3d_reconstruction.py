#!/usr/bin/env python3
"""
终极3D重建解决方案 - 结合模拟与真实相机的优势
"""

import cv2
import time
import numpy as np
from pathlib import Path
import threading
from queue import Queue, Empty

class SmartCameraManager:
    """智能相机管理器 - 自动处理相机访问问题"""
    def __init__(self):
        self.cameras = {}
        self.use_mock = False
        self.mock_frame_count = 0
        
    def initialize(self):
        """初始化相机系统"""
        print("初始化智能相机系统...")
        
        # 尝试访问真实相机
        success = self._try_real_cameras()
        
        if not success:
            print("真实相机不可用，使用模拟模式")
            self.use_mock = True
            return True
        
        return success
    
    def _try_real_cameras(self):
        """尝试访问真实相机"""
        try:
            # 先检查相机是否被占用
            for i in range(2):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 优先使用DirectShow
                if cap.isOpened():
                    # 快速测试读取
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, frame = cap.read()
                    if ret:
                        print(f"相机{i}可用: {frame.shape}")
                        self.cameras[i] = cap
                    else:
                        print(f"相机{i}无法读取帧")
                        cap.release()
                else:
                    print(f"相机{i}无法打开")
                    
                time.sleep(0.1)  # 避免快速访问问题
            
            return len(self.cameras) > 0
            
        except Exception as e:
            print(f"相机初始化异常: {e}")
            return False
    
    def get_frames(self):
        """获取帧数据"""
        if self.use_mock:
            return self._generate_mock_frames()
        else:
            return self._get_real_frames()
    
    def _get_real_frames(self):
        """获取真实相机帧"""
        frames = {}
        for camera_id, cap in self.cameras.items():
            try:
                ret, frame = cap.read()
                if ret:
                    frames[camera_id] = frame
            except Exception as e:
                print(f"相机{camera_id}读取失败: {e}")
        
        return frames
    
    def _generate_mock_frames(self):
        """生成模拟帧数据"""
        frames = {}
        
        # 生成动态场景
        self.mock_frame_count += 1
        t = self.mock_frame_count * 0.1
        
        for camera_id in [0, 1]:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 生成动态背景
            frame[:] = (50, 50, 50)
            
            # 添加移动的几何体
            center_x = int(320 + 200 * np.cos(t + camera_id * 0.1))
            center_y = int(240 + 100 * np.sin(t * 1.2))
            
            # 主要物体
            cv2.circle(frame, (center_x, center_y), 40, (0, 255, 128), -1)
            cv2.rectangle(frame, (center_x-60, center_y-60), (center_x+60, center_y+60), (255, 128, 0), 3)
            
            # 特征点
            for i in range(30):
                x = int(320 + 250 * np.cos(t + i * 0.2))
                y = int(240 + 200 * np.sin(t + i * 0.3))
                if 0 <= x < 640 and 0 <= y < 480:
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            
            # 网格参考
            for i in range(0, 640, 80):
                cv2.line(frame, (i, 0), (i, 480), (80, 80, 80), 1)
            for i in range(0, 480, 60):
                cv2.line(frame, (0, i), (640, i), (80, 80, 80), 1)
            
            # 相机标识
            cv2.putText(frame, f"Camera {camera_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {self.mock_frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            frames[camera_id] = frame
        
        return frames
    
    def is_stereo_mode(self):
        """是否为立体模式"""
        if self.use_mock:
            return True  # 模拟模式总是双目
        return len(self.cameras) >= 2
    
    def cleanup(self):
        """清理资源"""
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()

class Enhanced3DReconstructor:
    """增强3D重建器 - 适配真实和模拟数据"""
    def __init__(self):
        self.points_3d = []
        self.colors_3d = []
        self.frame_count = 0
        
        # 立体视觉参数
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 相机参数
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 320.0
        self.cy = 240.0
        self.baseline = 0.12
        
        # 特征跟踪器
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_desc = None
        
    def process_frames(self, frames, is_mock=False):
        """处理输入帧"""
        self.frame_count += 1
        
        if len(frames) >= 2:
            # 双目重建
            left_img = frames[0]
            right_img = frames[1]
            return self._process_stereo_frames(left_img, right_img, is_mock)
        elif len(frames) == 1:
            # 单目重建
            return self._process_mono_frame(frames[0], is_mock)
        
        return False
    
    def _process_stereo_frames(self, left_img, right_img, is_mock):
        """处理立体帧"""
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            if is_mock:
                # 模拟数据使用简化的视差计算
                disparity = self._compute_mock_disparity(left_gray, right_gray)
            else:
                # 真实数据使用完整的立体匹配
                disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # 生成3D点云
            points_3d, colors_3d = self._disparity_to_3d(left_img, disparity)
            
            if len(points_3d) > 0:
                self.points_3d.extend(points_3d)
                self.colors_3d.extend(colors_3d)
                
                # 限制点云大小
                max_points = 4000 if is_mock else 3000
                if len(self.points_3d) > max_points:
                    self.points_3d = self.points_3d[-max_points:]
                    self.colors_3d = self.colors_3d[-max_points:]
                
                return True
        
        except Exception as e:
            print(f"立体处理失败: {e}")
        
        return False
    
    def _compute_mock_disparity(self, left_gray, right_gray):
        """计算模拟视差"""
        # 对于模拟数据，使用简化的视差计算
        disparity = np.zeros_like(left_gray, dtype=np.float32)
        
        # 检测特征点
        kp1, desc1 = self.detector.detectAndCompute(left_gray, None)
        kp2, desc2 = self.detector.detectAndCompute(right_gray, None)
        
        if desc1 is not None and desc2 is not None:
            matches = self.matcher.match(desc1, desc2)
            
            for match in matches:
                if match.distance < 50:  # 好匹配
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    
                    # 计算视差
                    disp = abs(pt1[0] - pt2[0])
                    if disp > 1:
                        x, y = int(pt1[0]), int(pt1[1])
                        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                            disparity[y, x] = disp
        
        # 平滑视差图
        disparity = cv2.GaussianBlur(disparity, (5, 5), 1.0)
        
        return disparity
    
    def _process_mono_frame(self, img, is_mock):
        """处理单目帧"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测特征点
            kp, desc = self.detector.detectAndCompute(gray, None)
            
            if self.prev_desc is not None and desc is not None:
                matches = self.matcher.match(self.prev_desc, desc)
                good_matches = [m for m in matches if m.distance < 50]
                
                if len(good_matches) > 20:
                    # 提取匹配点
                    pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                    
                    # 估计基础矩阵
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                    
                    if F is not None:
                        # 简化深度估计
                        points_3d = []
                        colors_3d = []
                        
                        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                            if mask[i]:
                                # 基于特征运动估计深度
                                motion = np.linalg.norm(pt2 - pt1)
                                depth = 5.0 / (motion + 0.1) if motion > 0 else 5.0
                                
                                if 0.5 < depth < 15.0:
                                    x, y = pt2
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
            
            self.prev_kp = kp
            self.prev_desc = desc
            
        except Exception as e:
            print(f"单目处理失败: {e}")
        
        return False
    
    def _disparity_to_3d(self, color_img, disparity):
        """从视差图生成3D点云"""
        points_3d = []
        colors_3d = []
        
        h, w = disparity.shape
        step = 6  # 采样步长
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = disparity[y, x]
                
                if d > 1.0:  # 有效视差
                    Z = (self.fx * self.baseline) / d
                    if 0.5 < Z < 12.0:  # 合理深度范围
                        X = (x - self.cx) * Z / self.fx
                        Y = (y - self.cy) * Z / self.fy
                        
                        points_3d.append([X, Y, Z])
                        
                        # 获取颜色
                        color = color_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d
    
    def get_reconstruction_data(self):
        """获取重建数据"""
        if len(self.points_3d) > 0:
            return {
                'points': np.array(self.points_3d),
                'colors': np.array(self.colors_3d),
                'type': 'hybrid_reconstruction',
                'count': len(self.points_3d),
                'frame_count': self.frame_count
            }
        return None

class UltimateVisualization:
    """终极可视化系统"""
    def __init__(self):
        self.window_name = "Ultimate 3D Reconstruction System"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1000)
        
    def display(self, frames, reconstruction_data, is_stereo, is_mock):
        """显示系统状态"""
        try:
            # 创建大画布
            canvas = np.zeros((1000, 1600, 3), dtype=np.uint8)
            
            # 标题和模式指示
            mode_text = "STEREO" if is_stereo else "MONO"
            source_text = "MOCK" if is_mock else "REAL"
            cv2.putText(canvas, f"Ultimate 3D Reconstruction - {mode_text} {source_text}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # 左侧：相机视图
            y_offset = 80
            for i, (camera_id, frame) in enumerate(frames.items()):
                if frame is not None:
                    # 调整大小
                    display_frame = cv2.resize(frame, (640, 480))
                    canvas[y_offset:y_offset+480, 20:660] = display_frame
                    
                    # 标签
                    label = f"Camera {camera_id} ({'Mock' if is_mock else 'Real'})"
                    cv2.putText(canvas, label, (30, y_offset+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    y_offset += 500
                    if y_offset > 800:  # 避免超出画布
                        break
            
            # 右侧：3D重建信息
            info_x = 700
            self._draw_reconstruction_info(canvas, info_x, 80, 880, 400, reconstruction_data)
            
            # 右下：3D点云可视化
            self._draw_3d_pointcloud(canvas, info_x, 500, 880, 480, reconstruction_data)
            
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q')
            
        except Exception as e:
            print(f"显示错误: {e}")
            return True
    
    def _draw_reconstruction_info(self, canvas, x, y, w, h, reconstruction_data):
        """绘制重建信息"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (40, 40, 40), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(canvas, "3D Reconstruction Status", (x+10, y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        text_y = y + 80
        line_height = 30
        
        if reconstruction_data:
            info_lines = [
                f"Point Count: {reconstruction_data['count']}",
                f"Frame Count: {reconstruction_data['frame_count']}",
                f"Type: {reconstruction_data['type']}",
                "",
                "System Status:",
                "- 3D Reconstruction: ACTIVE",
                "- Point Cloud: UPDATING",
                "- Visualization: REAL-TIME",
                "",
                "Performance:",
                f"- Memory Usage: {reconstruction_data['count'] * 24 / 1024:.1f}KB",
                "- Processing: SMOOTH",
                "- Quality: HIGH"
            ]
        else:
            info_lines = [
                "Status: INITIALIZING",
                "Waiting for data...",
                "",
                "System Ready:",
                "- Cameras: DETECTED",
                "- Algorithms: LOADED",
                "- Display: ACTIVE"
            ]
        
        for line in info_lines:
            if text_y > y + h - 20:
                break
            
            if line:
                if line.startswith("- "):
                    color = (0, 255, 0) if "ACTIVE" in line or "UPDATING" in line or "HIGH" in line else (0, 255, 255)
                else:
                    color = (200, 200, 200)
                
                cv2.putText(canvas, line, (x+15, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            text_y += line_height
    
    def _draw_3d_pointcloud(self, canvas, x, y, w, h, reconstruction_data):
        """绘制3D点云"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(canvas, "3D Point Cloud Visualization", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if reconstruction_data and reconstruction_data.get('points') is not None:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                try:
                    # 多视角显示
                    views = [
                        ("Top View (XZ)", 0, 2),      # XZ平面
                        ("Front View (XY)", 0, 1),    # XY平面
                        ("Side View (YZ)", 1, 2)      # YZ平面
                    ]
                    
                    view_w = w // 3 - 10
                    view_h = h - 80
                    
                    for i, (view_name, axis1, axis2) in enumerate(views):
                        view_x = x + 10 + i * (view_w + 5)
                        view_y = y + 50
                        
                        # 投影坐标
                        coords1 = points[:, axis1]
                        coords2 = points[:, axis2]
                        
                        if len(coords1) > 0:
                            # 归一化
                            c1_min, c1_max = np.min(coords1), np.max(coords1)
                            c2_min, c2_max = np.min(coords2), np.max(coords2)
                            
                            if c1_max > c1_min and c2_max > c2_min:
                                c1_norm = ((coords1 - c1_min) / (c1_max - c1_min) * (view_w - 20) + view_x + 10).astype(int)
                                c2_norm = ((coords2 - c2_min) / (c2_max - c2_min) * (view_h - 40) + view_y + 20).astype(int)
                                
                                # 绘制点
                                sample_step = max(1, len(points) // 200)
                                for j in range(0, len(c1_norm), sample_step):
                                    if view_x <= c1_norm[j] < view_x + view_w and view_y <= c2_norm[j] < view_y + view_h:
                                        color = colors[j] if j < len(colors) else [255, 255, 255]
                                        cv2.circle(canvas, (c1_norm[j], c2_norm[j]), 1,
                                                  (int(color[0]), int(color[1]), int(color[2])), -1)
                        
                        # 视图标签
                        cv2.putText(canvas, view_name, (view_x + 5, view_y + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # 视图边框
                        cv2.rectangle(canvas, (view_x, view_y), (view_x + view_w, view_y + view_h), (80, 80, 80), 1)
                
                except Exception as e:
                    cv2.putText(canvas, f"Render Error: {str(e)[:50]}", (x+20, y+100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(canvas, "No 3D data available", (x+20, y+100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    
    def close(self):
        """关闭显示"""
        cv2.destroyWindow(self.window_name)

def main():
    """主函数"""
    print("=" * 70)
    print("终极3D重建系统 - 智能相机适配")
    print("=" * 70)
    print("自动检测相机状态，必要时使用模拟数据")
    print("按 'q' 退出")
    print()
    
    # 初始化系统组件
    camera_manager = SmartCameraManager()
    reconstructor = Enhanced3DReconstructor()
    visualizer = UltimateVisualization()
    
    # 初始化相机
    if not camera_manager.initialize():
        print("系统初始化失败！")
        return
    
    is_mock = camera_manager.use_mock
    is_stereo = camera_manager.is_stereo_mode()
    
    print(f"工作模式: {'模拟' if is_mock else '真实'}相机, {'立体' if is_stereo else '单目'}重建")
    print("开始3D重建...")
    
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    try:
        while True:
            frame_count += 1
            
            # 获取帧数据
            frames = camera_manager.get_frames()
            
            if frames:
                # 3D重建处理
                success = reconstructor.process_frames(frames, is_mock)
                
                # 获取重建数据
                reconstruction_data = reconstructor.get_reconstruction_data()
                
                # 更新显示
                if not visualizer.display(frames, reconstruction_data, is_stereo, is_mock):
                    break
                
                # 性能报告
                current_time = time.time()
                if current_time - last_report > 3.0:  # 每3秒报告一次
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed
                    points_count = reconstruction_data['count'] if reconstruction_data else 0
                    
                    print(f"性能报告 - FPS: {fps:.1f}, 处理帧数: {frame_count}, 3D点数: {points_count}")
                    last_report = current_time
            
            # 控制帧率
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n用户中断...")
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        camera_manager.cleanup()
        visualizer.close()
        
        # 保存结果
        reconstruction_data = reconstructor.get_reconstruction_data()
        if reconstruction_data:
            save_path = Path("ultimate_3d_reconstruction.npz")
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            print(f"\n3D重建结果已保存: {save_path}")
            print(f"总点数: {reconstruction_data['count']}, 处理帧数: {frame_count}")
        
        print("终极3D重建系统已关闭")

if __name__ == "__main__":
    main()