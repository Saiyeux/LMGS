#!/usr/bin/env python3
"""
最终工作解决方案 - 确保3D重建能显示且不卡死
"""

import cv2
import time
import numpy as np
import torch
from pathlib import Path
import threading
from queue import Queue, Empty

class SafeStereoCamera:
    """安全的立体相机采集"""
    def __init__(self, left_device=0, right_device=1):
        self.left_device = left_device
        self.right_device = right_device
        self.left_cap = None
        self.right_cap = None
        self.running = False
        
    def start(self):
        """启动相机"""
        try:
            self.left_cap = cv2.VideoCapture(self.left_device)
            self.right_cap = cv2.VideoCapture(self.right_device)
            
            if not (self.left_cap.isOpened() and self.right_cap.isOpened()):
                print("无法打开摄像头")
                return False
            
            # 设置分辨率
            self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            return True
        except Exception as e:
            print(f"相机启动失败: {e}")
            return False
    
    def get_stereo_frame(self):
        """获取立体帧"""
        if not self.running:
            return None, None
        
        try:
            ret_l, left_img = self.left_cap.read()
            ret_r, right_img = self.right_cap.read()
            
            if ret_l and ret_r:
                return left_img, right_img
            else:
                return None, None
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None
    
    def stop(self):
        """停止相机"""
        self.running = False
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()

class Simple3DReconstructor:
    """简单3D重建器 - 不依赖复杂的SLAM系统"""
    def __init__(self):
        self.points_3d = []
        self.colors_3d = []
        self.trajectory = []
        
        # 立体视觉参数
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 相机内参（可以通过标定获得更准确的值）
        self.fx = 525.0
        self.fy = 525.0  
        self.cx = 320.0
        self.cy = 240.0
        self.baseline = 0.12  # 12cm基线，根据实际情况调整
        
    def process_stereo_frame(self, left_img, right_img):
        """处理立体帧，生成3D点云"""
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 计算视差图
            disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # 从视差生成3D点
            points_3d, colors_3d = self._disparity_to_3d(left_img, disparity)
            
            # 累积点云（简单的SLAM）
            if len(points_3d) > 0:
                self.points_3d.extend(points_3d)
                self.colors_3d.extend(colors_3d)
                
                # 限制点云大小
                if len(self.points_3d) > 5000:
                    self.points_3d = self.points_3d[-5000:]
                    self.colors_3d = self.colors_3d[-5000:]
                
                # 添加到轨迹（简化的位姿估计）
                current_pos = np.array([len(self.trajectory) * 0.1, 0, 0])
                self.trajectory.append(current_pos)
                
                return True
        except Exception as e:
            print(f"3D重建失败: {e}")
            return False
        
        return False
    
    def _disparity_to_3d(self, color_img, disparity):
        """从视差图生成3D点云"""
        points_3d = []
        colors_3d = []
        
        h, w = disparity.shape
        step = 8  # 采样间隔
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = disparity[y, x]
                
                if d > 0:  # 有效视差
                    # 3D坐标计算
                    Z = (self.fx * self.baseline) / d
                    if 0.5 < Z < 10.0:  # 深度范围过滤
                        X = (x - self.cx) * Z / self.fx
                        Y = (y - self.cy) * Z / self.fy
                        
                        points_3d.append([X, Y, Z])
                        
                        # 获取颜色
                        color = color_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d
    
    def get_reconstruction_data(self):
        """获取3D重建数据"""
        if len(self.points_3d) > 0:
            return {
                'points': np.array(self.points_3d),
                'colors': np.array(self.colors_3d),
                'type': 'stereo_reconstruction',
                'count': len(self.points_3d)
            }
        return None

class RealtimeVisualization:
    """实时可视化显示"""
    def __init__(self):
        self.window_name = "Dual Camera 3D Reconstruction"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 800)
        
    def update_display(self, left_img, right_img, reconstruction_data):
        """更新显示"""
        try:
            # 创建显示画布
            h, w = left_img.shape[:2]
            canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            
            # 左上：左相机图像
            canvas[0:h, 0:w] = left_img
            cv2.putText(canvas, "Left Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 右上：右相机图像  
            canvas[0:h, w:w*2] = right_img
            cv2.putText(canvas, "Right Camera", (w+10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 左下：深度信息显示
            depth_info = self._create_depth_info(w, h, reconstruction_data)
            canvas[h:h*2, 0:w] = depth_info
            
            # 右下：3D点云投影显示
            pointcloud_display = self._create_pointcloud_display(w, h, reconstruction_data)
            canvas[h:h*2, w:w*2] = pointcloud_display
            
            # 显示
            cv2.imshow(self.window_name, canvas)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q')
            
        except Exception as e:
            print(f"显示更新失败: {e}")
            return True
    
    def _create_depth_info(self, width, height, reconstruction_data):
        """创建深度信息显示"""
        info_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(info_img, "3D Reconstruction Info", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if reconstruction_data:
            points_count = reconstruction_data.get('count', 0)
            recon_type = reconstruction_data.get('type', 'unknown')
            
            cv2.putText(info_img, f"Points: {points_count}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(info_img, f"Type: {recon_type}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(info_img, "Status: ACTIVE", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(info_img, "Status: NO DATA", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return info_img
    
    def _create_pointcloud_display(self, width, height, reconstruction_data):
        """创建点云投影显示"""
        display_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(display_img, "3D Point Cloud (Top View)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if reconstruction_data and reconstruction_data.get('points') is not None:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                # 投影到XZ平面（俯视图）
                x_coords = points[:, 0]
                z_coords = points[:, 2]
                
                # 归一化到显示坐标
                if len(x_coords) > 0:
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    z_min, z_max = np.min(z_coords), np.max(z_coords)
                    
                    if x_max > x_min and z_max > z_min:
                        x_norm = ((x_coords - x_min) / (x_max - x_min) * (width - 40) + 20).astype(int)
                        z_norm = ((z_coords - z_min) / (z_max - z_min) * (height - 80) + 50).astype(int)
                        
                        # 绘制点云
                        for i in range(0, len(x_norm), max(1, len(x_norm)//500)):  # 限制显示点数
                            if 0 <= x_norm[i] < width and 0 <= z_norm[i] < height:
                                color = colors[i] if i < len(colors) else [255, 255, 255]
                                cv2.circle(display_img, (x_norm[i], z_norm[i]), 1,
                                          (int(color[0]), int(color[1]), int(color[2])), -1)
        
        return display_img
    
    def close(self):
        """关闭显示"""
        cv2.destroyWindow(self.window_name)

def main():
    """主函数"""
    print("=" * 60)
    print("双目相机3D重建系统 - 最终工作版本")
    print("=" * 60)
    print("按 'q' 退出")
    print()
    
    # 初始化组件
    camera = SafeStereoCamera(0, 1)
    reconstructor = Simple3DReconstructor()
    visualizer = RealtimeVisualization()
    
    # 启动相机
    if not camera.start():
        print("相机启动失败！")
        return
    
    print("系统启动成功，开始3D重建...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 获取立体图像
            left_img, right_img = camera.get_stereo_frame()
            
            if left_img is not None and right_img is not None:
                frame_count += 1
                
                # 3D重建处理
                if frame_count % 2 == 0:  # 每2帧进行一次3D重建
                    success = reconstructor.process_stereo_frame(left_img, right_img)
                    if success:
                        print(f"Frame {frame_count}: 3D重建成功")
                
                # 获取重建数据
                reconstruction_data = reconstructor.get_reconstruction_data()
                
                # 更新显示
                if not visualizer.update_display(left_img, right_img, reconstruction_data):
                    break
                
                # 性能统计
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    points_count = reconstruction_data['count'] if reconstruction_data else 0
                    print(f"处理了 {frame_count} 帧, FPS: {fps:.1f}, 3D点数: {points_count}")
            
            time.sleep(0.01)  # 控制帧率
            
    except KeyboardInterrupt:
        print("\\n用户中断...")
    except Exception as e:
        print(f"\\n系统错误: {e}")
    finally:
        # 清理资源
        camera.stop()
        visualizer.close()
        
        # 保存结果
        reconstruction_data = reconstructor.get_reconstruction_data()
        if reconstruction_data:
            save_path = Path("final_3d_reconstruction.npz")
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            print(f"3D重建结果已保存: {save_path}")
        
        print("系统已关闭")

if __name__ == "__main__":
    main()