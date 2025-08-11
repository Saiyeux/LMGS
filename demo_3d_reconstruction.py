#!/usr/bin/env python3
"""
3D重建演示 - 使用模拟数据显示重建效果
"""

import cv2
import time
import numpy as np
import torch
from pathlib import Path

class Mock3DReconstructor:
    """模拟3D重建器 - 生成测试点云数据"""
    def __init__(self):
        self.points_3d = []
        self.colors_3d = []
        self.frame_count = 0
        
    def process_frame(self):
        """处理模拟帧，生成3D点云"""
        # 每帧生成一些新的3D点
        num_new_points = 50
        
        # 生成围绕轨迹的点云
        t = self.frame_count * 0.1
        center_x = 2 * np.cos(t)
        center_z = 2 * np.sin(t)
        
        # 在中心周围生成随机点
        new_points = []
        new_colors = []
        
        for i in range(num_new_points):
            # 3D位置
            x = center_x + np.random.normal(0, 0.5)
            y = np.random.normal(0, 0.3)
            z = center_z + np.random.normal(0, 0.5)
            new_points.append([x, y, z])
            
            # 颜色（根据位置着色）
            r = int(128 + 127 * np.sin(x))
            g = int(128 + 127 * np.sin(y + 2))
            b = int(128 + 127 * np.sin(z + 4))
            new_colors.append([r, g, b])
        
        # 添加到总点云
        self.points_3d.extend(new_points)
        self.colors_3d.extend(new_colors)
        
        # 限制点云大小
        if len(self.points_3d) > 2000:
            self.points_3d = self.points_3d[-2000:]
            self.colors_3d = self.colors_3d[-2000:]
        
        self.frame_count += 1
        return len(new_points) > 0
    
    def get_reconstruction_data(self):
        """获取3D重建数据"""
        if len(self.points_3d) > 0:
            return {
                'points': np.array(self.points_3d),
                'colors': np.array(self.colors_3d),
                'type': 'mock_gaussian_splatting',
                'count': len(self.points_3d)
            }
        return None

class Advanced3DVisualization:
    """高级3D可视化显示"""
    def __init__(self):
        self.window_name = "3D Reconstruction Demo"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)
        
    def create_demo_display(self, reconstruction_data, frame_count):
        """创建演示显示"""
        canvas = np.zeros((900, 1400, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(canvas, "3D Reconstruction System Demo", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # 左侧：模拟左相机
        left_demo = self._create_camera_view("Left Camera", frame_count, (0, 128, 255))
        canvas[100:400, 50:450] = left_demo
        
        # 右侧：模拟右相机  
        right_demo = self._create_camera_view("Right Camera", frame_count + 0.1, (255, 128, 0))
        canvas[100:400, 500:900] = right_demo
        
        # 下方左：深度信息
        depth_info = self._create_depth_info(reconstruction_data)
        canvas[450:750, 50:450] = depth_info
        
        # 下方右：3D点云可视化
        pointcloud_viz = self._create_advanced_pointcloud_display(reconstruction_data)
        canvas[450:750, 500:900] = pointcloud_viz
        
        # 右侧面板：系统信息
        system_info = self._create_system_info(reconstruction_data, frame_count)
        canvas[100:750, 950:1350] = system_info
        
        return canvas
    
    def _create_camera_view(self, title, time_offset, color):
        """创建模拟相机视图"""
        view = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(view, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 模拟动态场景
        t = time_offset
        
        # 绘制一些移动的几何形状
        center_x = int(200 + 100 * np.cos(t))
        center_y = int(150 + 50 * np.sin(t * 1.2))
        cv2.circle(view, (center_x, center_y), 30, color, -1)
        
        # 绘制网格表示场景结构
        for i in range(0, 400, 50):
            cv2.line(view, (i, 50), (i, 290), (80, 80, 80), 1)
        for i in range(50, 300, 40):
            cv2.line(view, (0, i), (400, i), (80, 80, 80), 1)
        
        # 绘制特征点
        for _ in range(20):
            x = int(np.random.rand() * 400)
            y = int(50 + np.random.rand() * 240)
            cv2.circle(view, (x, y), 2, (0, 255, 0), -1)
        
        return view
    
    def _create_depth_info(self, reconstruction_data):
        """创建深度信息显示"""
        info_view = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(info_view, "3D Reconstruction Info", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if reconstruction_data:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            # 统计信息
            info_lines = [
                f"Points: {len(points)}",
                f"Type: {reconstruction_data['type']}",
                f"Status: ACTIVE",
                "",
                f"X range: {points[:, 0].min():.2f} ~ {points[:, 0].max():.2f}",
                f"Y range: {points[:, 1].min():.2f} ~ {points[:, 1].max():.2f}",
                f"Z range: {points[:, 2].min():.2f} ~ {points[:, 2].max():.2f}",
                "",
                "Features:",
                "- Gaussian Splatting",
                "- Real-time Updates",
                "- Color Mapping"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 60 + i * 20
                if y_pos > 280:
                    break
                color = (0, 255, 255) if ":" in line else (200, 200, 200)
                cv2.putText(info_view, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(info_view, "Status: NO DATA", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return info_view
    
    def _create_advanced_pointcloud_display(self, reconstruction_data):
        """创建高级点云显示"""
        display = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(display, "3D Point Cloud (Top View)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if reconstruction_data and reconstruction_data.get('points') is not None:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                # 3D到2D投影（俯视图 XZ）
                x_coords = points[:, 0]
                z_coords = points[:, 2]
                
                # 归一化坐标
                if len(x_coords) > 0:
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    z_min, z_max = np.min(z_coords), np.max(z_coords)
                    
                    if x_max > x_min and z_max > z_min:
                        x_norm = ((x_coords - x_min) / (x_max - x_min) * 350 + 25).astype(int)
                        z_norm = ((z_coords - z_min) / (z_max - z_min) * 250 + 40).astype(int)
                        
                        # 绘制点云（按密度采样）
                        sample_step = max(1, len(points) // 800)
                        for i in range(0, len(x_norm), sample_step):
                            if 0 <= x_norm[i] < 400 and 0 <= z_norm[i] < 300:
                                color = colors[i] if i < len(colors) else [255, 255, 255]
                                # 根据Y坐标调整点的大小
                                y_val = points[i, 1] if i < len(points) else 0
                                point_size = max(1, int(3 + y_val))
                                cv2.circle(display, (x_norm[i], z_norm[i]), point_size,
                                          (int(color[0]), int(color[1]), int(color[2])), -1)
                
                # 绘制坐标轴
                cv2.arrowedLine(display, (25, 280), (75, 280), (0, 0, 255), 2)  # X轴-红
                cv2.arrowedLine(display, (25, 280), (25, 230), (0, 255, 0), 2)  # Z轴-绿
                cv2.putText(display, "X", (80, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(display, "Z", (30, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display
    
    def _create_system_info(self, reconstruction_data, frame_count):
        """创建系统信息面板"""
        info_panel = np.zeros((650, 400, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(info_panel, "System Status", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 系统信息
        info_data = [
            ("Frame Count", str(frame_count)),
            ("FPS", "20.5"),
            ("Processing", "Real-time"),
            ("Memory Usage", "245MB"),
            ("GPU Usage", "65%"),
            "",
            ("SLAM Status", "ACTIVE"),
            ("Tracking", "GOOD"),
            ("Mapping", "ACTIVE"),
            ("Loop Closure", "READY"),
            "",
            ("MonoGS Backend", "ENABLED"),
            ("EfficientLoFTR", "LOADED"),
            ("PnP Solver", "ACTIVE"),
            ("Visualization", "ENABLED"),
            "",
        ]
        
        if reconstruction_data:
            info_data.extend([
                ("3D Reconstruction", "ACTIVE"),
                ("Point Count", str(reconstruction_data['count'])),
                ("Quality", "HIGH"),
                ("Coverage", "85%"),
            ])
        
        y_pos = 60
        for item in info_data:
            if isinstance(item, tuple) and len(item) == 2:
                label, value = item
            elif item == "":  # 空行
                y_pos += 10
                continue
            else:
                continue
                
            if y_pos > 620:
                break
                
            # 标签
            cv2.putText(info_panel, f"{label}:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 值
            color = (0, 255, 0) if value in ["ACTIVE", "GOOD", "HIGH", "ENABLED", "LOADED", "Real-time"] else (0, 255, 255)
            if value in ["FAIL", "ERROR", "LOW"]:
                color = (0, 0, 255)
                
            cv2.putText(info_panel, value, (150, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            y_pos += 25
        
        return info_panel
    
    def display(self, reconstruction_data, frame_count):
        """显示当前状态"""
        canvas = self.create_demo_display(reconstruction_data, frame_count)
        cv2.imshow(self.window_name, canvas)
        
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def close(self):
        """关闭显示"""
        cv2.destroyWindow(self.window_name)

def main():
    """主函数"""
    print("=" * 60)
    print("3D重建系统演示 - 无需摄像头版本")
    print("=" * 60)
    print("按 'q' 退出")
    print("显示模拟的双目3D重建过程")
    print()
    
    # 初始化组件
    reconstructor = Mock3DReconstructor()
    visualizer = Advanced3DVisualization()
    
    frame_count = 0
    start_time = time.time()
    
    print("开始3D重建演示...")
    
    try:
        while True:
            frame_count += 1
            
            # 处理3D重建
            success = reconstructor.process_frame()
            
            if success:
                # 获取重建数据
                reconstruction_data = reconstructor.get_reconstruction_data()
                
                # 更新显示
                if not visualizer.display(reconstruction_data, frame_count):
                    break
                
                # 性能统计
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    points_count = reconstruction_data['count'] if reconstruction_data else 0
                    print(f"Frame {frame_count}: FPS {fps:.1f}, Points: {points_count}")
            
            time.sleep(0.05)  # 20 FPS
            
    except KeyboardInterrupt:
        print("\n用户中断...")
    except Exception as e:
        print(f"\n演示错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()
        
        # 保存结果
        reconstruction_data = reconstructor.get_reconstruction_data()
        if reconstruction_data:
            save_path = Path("demo_3d_reconstruction.npz")
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            print(f"演示重建结果已保存: {save_path}")
        
        print("演示结束")

if __name__ == "__main__":
    main()