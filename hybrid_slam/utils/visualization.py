"""
实时3D重建可视化工具
支持立体SLAM的多窗口可视化界面
"""

import cv2
import time
import numpy as np
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available, using matplotlib for 3D visualization")

class RealtimeVisualizer:
    """实时3D重建可视化器"""
    
    def __init__(self, window_size: Tuple[int, int] = (1200, 800),
                 show_trajectory: bool = True, show_pointcloud: bool = True,
                 max_trajectory_points: int = 1000, 
                 save_mode: bool = False, save_dir: str = "visualization_output",
                 save_frequency: int = 1, **kwargs):
        """
        初始化可视化器
        
        Args:
            window_size: 窗口大小 (width, height)
            show_trajectory: 是否显示轨迹
            show_pointcloud: 是否显示点云
            max_trajectory_points: 最大轨迹点数
            save_mode: 保存模式而不是实时显示
            save_dir: 保存目录
            save_frequency: 保存频率（每N帧保存一次）
        """
        self.window_size = window_size
        self.show_trajectory = show_trajectory
        self.show_pointcloud = show_pointcloud
        self.max_trajectory_points = max_trajectory_points
        
        # 保存模式参数
        self.save_mode = save_mode
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.frame_counter = 0
        
        # 可视化数据
        self.trajectory_points = deque(maxlen=max_trajectory_points)
        self.current_pose = np.eye(4)
        self.pointcloud = None
        self.reconstruction_data = None  # 3D重建数据（高斯点云）
        self.left_image = None
        self.right_image = None
        self.depth_map = None
        
        # 统计信息
        self.tracking_info = {}
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # 可视化状态
        self.is_running = True
        self.update_lock = threading.Lock()
        
        # 初始化可视化组件
        self._init_visualization()
        
        # 创建保存目录
        if self.save_mode:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualization save mode enabled: {self.save_dir}")
    
    def _init_visualization(self):
        """初始化可视化组件"""
        if self.save_mode:
            print("Initializing save-mode visualization...")
            # 保存模式不需要创建窗口
        else:
            print("Initializing real-time visualization...")
            # 创建统一的OpenCV窗口
            cv2.namedWindow('Hybrid SLAM - Real-time Visualization', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Hybrid SLAM - Real-time Visualization', 1200, 600)
    
    def _init_open3d_viewer(self):
        """初始化Open3D可视化器"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Hybrid SLAM - 3D Reconstruction", 
                              width=self.window_size[0], height=self.window_size[1])
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 2.0
        render_option.line_width = 2.0
        
        # 创建坐标系
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(self.coordinate_frame)
        
        # 初始化轨迹线
        if self.show_trajectory:
            self.trajectory_line = o3d.geometry.LineSet()
            self.vis.add_geometry(self.trajectory_line)
        
        # 初始化点云
        if self.show_pointcloud:
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
    
    def _init_matplotlib_viewer(self):
        """初始化Matplotlib 3D可视化"""
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Hybrid SLAM - 3D Reconstruction')
        
        # 设置坐标轴范围
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-2, 5)
    
    def update(self, vis_data: Dict[str, Any]):
        """更新可视化数据"""
        with self.update_lock:
            # 更新图像数据
            if 'left_image' in vis_data:
                self.left_image = vis_data['left_image']
            if 'right_image' in vis_data:
                self.right_image = vis_data['right_image']
            if 'depth_map' in vis_data:
                self.depth_map = vis_data['depth_map']
            
            # 更新位姿和轨迹
            if 'current_pose' in vis_data:
                self.current_pose = vis_data['current_pose'].cpu().numpy() if hasattr(vis_data['current_pose'], 'cpu') else vis_data['current_pose']
                position = self.current_pose[:3, 3]
                self.trajectory_points.append(position.copy())
            
            # 更新轨迹数据
            if 'trajectory' in vis_data:
                self.trajectory_points.clear()
                for traj_point in vis_data['trajectory'][-self.max_trajectory_points:]:
                    pose = traj_point['pose'].cpu().numpy() if hasattr(traj_point['pose'], 'cpu') else traj_point['pose']
                    self.trajectory_points.append(pose[:3, 3])
            
            # 更新跟踪信息
            if 'tracking_info' in vis_data:
                self.tracking_info = vis_data['tracking_info']
            
            # 更新点云（如果有）
            if 'pointcloud' in vis_data:
                self.pointcloud = vis_data['pointcloud']
            
            # 更新3D重建数据（高斯点云）
            if '3d_reconstruction' in vis_data:
                self.reconstruction_data = vis_data['3d_reconstruction']
    
    def render(self):
        """渲染可视化界面"""
        if not self.is_running:
            return
        
        with self.update_lock:
            # 渲染2D图像
            self._render_images()
            
            # 暂时跳过3D渲染
            
            # 更新FPS计数
            self._update_fps()
    
    def _render_images(self):
        """渲染2D图像 - 统一界面布局"""
        # 创建统一的显示画布
        canvas_height = 600
        canvas_width = 1200
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 定义四个区域的位置和大小
        panel_width = 300
        panel_height = 240
        
        # 区域1：左图像 (左上)
        if self.left_image is not None:
            left_resized = cv2.resize(self.left_image, (panel_width, panel_height))
            self._add_image_overlay(left_resized, "Left Camera")
            canvas[10:panel_height+10, 10:panel_width+10] = left_resized
        else:
            # 显示占位符
            cv2.rectangle(canvas, (10, 10), (panel_width+10, panel_height+10), (50, 50, 50), -1)
            cv2.putText(canvas, "No Left Image", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 区域2：右图像 (右上)
        if self.right_image is not None:
            right_resized = cv2.resize(self.right_image, (panel_width, panel_height))
            self._add_image_overlay(right_resized, "Right Camera")
            canvas[10:panel_height+10, panel_width+30:panel_width*2+30] = right_resized
        else:
            cv2.rectangle(canvas, (panel_width+30, 10), (panel_width*2+30, panel_height+10), (50, 50, 50), -1)
            cv2.putText(canvas, "No Right Image", (panel_width+60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 区域3：深度图 (左下)
        if self.depth_map is not None:
            depth_display = self._colorize_depth_map(self.depth_map)
            depth_resized = cv2.resize(depth_display, (panel_width, panel_height))
            self._add_image_overlay(depth_resized, "Depth Map")
            canvas[panel_height+30:panel_height*2+30, 10:panel_width+10] = depth_resized
        else:
            cv2.rectangle(canvas, (10, panel_height+30), (panel_width+10, panel_height*2+30), (50, 50, 50), -1)
            cv2.putText(canvas, "No Depth Data", (60, panel_height+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 区域4：跟踪信息和3D重建预览 (右下)
        info_panel = self._create_tracking_info_image(panel_width, panel_height)
        canvas[panel_height+30:panel_height*2+30, panel_width+30:panel_width*2+30] = info_panel
        
        # 添加分割线
        cv2.line(canvas, (panel_width+20, 0), (panel_width+20, canvas_height), (100, 100, 100), 2)
        cv2.line(canvas, (0, panel_height+20), (canvas_width, panel_height+20), (100, 100, 100), 2)
        
        # 在右侧显示3D轨迹预览
        if len(self.trajectory_points) > 1:
            self._draw_trajectory_preview(canvas, panel_width*2+50, 10, canvas_width-panel_width*2-60, panel_height*2+20)
        
        # 显示或保存统一界面
        if self.save_mode:
            # 保存模式：定期保存图像
            if self.frame_counter % self.save_frequency == 0:
                filename = self.save_dir / f"visualization_frame_{self.frame_counter:06d}.png"
                cv2.imwrite(str(filename), canvas)
                if self.frame_counter % (self.save_frequency * 10) == 0:  # 每10次保存时打印一次
                    print(f"Saved visualization: {filename}")
            self.frame_counter += 1
        else:
            # 实时显示模式
            cv2.imshow('Hybrid SLAM - Real-time Visualization', canvas)
            # 关键修复：添加waitKey调用以处理窗口事件
            cv2.waitKey(1)
    
    def _add_image_overlay(self, image: np.ndarray, title: str):
        """在图像上添加文字覆盖"""
        # 添加标题
        cv2.putText(image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 添加FPS信息
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", 
                   (image.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _colorize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """将深度图着色为可视化图像"""
        # 归一化深度值
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def _create_tracking_info_image(self, width: int = 400, height: int = 200) -> np.ndarray:
        """创建跟踪信息显示图像"""
        info_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 30
        line_height = 25
        
        # 显示跟踪信息
        if self.tracking_info:
            info_lines = [
                f"Method: {self.tracking_info.get('method', 'N/A')}",
                f"Confidence: {self.tracking_info.get('confidence', 0.0):.3f}",
                f"Matches: {self.tracking_info.get('num_matches', 0)}",
                f"Proc Time: {self.tracking_info.get('processing_time', 0.0):.1f}ms"
            ]
        else:
            info_lines = ["No tracking info available"]
        
        # 添加轨迹统计
        info_lines.extend([
            f"Trajectory Points: {len(self.trajectory_points)}",
            f"Current FPS: {self.current_fps:.1f}"
        ])
        
        for i, line in enumerate(info_lines):
            cv2.putText(info_img, line, (10, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return info_img
    
    def _render_open3d(self):
        """使用Open3D渲染3D场景"""
        if not hasattr(self, 'vis'):
            return
        
        # 更新轨迹
        if self.show_trajectory and len(self.trajectory_points) > 1:
            trajectory_array = np.array(list(self.trajectory_points))
            
            # 创建轨迹线
            lines = [[i, i + 1] for i in range(len(trajectory_array) - 1)]
            colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色轨迹
            
            self.trajectory_line.points = o3d.utility.Vector3dVector(trajectory_array)
            self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
            self.trajectory_line.colors = o3d.utility.Vector3dVector(colors)
            
            self.vis.update_geometry(self.trajectory_line)
        
        # 更新当前位姿（相机坐标系）
        if hasattr(self, 'current_camera_frame'):
            self.vis.remove_geometry(self.current_camera_frame)
        
        self.current_camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.current_camera_frame.transform(self.current_pose)
        self.vis.add_geometry(self.current_camera_frame)
        
        # 更新点云
        if self.show_pointcloud and self.pointcloud is not None:
            if hasattr(self.pointcloud, 'shape') and len(self.pointcloud) > 0:
                self.pcd.points = o3d.utility.Vector3dVector(self.pointcloud[:, :3])
                if self.pointcloud.shape[1] >= 6:  # 包含颜色信息
                    self.pcd.colors = o3d.utility.Vector3dVector(self.pointcloud[:, 3:6] / 255.0)
                else:
                    # 使用深度着色
                    depths = self.pointcloud[:, 2]
                    colors = plt.cm.jet(depths / depths.max())[:, :3]
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                
                self.vis.update_geometry(self.pcd)
        
        # 更新视图
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _render_matplotlib(self):
        """使用Matplotlib渲染3D场景"""
        if not hasattr(self, 'ax'):
            return
        
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Hybrid SLAM - 3D Reconstruction (FPS: {self.current_fps:.1f})')
        
        # 绘制轨迹
        if self.show_trajectory and len(self.trajectory_points) > 1:
            trajectory_array = np.array(list(self.trajectory_points))
            self.ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                        'r-', linewidth=2, label='Trajectory')
            
            # 绘制当前位置
            current_pos = self.current_pose[:3, 3]
            self.ax.scatter(current_pos[0], current_pos[1], current_pos[2], 
                           c='red', s=50, marker='o', label='Current Position')
        
        # 绘制点云
        if self.show_pointcloud and self.pointcloud is not None:
            if hasattr(self.pointcloud, 'shape') and len(self.pointcloud) > 0:
                points = self.pointcloud[:, :3]
                # 采样显示（避免太多点导致卡顿）
                if len(points) > 1000:
                    indices = np.random.choice(len(points), 1000, replace=False)
                    points = points[indices]
                
                self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=points[:, 2], cmap='jet', s=1, alpha=0.6)
        
        # 绘制坐标系
        origin = np.array([0, 0, 0])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      x_axis[0], x_axis[1], x_axis[2], color='red', length=1)
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      y_axis[0], y_axis[1], y_axis[2], color='green', length=1)
        self.ax.quiver(origin[0], origin[1], origin[2], 
                      z_axis[0], z_axis[1], z_axis[2], color='blue', length=1)
        
        # 设置坐标轴范围（动态调整）
        if len(self.trajectory_points) > 0:
            trajectory_array = np.array(list(self.trajectory_points))
            margin = 2.0
            self.ax.set_xlim(trajectory_array[:, 0].min() - margin, 
                            trajectory_array[:, 0].max() + margin)
            self.ax.set_ylim(trajectory_array[:, 1].min() - margin, 
                            trajectory_array[:, 1].max() + margin)
            self.ax.set_zlim(trajectory_array[:, 2].min() - margin, 
                            trajectory_array[:, 2].max() + margin)
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)
    
    def _draw_trajectory_preview(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """在画布上绘制2D轨迹预览"""
        if len(self.trajectory_points) < 2:
            return
        
        # 获取轨迹点
        points = np.array(list(self.trajectory_points))
        
        # 投影到XZ平面（鸟瞰图）
        x_coords = points[:, 0]  # X坐标
        z_coords = points[:, 2]  # Z坐标（深度）
        
        # 归一化坐标到显示区域
        if len(x_coords) > 1:
            x_range = x_coords.max() - x_coords.min() + 0.1  # 避免除零
            z_range = z_coords.max() - z_coords.min() + 0.1
            
            # 归一化并转换到像素坐标
            x_normalized = ((x_coords - x_coords.min()) / x_range * (width - 40) + 20).astype(int)
            z_normalized = ((z_coords - z_coords.min()) / z_range * (height - 40) + 20).astype(int)
            
            # 绘制轨迹线
            for i in range(len(x_normalized) - 1):
                pt1 = (x + x_normalized[i], y + z_normalized[i])
                pt2 = (x + x_normalized[i + 1], y + z_normalized[i + 1])
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)  # 绿色轨迹线
            
            # 绘制当前位置
            if len(x_normalized) > 0:
                current_pos = (x + x_normalized[-1], y + z_normalized[-1])
                cv2.circle(canvas, current_pos, 5, (0, 0, 255), -1)  # 红色当前位置
        
        # 添加标题和坐标轴标签
        cv2.putText(canvas, "Trajectory (Top View)", (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, "X", (x + width - 20, y + height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "Z", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制边框
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 2)
    
    def _update_fps(self):
        """更新FPS计数"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:  # 每秒更新一次FPS
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def save_screenshot(self, save_path: str):
        """保存当前可视化截图"""
        if OPEN3D_AVAILABLE and hasattr(self, 'vis'):
            self.vis.capture_screen_image(save_path)
        else:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def close(self):
        """关闭可视化器"""
        self.is_running = False
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭Open3D或Matplotlib
        if OPEN3D_AVAILABLE and hasattr(self, 'vis'):
            self.vis.destroy_window()
        elif hasattr(self, 'fig'):
            plt.close(self.fig)

# 工具函数
def visualize_matches(img0: np.ndarray, img1: np.ndarray, matches: Dict[str, Any], 
                     save_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    可视化特征匹配结果
    
    Args:
        img0: 参考帧图像
        img1: 当前帧图像  
        matches: 匹配结果字典
        save_path: 保存路径
        
    Returns:
        匹配可视化图像
    """
    if matches['num_matches'] == 0:
        print("No matches to visualize")
        return None
    
    # 创建并排显示的图像
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    
    if len(img0.shape) == 3:
        vis_img = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
        vis_img[:h0, :w0] = img0
        vis_img[:h1, w0:] = img1
    else:
        vis_img = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
        vis_img[:h0, :w0] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        vis_img[:h1, w0:] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    # 绘制匹配点和连线
    kpts0 = matches['keypoints0']
    kpts1 = matches['keypoints1']
    confidence = matches.get('confidence', np.ones(len(kpts0)))
    
    for i, (pt0, pt1, conf) in enumerate(zip(kpts0, kpts1, confidence)):
        # 根据置信度设置颜色
        color = (int(255 * conf), int(255 * (1-conf)), 0)
        
        # 绘制关键点
        cv2.circle(vis_img, (int(pt0[0]), int(pt0[1])), 3, color, -1)
        cv2.circle(vis_img, (int(pt1[0] + w0), int(pt1[1])), 3, color, -1)
        
        # 绘制连线
        cv2.line(vis_img, (int(pt0[0]), int(pt0[1])), 
                (int(pt1[0] + w0), int(pt1[1])), color, 1)
    
    # 添加信息文本
    cv2.putText(vis_img, f"Matches: {matches['num_matches']}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img

def plot_trajectory(poses: List[np.ndarray], gt_poses: Optional[List[np.ndarray]] = None,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制轨迹对比图
    
    Args:
        poses: 估计的位姿列表
        gt_poses: 真值位姿列表
        save_path: 保存路径
        
    Returns:
        fig: matplotlib图形对象
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 提取位置信息
    if len(poses) > 0:
        estimated_positions = np.array([pose[:3, 3] for pose in poses])
        
        # 2D轨迹图 (X-Y平面)
        ax1 = fig.add_subplot(221)
        ax1.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b-', label='Estimated')
        if gt_poses is not None:
            gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
            ax1.plot(gt_positions[:, 0], gt_positions[:, 1], 'r-', label='Ground Truth')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trajectory (Top View)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # 3D轨迹图
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot(estimated_positions[:, 0], estimated_positions[:, 1], 
                estimated_positions[:, 2], 'b-', label='Estimated')
        if gt_poses is not None:
            ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 
                    gt_positions[:, 2], 'r-', label='Ground Truth')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('3D Trajectory')
        ax2.legend()
        
        # X-Z侧视图
        ax3 = fig.add_subplot(223)
        ax3.plot(estimated_positions[:, 0], estimated_positions[:, 2], 'b-', label='Estimated')
        if gt_poses is not None:
            ax3.plot(gt_positions[:, 0], gt_positions[:, 2], 'r-', label='Ground Truth')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Trajectory (Side View)')
        ax3.legend()
        ax3.grid(True)
        
        # 误差分析（如果有真值）
        ax4 = fig.add_subplot(224)
        if gt_poses is not None and len(gt_poses) == len(poses):
            errors = np.linalg.norm(estimated_positions - gt_positions, axis=1)
            ax4.plot(errors, 'g-')
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Position Error (m)')
            ax4.set_title(f'Position Error (RMSE: {np.sqrt(np.mean(errors**2)):.3f}m)')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No Ground Truth Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Error Analysis')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_performance_metrics(metrics: Dict[str, List[float]], 
                                save_path: Optional[str] = None) -> plt.Figure:
    """可视化性能指标"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i >= 4:  # 最多显示4个指标
            break
        
        ax = axes[i // 2, i % 2]
        ax.plot(values)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # 添加统计信息
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_val:.3f}')
        ax.fill_between(range(len(values)), 
                       mean_val - std_val, mean_val + std_val, 
                       alpha=0.2, color='gray', label=f'±1σ: {std_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_3d_reconstruction_visualization(points, colors=None, title="3D Reconstruction"):
    """创建独立的3D重建可视化窗口"""
    if OPEN3D_AVAILABLE:
        # 使用Open3D创建高质量可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if colors is not None:
            if colors.max() > 1.0:  # 0-255范围
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        o3d.visualization.draw_geometries([pcd], window_name=title)
    else:
        # 使用matplotlib作为备选
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors[:, :3] if len(colors.shape) > 1 else colors, s=0.1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class RealtimeVisualizerExtended(RealtimeVisualizer):
    """扩展的可视化器，支持3D高斯点云渲染"""
    
    def _render_3d_reconstruction(self, canvas: np.ndarray, x: int, y: int, width: int, height: int):
        """在canvas上渲染3D重建预览"""
        if self.reconstruction_data is None:
            # 绘制占位符
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (30, 30, 30), -1)
            cv2.putText(canvas, "3D Reconstruction", (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, "No data available", (x + 10, y + height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return
        
        # 创建3D重建的2D投影预览
        recon_preview = self._create_3d_reconstruction_preview(width, height)
        if recon_preview is not None:
            canvas[y:y+height, x:x+width] = recon_preview

    def _create_3d_reconstruction_preview(self, width: int, height: int) -> np.ndarray:
        """创建3D重建的2D预览图"""
        if self.reconstruction_data is None:
            return None
        
        try:
            points = self.reconstruction_data.get('points')
            colors = self.reconstruction_data.get('colors')
            recon_type = self.reconstruction_data.get('type', 'unknown')
            
            if points is None or len(points) == 0:
                return None
            
            # 创建预览图
            preview = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 计算投影 - 简单的正交投影到XY平面
            if len(points.shape) == 2 and points.shape[1] >= 3:
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                z_coords = points[:, 2]
                
                # 归一化坐标到图像坐标
                if len(x_coords) > 0:
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    
                    if x_max > x_min and y_max > y_min:
                        x_norm = ((x_coords - x_min) / (x_max - x_min) * (width - 20) + 10).astype(int)
                        y_norm = ((y_coords - y_min) / (y_max - y_min) * (height - 20) + 10).astype(int)
                        
                        # 使用深度信息进行颜色编码
                        z_min, z_max = np.min(z_coords), np.max(z_coords)
                        if z_max > z_min:
                            z_norm = (z_coords - z_min) / (z_max - z_min)
                        else:
                            z_norm = np.ones_like(z_coords) * 0.5
                        
                        # 绘制点
                        for i in range(min(len(x_norm), 5000)):  # 限制点数以提高性能
                            if 0 <= x_norm[i] < width and 0 <= y_norm[i] < height:
                                if colors is not None and len(colors) > i:
                                    # 使用原始颜色
                                    if recon_type == 'gaussian_splatting':
                                        # MonoGS高斯点云颜色处理
                                        color = colors[i] if len(colors.shape) > 1 else [128, 128, 128]
                                    else:
                                        # 传统立体视觉颜色
                                        color = colors[i]
                                    
                                    if hasattr(color, '__len__') and len(color) >= 3:
                                        b, g, r = int(color[2]), int(color[1]), int(color[0])
                                    else:
                                        b = g = r = int(color) if np.isscalar(color) else 128
                                else:
                                    # 使用深度着色
                                    intensity = int(z_norm[i] * 255)
                                    b, g, r = intensity, intensity, intensity
                                
                                # 限制颜色值范围
                                b = max(0, min(255, b))
                                g = max(0, min(255, g))
                                r = max(0, min(255, r))
                                
                                cv2.circle(preview, (x_norm[i], y_norm[i]), 1, (b, g, r), -1)
            
            # 添加标题和信息
            cv2.putText(preview, f"3D {recon_type}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(preview, f"Points: {len(points)}", (5, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return preview
            
        except Exception as e:
            print(f"Error creating 3D reconstruction preview: {e}")
            return None

    def save_3d_reconstruction(self, filepath: str):
        """保存3D重建数据"""
        if self.reconstruction_data is None:
            print("No 3D reconstruction data to save")
            return
        
        try:
            points = self.reconstruction_data.get('points')
            colors = self.reconstruction_data.get('colors')
            recon_type = self.reconstruction_data.get('type', 'unknown')
            
            if points is None:
                print("No points data in reconstruction")
                return
            
            filepath = Path(filepath)
            
            if filepath.suffix.lower() == '.ply':
                self._save_ply(points, colors, filepath)
            elif filepath.suffix.lower() == '.npz':
                np.savez(filepath, points=points, colors=colors, type=recon_type)
            else:
                print(f"Unsupported file format: {filepath.suffix}")
                
            print(f"3D reconstruction saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving 3D reconstruction: {e}")
    
    def _save_ply(self, points: np.ndarray, colors: np.ndarray, filepath: Path):
        """保存PLY格式点云"""
        with open(filepath, 'w') as f:
            # PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if colors is not None and len(colors) == len(points):
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # 点数据
            for i in range(len(points)):
                x, y, z = points[i][:3]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}")
                
                if colors is not None and len(colors) > i:
                    if hasattr(colors[i], '__len__') and len(colors[i]) >= 3:
                        r, g, b = colors[i][:3]
                        f.write(f" {int(r)} {int(g)} {int(b)}")
                    else:
                        intensity = int(colors[i]) if np.isscalar(colors[i]) else 128
                        f.write(f" {intensity} {intensity} {intensity}")
                else:
                    f.write(" 128 128 128")
                
                f.write("\n")