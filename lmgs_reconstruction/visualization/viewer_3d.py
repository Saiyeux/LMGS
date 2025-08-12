"""
Interactive 3D Point Cloud Viewer
交互式3D点云查看器
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
import cv2


class Interactive3DViewer:
    """交互式3D点云查看器"""
    
    def __init__(self, figsize=(8, 6)):
        """
        初始化3D查看器
        
        Args:
            figsize: 图像尺寸
        """
        self.fig = None
        self.ax = None
        self.scatter = None
        self.figsize = figsize
        
        # 固定的最佳观察视角
        self.azimuth = 45      # 方位角 
        self.elevation = 30    # 仰角
        
        self.setup_3d_plot()
        
    def setup_3d_plot(self):
        """设置3D绘图环境"""
        try:
            # 创建3D图形
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # 设置标签和标题
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title('Real-time 3D Point Cloud')
            
            # 设置固定视角
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            
            # 设置背景色
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            
            # 设置网格
            self.ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"3D绘图初始化失败: {e}")
            self.fig = None
            self.ax = None
    
    def update_3d_view(self, points, colors):
        """
        更新3D视图
        
        Args:
            points: 3D点数组
            colors: 颜色数组
            
        Returns:
            numpy.ndarray: 渲染的图像 (BGR格式)
        """
        if self.ax is None or len(points) == 0:
            return None
            
        try:
            # 清除之前的点
            self.ax.clear()
            
            # 重新设置标签和标题
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title(f'3D Point Cloud ({len(points)} points)')
            
            # 设置固定视角
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            
            # 绘制点云
            if len(points) > 0:
                sample_points, sample_colors = self._sample_points(points, colors)
                
                # 绘制散点图
                self.ax.scatter(sample_points[:, 0], 
                              sample_points[:, 1], 
                              sample_points[:, 2],
                              c=sample_colors,
                              s=1.5,
                              alpha=0.6)
                
                # 设置固定坐标轴范围
                self._set_fixed_axis_limits()
            
            # 转换为图像
            return self._render_to_image()
            
        except Exception as e:
            print(f"3D视图更新失败: {e}")
            return None
    
    def _sample_points(self, points, colors, max_points=1000):
        """采样点以提高性能"""
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
            sample_colors = colors[indices] if len(colors) > 0 else None
        else:
            sample_points = points
            sample_colors = colors
        
        # 归一化颜色
        if sample_colors is not None and len(sample_colors) > 0:
            sample_colors = sample_colors / 255.0
        else:
            sample_colors = 'blue'
        
        return sample_points, sample_colors
    
    def _set_fixed_axis_limits(self):
        """设置固定的坐标轴范围"""
        # 固定的坐标轴范围 - 适合室内场景重建
        self.ax.set_xlim(-5.0, 5.0)   # X轴范围: -5米到+5米
        self.ax.set_ylim(-5.0, 5.0)   # Y轴范围: -5米到+5米  
        self.ax.set_zlim(0.0, 10.0)   # Z轴范围: 0米到+10米（深度）
    
    def _render_to_image(self):
        """将matplotlib图形渲染为图像"""
        try:
            # 转换为图像
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            size = canvas.get_width_height()
            
            # 处理不同matplotlib版本的API差异
            try:
                raw_data = renderer.tostring_argb()
                # ARGB格式需要转换
                img_array = np.frombuffer(raw_data, dtype=np.uint8)
                img_array = img_array.reshape((size[1], size[0], 4))
                # 去掉alpha通道并转换ARGB到RGB
                img_array = img_array[:, :, 1:4]  # 去掉alpha通道
            except AttributeError:
                try:
                    raw_data = renderer.tobytes()
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    img_array = img_array.reshape((size[1], size[0], 3))
                except AttributeError:
                    # 兼容旧版本matplotlib
                    raw_data = renderer.tostring_rgb()
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    img_array = img_array.reshape((size[1], size[0], 3))
            
            # 转换为BGR格式（OpenCV格式）
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            print(f"图像渲染失败: {e}")
            return None
    
    def set_view_angle(self, azimuth=None, elevation=None):
        """设置视角"""
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            self.elevation = elevation
    
    def close(self):
        """关闭3D查看器"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None