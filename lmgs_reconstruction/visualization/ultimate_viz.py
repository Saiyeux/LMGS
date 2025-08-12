"""
Ultimate Visualization System - 终极可视化系统
整合所有可视化组件的主系统
"""

import cv2
import sys
import select
from .viewer_3d import Interactive3DViewer
from .display_manager import DisplayManager


class UltimateVisualization:
    """终极可视化系统"""
    
    def __init__(self, window_name="Ultimate 3D Reconstruction System", canvas_size=(1600, 1000)):
        """
        初始化可视化系统
        
        Args:
            window_name: 窗口名称
            canvas_size: 画布尺寸
        """
        self.window_name = window_name
        self.canvas_size = canvas_size
        self.headless = False
        
        # 初始化组件
        self.viewer_3d = Interactive3DViewer()
        self.display_manager = DisplayManager(canvas_size)
        
        # 尝试初始化GUI
        self._init_gui()
        
    def _init_gui(self):
        """初始化GUI"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.canvas_size[0], self.canvas_size[1])
        except cv2.error as e:
            print(f"GUI不可用，切换到无头模式: {e}")
            self.headless = True
    
    def display(self, frames, reconstruction_data, is_stereo, is_mock):
        """
        显示系统状态
        
        Args:
            frames: 相机帧字典
            reconstruction_data: 重建数据
            is_stereo: 是否为立体模式
            is_mock: 是否为模拟数据
            
        Returns:
            bool: 是否继续运行
        """
        if self.headless:
            return self._display_headless(reconstruction_data, is_stereo, is_mock)
        
        try:
            # 创建画布
            canvas = self.display_manager.create_canvas()
            
            # 添加标题
            mode_text = "STEREO" if is_stereo else "MONO"
            source_text = "MOCK" if is_mock else "REAL"
            title = f"Ultimate 3D Reconstruction - {mode_text} {source_text}"
            self.display_manager.add_title(title)
            
            # 添加相机视图
            self._add_camera_views(frames, is_mock)
            
            # 添加信息面板
            self.display_manager.add_info_panel(700, 80, 880, 400, reconstruction_data)
            
            # 添加3D视图
            self._add_3d_visualization(reconstruction_data)
            
            # 显示画布
            cv2.imshow(self.window_name, canvas)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q')
            
        except Exception as e:
            print(f"显示错误: {e}")
            return True
    
    def _display_headless(self, reconstruction_data, is_stereo, is_mock):
        """无头模式显示"""
        # 每30帧打印一次状态
        if reconstruction_data and reconstruction_data['frame_count'] % 30 == 0:
            print(f"3D重建状态 - 点数: {reconstruction_data['count']}, "
                  f"帧数: {reconstruction_data['frame_count']}, "
                  f"模式: {'立体' if is_stereo else '单目'} {'模拟' if is_mock else '真实'}")
        
        # 检查键盘输入（简化版）
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
            if key == 'q':
                return False
        
        return True
    
    def _add_camera_views(self, frames, is_mock):
        """添加相机视图"""
        y_offset = 80
        for camera_id, frame in frames.items():
            if frame is not None:
                # 计算可用空间
                available_height = min(420, self.canvas_size[1] - y_offset - 20)
                
                self.display_manager.add_camera_view(
                    frame, camera_id, 20, y_offset, 
                    max_width=640, max_height=available_height, 
                    is_mock=is_mock
                )
                
                y_offset += available_height + 20
                if y_offset > 800:  # 避免超出画布
                    break
    
    def _add_3d_visualization(self, reconstruction_data):
        """添加3D可视化"""
        view_3d = None
        
        if reconstruction_data and reconstruction_data.get('points') is not None:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                try:
                    # 生成3D视图
                    view_3d = self.viewer_3d.update_3d_view(points, colors)
                except Exception as e:
                    print(f"3D渲染错误: {e}")
        
        # 添加3D视图到显示管理器
        self.display_manager.add_3d_view(700, 500, 880, 480, view_3d)
    
    def close(self):
        """关闭显示系统"""
        if not self.headless:
            cv2.destroyWindow(self.window_name)
        
        # 关闭3D查看器
        self.viewer_3d.close()
    
    def is_headless(self):
        """是否为无头模式"""
        return self.headless
    
    def set_window_size(self, width, height):
        """设置窗口大小"""
        self.canvas_size = (width, height)
        self.display_manager = DisplayManager(self.canvas_size)
        
        if not self.headless:
            cv2.resizeWindow(self.window_name, width, height)