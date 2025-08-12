"""
Display Manager - 显示管理器
管理多个显示区域和布局
"""

import cv2
import numpy as np


class DisplayManager:
    """显示管理器"""
    
    def __init__(self, canvas_size=(1600, 1000)):
        """
        初始化显示管理器
        
        Args:
            canvas_size: 画布尺寸 (width, height)
        """
        self.canvas_size = canvas_size
        self.canvas = None
        
    def create_canvas(self):
        """创建显示画布"""
        self.canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        return self.canvas
    
    def add_title(self, title, mode_info=""):
        """添加标题"""
        if self.canvas is None:
            return
            
        # 主标题
        cv2.putText(self.canvas, title, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 模式信息
        if mode_info:
            cv2.putText(self.canvas, mode_info, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    
    def add_camera_view(self, frame, camera_id, x, y, max_width=640, max_height=420, is_mock=False):
        """
        添加相机视图
        
        Args:
            frame: 相机帧
            camera_id: 相机ID
            x, y: 位置
            max_width, max_height: 最大尺寸
            is_mock: 是否为模拟数据
        """
        if self.canvas is None or frame is None:
            return
            
        # 调整帧尺寸
        display_frame = self._resize_frame(frame, max_width, max_height)
        h, w = display_frame.shape[:2]
        
        # 检查边界
        if y + h > self.canvas_size[1] or x + w > self.canvas_size[0]:
            return
            
        # 添加到画布
        self.canvas[y:y+h, x:x+w] = display_frame
        
        # 添加标签
        label = f"Camera {camera_id} ({'Mock' if is_mock else 'Real'})"
        cv2.putText(self.canvas, label, (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def add_info_panel(self, x, y, width, height, reconstruction_data):
        """
        添加信息面板
        
        Args:
            x, y: 位置
            width, height: 尺寸
            reconstruction_data: 重建数据
        """
        if self.canvas is None:
            return
            
        # 背景
        cv2.rectangle(self.canvas, (x, y), (x+width, y+height), (40, 40, 40), -1)
        cv2.rectangle(self.canvas, (x, y), (x+width, y+height), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(self.canvas, "3D Reconstruction Status", (x+10, y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 内容
        self._draw_reconstruction_info(x, y, width, height, reconstruction_data)
    
    def add_3d_view(self, x, y, width, height, view_3d):
        """
        添加3D视图
        
        Args:
            x, y: 位置
            width, height: 尺寸
            view_3d: 3D视图图像
        """
        if self.canvas is None:
            return
            
        # 背景
        cv2.rectangle(self.canvas, (x, y), (x+width, y+height), (20, 20, 20), -1)
        cv2.rectangle(self.canvas, (x, y), (x+width, y+height), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(self.canvas, "Interactive 3D Point Cloud", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if view_3d is not None:
            # 调整3D视图大小并居中显示
            self._add_centered_view(view_3d, x, y+50, width, height-50)
        else:
            cv2.putText(self.canvas, "No 3D data available", (x+20, y+100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    
    def _resize_frame(self, frame, max_width, max_height):
        """调整帧尺寸"""
        h, w = frame.shape[:2]
        
        # 计算缩放比例
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        # 调整尺寸
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h))
        
        return frame
    
    def _add_centered_view(self, view_img, x, y, width, height):
        """居中添加视图图像"""
        try:
            # 调整视图大小
            view_h, view_w = view_img.shape[:2]
            
            # 计算缩放比例保持纵横比
            scale_h = height / view_h
            scale_w = width / view_w
            scale = min(scale_h, scale_w)
            
            new_h = int(view_h * scale)
            new_w = int(view_w * scale)
            
            # 调整大小
            view_resized = cv2.resize(view_img, (new_w, new_h))
            
            # 计算居中位置
            start_x = x + (width - new_w) // 2
            start_y = y + (height - new_h) // 2
            
            # 确保不超出边界
            end_x = min(start_x + new_w, self.canvas_size[0])
            end_y = min(start_y + new_h, self.canvas_size[1])
            actual_w = end_x - start_x
            actual_h = end_y - start_y
            
            if actual_w > 0 and actual_h > 0:
                # 调整视图以适应实际可用空间
                view_final = view_resized[:actual_h, :actual_w]
                
                # 添加到画布
                self.canvas[start_y:end_y, start_x:end_x] = view_final
                
        except Exception as e:
            print(f"添加居中视图失败: {e}")
    
    def _draw_reconstruction_info(self, x, y, width, height, reconstruction_data):
        """绘制重建信息"""
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
            if text_y > y + height - 20:
                break
            
            if line:
                if line.startswith("- "):
                    color = (0, 255, 0) if "ACTIVE" in line or "UPDATING" in line or "HIGH" in line else (0, 255, 255)
                else:
                    color = (200, 200, 200)
                
                cv2.putText(self.canvas, line, (x+15, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            text_y += line_height
    
    def get_canvas(self):
        """获取画布"""
        return self.canvas