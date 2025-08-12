"""
Mock Camera Generator - 模拟相机数据生成器
用于在没有真实相机时生成测试数据
"""

import cv2
import numpy as np


class MockCameraGenerator:
    """模拟相机数据生成器"""
    
    def __init__(self, width=640, height=480):
        """
        初始化模拟相机
        
        Args:
            width: 图像宽度
            height: 图像高度
        """
        self.width = width
        self.height = height
        self.frame_count = 0
        
    def initialize(self):
        """初始化模拟相机系统"""
        print("初始化模拟相机系统...")
        self.frame_count = 0
        
    def get_frames(self):
        """生成模拟帧数据"""
        frames = {}
        
        # 生成动态场景
        self.frame_count += 1
        t = self.frame_count * 0.1
        
        for camera_id in [0, 1]:
            frame = self._generate_frame(camera_id, t)
            frames[camera_id] = frame
        
        return frames
    
    def _generate_frame(self, camera_id, t):
        """生成单个相机帧"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 生成动态背景
        frame[:] = (50, 50, 50)
        
        # 添加移动的几何体
        center_x = int(self.width/2 + 200 * np.cos(t + camera_id * 0.1))
        center_y = int(self.height/2 + 100 * np.sin(t * 1.2))
        
        # 主要物体
        cv2.circle(frame, (center_x, center_y), 40, (0, 255, 128), -1)
        cv2.rectangle(frame, 
                     (center_x-60, center_y-60), 
                     (center_x+60, center_y+60), 
                     (255, 128, 0), 3)
        
        # 特征点
        self._add_feature_points(frame, t)
        
        # 网格参考
        self._add_grid_reference(frame)
        
        # 相机标识
        self._add_camera_info(frame, camera_id)
        
        return frame
    
    def _add_feature_points(self, frame, t):
        """添加特征点"""
        for i in range(30):
            x = int(self.width/2 + 250 * np.cos(t + i * 0.2))
            y = int(self.height/2 + 200 * np.sin(t + i * 0.3))
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
    
    def _add_grid_reference(self, frame):
        """添加网格参考线"""
        # 垂直线
        for i in range(0, self.width, 80):
            cv2.line(frame, (i, 0), (i, self.height), (80, 80, 80), 1)
        
        # 水平线
        for i in range(0, self.height, 60):
            cv2.line(frame, (0, i), (self.width, i), (80, 80, 80), 1)
    
    def _add_camera_info(self, frame, camera_id):
        """添加相机信息文本"""
        cv2.putText(frame, f"Camera {camera_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    def cleanup(self):
        """清理资源"""
        pass  # 模拟相机无需清理