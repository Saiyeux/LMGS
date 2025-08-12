"""
Smart Camera Manager - 智能相机管理器
自动处理相机访问问题，支持真实相机和模拟相机
"""

import cv2
import time
import platform
from .mock_camera import MockCameraGenerator


class SmartCameraManager:
    """智能相机管理器 - 自动处理相机访问问题"""
    
    def __init__(self, max_cameras=5):
        """
        初始化相机管理器
        
        Args:
            max_cameras: 最大搜索相机数量
        """
        self.cameras = {}
        self.use_mock = False
        self.max_cameras = max_cameras
        self.mock_generator = MockCameraGenerator()
        
    def initialize(self):
        """初始化相机系统"""
        print("初始化智能相机系统...")
        
        # 尝试访问真实相机
        success = self._try_real_cameras()
        
        if not success:
            print("真实相机不可用，使用模拟模式")
            self.use_mock = True
            self.mock_generator.initialize()
            return True
        
        return success
    
    def _try_real_cameras(self):
        """尝试访问真实相机"""
        try:
            # 检查更多相机索引以找到可用的相机
            camera_indices_to_try = list(range(self.max_cameras))
            
            for i in camera_indices_to_try:
                try:
                    # 根据操作系统选择合适的后端
                    cap = self._create_camera_capture(i)
                    
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
                    
                except Exception as e:
                    print(f"相机{i}访问异常: {e}")
                    continue
            
            print(f"总共找到 {len(self.cameras)} 个可用相机: {list(self.cameras.keys())}")
            return len(self.cameras) > 0
            
        except Exception as e:
            print(f"相机初始化异常: {e}")
            return False
    
    def _create_camera_capture(self, camera_id):
        """创建相机捕获对象"""
        # Linux使用V4L2，Windows使用DirectShow
        if platform.system() == "Windows":
            return cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            return cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    def get_frames(self):
        """获取帧数据"""
        if self.use_mock:
            return self.mock_generator.get_frames()
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
    
    def is_stereo_mode(self):
        """是否为立体模式"""
        if self.use_mock:
            return True  # 模拟模式总是双目
        return len(self.cameras) >= 2
    
    def get_camera_count(self):
        """获取可用相机数量"""
        if self.use_mock:
            return 2  # 模拟模式提供双目
        return len(self.cameras)
    
    def cleanup(self):
        """清理资源"""
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()
        
        if self.mock_generator:
            self.mock_generator.cleanup()