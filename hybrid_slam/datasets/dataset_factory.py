"""
数据集工厂 - 创建各种类型的数据源
支持双摄像头实时输入、离线数据集等
"""

import cv2
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterator, List
from dataclasses import dataclass
import threading
from queue import Queue, Empty

from ..utils.data_structures import StereoFrame

@dataclass
class StereoCalibration:
    """立体相机标定参数"""
    K_left: np.ndarray      # 左相机内参矩阵
    K_right: np.ndarray     # 右相机内参矩阵
    D_left: np.ndarray      # 左相机畸变系数
    D_right: np.ndarray     # 右相机畸变系数
    R: np.ndarray           # 旋转矩阵
    T: np.ndarray           # 平移向量
    baseline: float         # 基线距离

class StereoCameraDataset:
    """双摄像头实时数据源"""
    
    def __init__(self, left_device: int = 0, right_device: int = 1, 
                 resolution: Tuple[int, int] = (640, 480), fps: int = 30,
                 calibration_file: Optional[str] = None):
        """
        初始化双摄像头数据源
        
        Args:
            left_device: 左摄像头设备ID
            right_device: 右摄像头设备ID 
            resolution: 图像分辨率 (width, height)
            fps: 目标帧率
            calibration_file: 标定文件路径
        """
        self.left_device = left_device
        self.right_device = right_device
        self.resolution = resolution
        self.fps = fps
        self.calibration_file = calibration_file
        
        # 相机对象
        self.left_cap = None
        self.right_cap = None
        
        # 标定参数
        self.calibration = None
        
        # 帧缓冲
        self.frame_buffer = Queue(maxsize=10)
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
        # 状态
        self.frame_id = 0
        self.start_time = time.time()
        self.is_initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """初始化摄像头和标定参数"""
        try:
            # 初始化摄像头
            self._init_cameras()
            
            # 加载标定参数
            if self.calibration_file:
                self._load_calibration()
            else:
                self._create_default_calibration()
            
            # 启动采集线程
            self._start_capture_thread()
            
            self.is_initialized = True
            print(f"StereoCameraDataset initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize StereoCameraDataset: {e}")
            self.close()
            raise
    
    def _init_cameras(self):
        """初始化左右摄像头 - 采用cam.py的初始化策略"""
        camera_indices = [self.left_device, self.right_device]
        cameras = []
        valid_cameras = []
        
        # 遍历摄像头索引列表并初始化 - 完全模仿cam.py
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # 设置分辨率 - 使用cam.py相同的设置顺序
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
                cameras.append(cap)
                valid_cameras.append(idx)
                print(f"成功打开摄像头 {idx}")
            else:
                print(f"无法打开摄像头 {idx}")
                cap.release()
        
        # 检查是否有可用的摄像头
        if len(cameras) != 2:
            error_msg = f"需要2个摄像头，但只成功打开了 {len(cameras)} 个: {valid_cameras}"
            for cap in cameras:
                cap.release()
            raise RuntimeError(error_msg)
        
        # 按照期望的顺序分配
        self.left_cap = cameras[0]    # 第一个成功的摄像头作为左摄像头
        self.right_cap = cameras[1]   # 第二个成功的摄像头作为右摄像头
        
        print(f"总共打开了 {len(cameras)} 个摄像头: {valid_cameras}")
        print(f"左摄像头: {valid_cameras[0]}, 右摄像头: {valid_cameras[1]}")
        
        # 获取实际的分辨率和帧率
        actual_width = int(self.left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.left_cap.get(cv2.CAP_PROP_FPS)
        
        print(f"实际分辨率: {actual_width}x{actual_height}, 帧率: {actual_fps}")
        
        # 更新分辨率（如果实际值不同）
        if (actual_width, actual_height) != self.resolution:
            print(f"分辨率已调整为: {actual_width}x{actual_height}")
            self.resolution = (actual_width, actual_height)
    
    def _load_calibration(self):
        """从文件加载标定参数"""
        calib_path = Path(self.calibration_file)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        try:
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            
            self.calibration = StereoCalibration(
                K_left=np.array(calib_data['K_left']),
                K_right=np.array(calib_data['K_right']),
                D_left=np.array(calib_data['D_left']),
                D_right=np.array(calib_data['D_right']),
                R=np.array(calib_data['R']),
                T=np.array(calib_data['T']),
                baseline=calib_data['baseline']
            )
            
            print(f"Loaded calibration from {calib_path}")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            self._create_default_calibration()
    
    def _create_default_calibration(self):
        """创建默认标定参数"""
        print("Using default calibration parameters")
        
        # 默认内参矩阵（根据分辨率调整）
        fx = fy = self.resolution[0] * 0.8  # 假设焦距约为图像宽度的0.8倍
        cx, cy = self.resolution[0] / 2, self.resolution[1] / 2
        
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)
        
        D = np.zeros(5, dtype=np.float32)  # 假设无畸变
        
        self.calibration = StereoCalibration(
            K_left=K.copy(),
            K_right=K.copy(),
            D_left=D.copy(),
            D_right=D.copy(),
            R=np.eye(3, dtype=np.float32),
            T=np.array([0.1, 0, 0], dtype=np.float32),  # 假设基线10cm
            baseline=0.1
        )
    
    def _start_capture_thread(self):
        """启动图像采集线程"""
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _capture_loop(self):
        """图像采集循环 - 采用cam.py的顺序读取策略"""
        target_interval = 1.0 / self.fps
        
        # 使用cam.py的摄像头管理方式
        cameras = [self.left_cap, self.right_cap]
        valid_cameras = [self.left_device, self.right_device]
        
        print(f"Capture thread started with cameras: {valid_cameras}")
        
        while not self.stop_capture.is_set():
            loop_start = time.time()
            
            try:
                frames = []
                all_success = True
                
                # 从所有摄像头顺序读取帧 - 完全模仿cam.py
                for i, cap in enumerate(cameras):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        # print(f"无法读取摄像头 {valid_cameras[i]} 的帧")  # 减少日志输出
                        all_success = False
                        break
                
                # 如果所有摄像头都成功读取帧 - 模仿cam.py逻辑
                if all_success and len(frames) == 2:
                    # 创建立体帧
                    timestamp = time.time() - self.start_time
                    stereo_frame = StereoFrame(
                        timestamp=timestamp,
                        left_image=frames[0],   # 顺序：第一个是左，第二个是右
                        right_image=frames[1],
                        frame_id=self.frame_id,
                        camera_matrices=(self.calibration.K_left, self.calibration.K_right),
                        baseline=self.calibration.baseline
                    )
                    
                    # 添加到缓冲区
                    try:
                        self.frame_buffer.put(stereo_frame, block=False)
                        self.frame_id += 1
                    except:
                        # 缓冲区满，丢弃最旧的帧
                        try:
                            self.frame_buffer.get(block=False)
                            self.frame_buffer.put(stereo_frame, block=False)
                            self.frame_id += 1
                        except Empty:
                            pass
                else:
                    # 部分摄像头读取失败，短暂等待
                    time.sleep(0.01)
                
                # 帧率控制
                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                    
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def __iter__(self) -> Iterator[StereoFrame]:
        """迭代器接口"""
        return self
    
    def __next__(self) -> StereoFrame:
        """获取下一帧"""
        if not self.is_initialized:
            raise StopIteration
        
        try:
            # 从缓冲区获取帧（阻塞），增加超时时间
            stereo_frame = self.frame_buffer.get(timeout=5.0)
            return stereo_frame
        except Empty:
            # 超时，可能摄像头断开或采集线程有问题
            print("Camera timeout, checking capture thread status...")
            if self.capture_thread and not self.capture_thread.is_alive():
                print("Capture thread died, restarting...")
                try:
                    self._start_capture_thread()
                    stereo_frame = self.frame_buffer.get(timeout=2.0)
                    return stereo_frame
                except:
                    pass
            print("Camera timeout, stopping iteration")
            raise StopIteration
    
    def close(self):
        """关闭数据源"""
        print("Closing StereoCameraDataset...")
        
        # 停止采集线程
        self.stop_capture.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # 关闭摄像头
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        
        # 清空缓冲区
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get(block=False)
            except Empty:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_frames': self.frame_id,
            'buffer_size': self.frame_buffer.qsize(),
            'resolution': self.resolution,
            'fps': self.fps,
            'baseline': self.calibration.baseline if self.calibration else 0.0,
            'runtime': time.time() - self.start_time
        }

class MockStereoDataset:
    """模拟立体数据集（用于测试）"""
    
    def __init__(self, num_frames: int = 1000, resolution: Tuple[int, int] = (640, 480)):
        self.num_frames = num_frames
        self.resolution = resolution
        self.current_frame = 0
        
        # 创建默认标定
        fx = fy = resolution[0] * 0.8
        cx, cy = resolution[0] / 2, resolution[1] / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        self.calibration = StereoCalibration(
            K_left=K.copy(),
            K_right=K.copy(),
            D_left=np.zeros(5),
            D_right=np.zeros(5),
            R=np.eye(3),
            T=np.array([0.1, 0, 0]),
            baseline=0.1
        )
        
        # 相机运动参数
        self.camera_position = np.array([0.0, 0.0, 0.0])
        self.camera_rotation = 0.0
        
        # 生成静态3D场景点（用于生成一致的特征）
        self.scene_points = self._generate_scene_points()
    
    def _generate_scene_points(self):
        """生成3D场景点"""
        np.random.seed(42)  # 固定种子确保一致性
        
        # 创建一个简单的3D场景
        points = []
        
        # 添加一些平面上的点（地面）
        for x in np.linspace(-5, 5, 20):
            for z in np.linspace(2, 10, 15):
                points.append([x, -1.5, z])
        
        # 添加一些立方体（建筑物）
        for i in range(5):
            cx = np.random.uniform(-3, 3)
            cz = np.random.uniform(3, 8)
            cy = np.random.uniform(-1, 1)
            
            # 立方体的8个顶点
            for dx in [-0.5, 0.5]:
                for dy in [-0.5, 0.5]:
                    for dz in [-0.5, 0.5]:
                        points.append([cx + dx, cy + dy, cz + dz])
        
        # 添加一些随机特征点
        for _ in range(100):
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-2, 2)
            z = np.random.uniform(1, 12)
            points.append([x, y, z])
        
        return np.array(points)
    
    def _generate_realistic_images(self):
        """生成更真实的立体图像对"""
        h, w = self.resolution[1], self.resolution[0]
        
        # 更新相机位置（模拟运动）
        t = self.current_frame * 0.033
        self.camera_position = np.array([
            0.2 * np.sin(0.1 * t),  # X方向慢速摆动
            0.1 * np.sin(0.2 * t),  # Y方向小幅摆动
            t * 0.05                # Z方向前进
        ])
        self.camera_rotation = 0.05 * np.sin(0.05 * t)  # 小幅旋转
        
        # 创建基础图像（渐变背景）
        left_img = np.zeros((h, w, 3), dtype=np.uint8)
        right_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 添加天空渐变
        for y in range(h):
            intensity = int(150 + 105 * (y / h))
            left_img[y, :] = [min(255, intensity + 20), min(255, intensity + 50), 255]
            right_img[y, :] = [min(255, intensity + 20), min(255, intensity + 50), 255]
        
        # 投影3D点到图像平面
        left_features = []
        right_features = []
        
        for point_3d in self.scene_points:
            # 将世界坐标转换到相机坐标
            point_cam = point_3d - self.camera_position
            
            # 应用旋转
            cos_r, sin_r = np.cos(self.camera_rotation), np.sin(self.camera_rotation)
            point_cam_rot = np.array([
                cos_r * point_cam[0] - sin_r * point_cam[2],
                point_cam[1],
                sin_r * point_cam[0] + cos_r * point_cam[2]
            ])
            
            # 只处理在相机前方的点
            if point_cam_rot[2] > 0.5:
                # 左相机投影
                x_left = int(self.calibration.K_left[0, 0] * point_cam_rot[0] / point_cam_rot[2] + self.calibration.K_left[0, 2])
                y_left = int(self.calibration.K_left[1, 1] * point_cam_rot[1] / point_cam_rot[2] + self.calibration.K_left[1, 2])
                
                # 右相机投影（考虑基线偏移）
                point_right = point_cam_rot - np.array([self.calibration.baseline, 0, 0])
                x_right = int(self.calibration.K_right[0, 0] * point_right[0] / point_right[2] + self.calibration.K_right[0, 2])
                y_right = int(self.calibration.K_right[1, 1] * point_right[1] / point_right[2] + self.calibration.K_right[1, 2])
                
                # 在有效范围内绘制特征点
                if 0 <= x_left < w and 0 <= y_left < h:
                    self._draw_feature_point(left_img, (x_left, y_left), point_cam_rot[2])
                    left_features.append((x_left, y_left, point_cam_rot[2]))
                
                if 0 <= x_right < w and 0 <= y_right < h:
                    self._draw_feature_point(right_img, (x_right, y_right), point_cam_rot[2])
                    right_features.append((x_right, y_right, point_cam_rot[2]))
        
        # 添加一些纹理和噪声
        self._add_texture(left_img)
        self._add_texture(right_img)
        
        return left_img, right_img, len(left_features), len(right_features)
    
    def _draw_feature_point(self, img, center, depth):
        """在图像上绘制特征点"""
        x, y = center
        
        # 根据深度确定颜色和大小
        intensity = max(50, min(255, int(255 - depth * 20)))
        size = max(1, min(5, int(8 - depth)))
        
        # 绘制彩色特征点
        color = (
            intensity,
            max(0, intensity - 50),
            max(0, intensity - 100)
        )
        
        # 绘制圆形特征点
        cv2.circle(img, (x, y), size, color, -1)
        
        # 添加高亮边缘
        cv2.circle(img, (x, y), size + 1, (255, 255, 255), 1)
    
    def _add_texture(self, img):
        """添加纹理和细节"""
        h, w = img.shape[:2]
        
        # 添加一些水平线条（模拟建筑物边缘）
        for i in range(3):
            y = int(h * (0.3 + i * 0.2))
            cv2.line(img, (0, y), (w, y), (100, 100, 100), 2)
        
        # 添加轻微的随机噪声
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img_noisy = img.astype(np.int16) + noise
        img[:] = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> StereoFrame:
        if self.current_frame >= self.num_frames:
            raise StopIteration
        
        # 生成真实的立体图像
        left_img, right_img, left_features, right_features = self._generate_realistic_images()
        
        # 生成模拟深度图
        depth_map = self._generate_depth_map()
        
        stereo_frame = StereoFrame(
            timestamp=self.current_frame * 0.033,  # 30fps
            left_image=left_img,
            right_image=right_img,
            frame_id=self.current_frame,
            camera_matrices=(self.calibration.K_left, self.calibration.K_right),
            baseline=self.calibration.baseline,
            depth_map=depth_map
        )
        
        # Add num_features to metadata instead
        stereo_frame.metadata['num_features'] = max(left_features, right_features)
        
        self.current_frame += 1
        return stereo_frame
    
    def _generate_depth_map(self):
        """生成模拟深度图"""
        h, w = self.resolution[1], self.resolution[0]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # 创建距离渐变（近处小值，远处大值）
        for y in range(h):
            for x in range(w):
                # 基础深度（向前看）
                base_depth = 2.0 + (y / h) * 8.0
                
                # 添加一些物体形状
                if abs(x - w//2) < w//4 and abs(y - h//2) < h//4:
                    base_depth *= 0.7  # 中央物体更近
                
                depth_map[y, x] = base_depth
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, depth_map.shape)
        depth_map += noise
        depth_map = np.clip(depth_map, 0.5, 15.0)
        
        return depth_map
    
    def close(self):
        pass

class DatasetFactory:
    """数据集工厂类"""
    
    @staticmethod
    def create_dataset(dataset_type: str, config: Dict[str, Any]):
        """
        创建数据集实例
        
        Args:
            dataset_type: 数据集类型 ('stereo_camera', 'tum', 'replica', 'euroc', 'mock')
            config: 数据集配置
            
        Returns:
            dataset: 数据集实例
        """
        if dataset_type == 'stereo_camera':
            return create_stereo_camera_dataset(**config)
        elif dataset_type == 'mock':
            return create_mock_stereo_dataset(**config)
        elif dataset_type == 'tum':
            try:
                from .tum_dataset import TUMStereoDataset
                return TUMStereoDataset(**config)
            except ImportError:
                print("TUM dataset not available, using mock dataset")
                return create_mock_stereo_dataset(**config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    @staticmethod
    def get_supported_datasets() -> List[str]:
        """获取支持的数据集类型"""
        return ['stereo_camera', 'tum', 'replica', 'euroc', 'mock']

def create_stereo_camera_dataset(left_device: int = 0, right_device: int = 1,
                                resolution: Tuple[int, int] = (640, 480),
                                fps: int = 30, calibration_file: Optional[str] = None) -> StereoCameraDataset:
    """创建双摄像头数据集"""
    try:
        return StereoCameraDataset(
            left_device=left_device,
            right_device=right_device,
            resolution=resolution,
            fps=fps,
            calibration_file=calibration_file
        )
    except Exception as e:
        print(f"Failed to create real stereo dataset, using mock dataset: {e}")
        return MockStereoDataset(resolution=resolution)

def create_mock_stereo_dataset(num_frames: int = 1000, 
                              resolution: Tuple[int, int] = (640, 480)) -> MockStereoDataset:
    """创建模拟立体数据集"""
    return MockStereoDataset(num_frames=num_frames, resolution=resolution)