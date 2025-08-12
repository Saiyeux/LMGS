"""
Hybrid Advanced 3D Reconstructor - 混合高级3D重建器
集成EfficientLoFTR和MonoGS的主重建系统
"""

import torch
import numpy as np
import yaml
from pathlib import Path

from ..utils.dependencies import LOFTR_AVAILABLE, MONOGS_AVAILABLE
from .loftr_processor import LoFTRProcessor
from .stereo_processor import StereoProcessor  
from .mono_processor import MonoProcessor


class HybridAdvanced3DReconstructor:
    """混合高级3D重建器 - 集成EfficientLoFTR和MonoGS"""
    
    def __init__(self, device='cpu', camera_config_path='camera_calibration.yaml'):
        """
        初始化重建器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            camera_config_path: 相机参数配置文件路径
        """
        self.points_3d = []
        self.colors_3d = []
        self.frame_count = 0
        
        # 设备配置
        self.device = torch.device(device)
        
        # 加载相机参数
        self.camera_params = self._load_camera_params(camera_config_path)
        
        # 初始化处理器
        self._init_processors()
        
    def _load_camera_params(self, config_path):
        """
        从YAML文件加载相机参数
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 相机参数字典
        """
        # 默认相机参数
        default_params = {
            'fx': 1129.8,
            'fy': 1130.4,
            'cx': 932.5,
            'cy': 542.1,
            'baseline': 0.12
        }
        
        try:
            # 尝试从当前目录或绝对路径加载
            if not Path(config_path).is_absolute():
                config_path = Path.cwd() / config_path
            
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if config and 'camera_matrix' in config:
                    # 从相机矩阵提取内参
                    camera_matrix = np.array(config['camera_matrix'])
                    params = {
                        'fx': float(camera_matrix[0, 0]),
                        'fy': float(camera_matrix[1, 1]),
                        'cx': float(camera_matrix[0, 2]),
                        'cy': float(camera_matrix[1, 2]),
                        'baseline': 0.12  # 立体相机基线，需要根据实际硬件配置
                    }
                    
                    # 如果有图像尺寸信息，更新默认值
                    if 'image_size' in config:
                        params['image_width'] = config['image_size'][0]
                        params['image_height'] = config['image_size'][1]
                    
                    # 如果有畸变参数，也保存
                    if 'distortion_coefficients' in config:
                        params['distortion_coeffs'] = np.array(config['distortion_coefficients']).flatten()
                    
                    print(f"已从 {config_path} 加载相机参数")
                    print(f"fx: {params['fx']:.1f}, fy: {params['fy']:.1f}")
                    print(f"cx: {params['cx']:.1f}, cy: {params['cy']:.1f}")
                    
                    return params
                    
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            print(f"使用默认相机参数")
        
        return default_params
        
    def _init_processors(self):
        """初始化各种处理器"""
        # LoFTR处理器
        if LOFTR_AVAILABLE:
            self.loftr_processor = LoFTRProcessor(
                device=self.device,
                camera_params=self.camera_params
            )
        else:
            self.loftr_processor = None
            
        # 立体视觉处理器
        self.stereo_processor = StereoProcessor(
            camera_params=self.camera_params
        )
        
        # 单目处理器
        self.mono_processor = MonoProcessor(
            camera_params=self.camera_params
        )
        
    def process_frames(self, frames, is_mock=False):
        """
        处理输入帧
        
        Args:
            frames: 相机帧字典 {camera_id: frame}
            is_mock: 是否为模拟数据
            
        Returns:
            bool: 处理是否成功
        """
        self.frame_count += 1
        
        if len(frames) >= 2:
            # 双目重建
            return self._process_stereo_frames(frames, is_mock)
        elif len(frames) == 1:
            # 单目重建
            return self._process_mono_frame(frames, is_mock)
        
        return False
    
    def _process_stereo_frames(self, frames, is_mock):
        """处理立体帧"""
        try:
            # 获取左右图像
            camera_ids = sorted(frames.keys())
            left_img = frames[camera_ids[0]]
            right_img = frames[camera_ids[1]]
            
            # 选择处理方法
            if self.loftr_processor and not is_mock:
                # 使用EfficientLoFTR进行高质量特征匹配
                points_3d, colors_3d = self.loftr_processor.process_stereo_pair(
                    left_img, right_img
                )
            else:
                # 使用传统立体视觉方法
                points_3d, colors_3d = self.stereo_processor.process_stereo_pair(
                    left_img, right_img, is_mock
                )
            
            # 更新点云
            if len(points_3d) > 0:
                self._update_point_cloud(points_3d, colors_3d, is_mock)
                return True
                
        except Exception as e:
            print(f"立体处理失败: {e}")
        
        return False
    
    def _process_mono_frame(self, frames, is_mock):
        """处理单目帧"""
        try:
            camera_id = list(frames.keys())[0]
            img = frames[camera_id]
            
            # 使用单目处理器
            points_3d, colors_3d = self.mono_processor.process_frame(img)
            
            # 更新点云
            if len(points_3d) > 0:
                self._update_point_cloud(points_3d, colors_3d, is_mock)
                return True
                
        except Exception as e:
            print(f"单目处理失败: {e}")
        
        return False
    
    def _update_point_cloud(self, points_3d, colors_3d, is_mock):
        """更新点云数据"""
        self.points_3d.extend(points_3d)
        self.colors_3d.extend(colors_3d)
        
        # 限制点云大小
        max_points = 4000 if is_mock else 3000
        if len(self.points_3d) > max_points:
            self.points_3d = self.points_3d[-max_points:]
            self.colors_3d = self.colors_3d[-max_points:]
    
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
    
    def save_reconstruction(self, save_path):
        """保存重建结果"""
        reconstruction_data = self.get_reconstruction_data()
        if reconstruction_data:
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            return True
        return False
    
    def clear_point_cloud(self):
        """清空点云"""
        self.points_3d.clear()
        self.colors_3d.clear()
        self.frame_count = 0