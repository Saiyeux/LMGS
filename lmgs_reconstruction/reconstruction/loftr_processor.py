"""
EfficientLoFTR Processing Module
使用EfficientLoFTR进行高质量特征匹配和3D重建
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy

from ..utils.dependencies import LOFTR_AVAILABLE

if LOFTR_AVAILABLE:
    from src.loftr import LoFTR, full_default_cfg, reparameter


class LoFTRProcessor:
    """EfficientLoFTR处理器"""
    
    def __init__(self, device='cpu', camera_params=None):
        """
        初始化LoFTR处理器
        
        Args:
            device: 计算设备
            camera_params: 相机参数字典
        """
        self.device = torch.device(device)
        self.camera_params = camera_params or {}
        self.matcher = None
        self.use_loftr = False
        
        if LOFTR_AVAILABLE:
            self._init_loftr()
    
    def _init_loftr(self):
        """初始化EfficientLoFTR匹配器"""
        try:
            print("初始化EfficientLoFTR...")
            _default_cfg = deepcopy(full_default_cfg)
            self.matcher = LoFTR(config=_default_cfg)
            
            # 尝试加载权重
            weights_path = "thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
            if Path(weights_path).exists():
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                self.matcher.load_state_dict(checkpoint['state_dict'])
                self.matcher = reparameter(self.matcher)
                self.matcher = self.matcher.eval().to(self.device)
                self.use_loftr = True
                print("EfficientLoFTR初始化成功")
            else:
                print(f"EfficientLoFTR权重文件不存在: {weights_path}")
        except Exception as e:
            print(f"EfficientLoFTR初始化失败: {e}")
    
    def process_stereo_pair(self, left_img, right_img):
        """
        使用EfficientLoFTR处理立体图像对
        
        Args:
            left_img: 左图像
            right_img: 右图像
            
        Returns:
            tuple: (points_3d, colors_3d) 3D点和颜色
        """
        if not self.use_loftr:
            return [], []
            
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 预处理图像
            left_tensor, right_tensor = self._preprocess_images(left_gray, right_gray)
            
            # 创建批次
            batch = {'image0': left_tensor, 'image1': right_tensor}
            
            # EfficientLoFTR匹配
            with torch.no_grad():
                self.matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
            
            # 过滤高置信度匹配
            mask = mconf > 0.3
            mkpts0_filtered = mkpts0[mask]
            mkpts1_filtered = mkpts1[mask]
            
            # 计算3D坐标
            points_3d, colors_3d = self._compute_3d_points(
                mkpts0_filtered, mkpts1_filtered, left_img
            )
            
            return points_3d, colors_3d
            
        except Exception as e:
            print(f"EfficientLoFTR处理失败: {e}")
            return [], []
    
    def _preprocess_images(self, left_gray, right_gray):
        """预处理图像"""
        # 转换为张量
        left_tensor = torch.from_numpy(left_gray)[None][None].to(self.device).float() / 255.0
        right_tensor = torch.from_numpy(right_gray)[None][None].to(self.device).float() / 255.0
        
        # 确保尺寸是32的倍数
        h, w = left_gray.shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        
        if new_h != h or new_w != w:
            left_tensor = torch.nn.functional.interpolate(left_tensor, size=(new_h, new_w))
            right_tensor = torch.nn.functional.interpolate(right_tensor, size=(new_h, new_w))
        
        return left_tensor, right_tensor
    
    def _compute_3d_points(self, mkpts0, mkpts1, left_img):
        """从匹配点计算3D坐标"""
        points_3d = []
        colors_3d = []
        
        # 获取相机参数
        fx = self.camera_params.get('fx', 525.0)
        fy = self.camera_params.get('fy', 525.0)
        cx = self.camera_params.get('cx', 320.0)
        cy = self.camera_params.get('cy', 240.0)
        baseline = self.camera_params.get('baseline', 0.12)
        
        for i in range(len(mkpts0)):
            pt0 = mkpts0[i]
            pt1 = mkpts1[i]
            
            # 计算视差
            disparity = abs(pt0[0] - pt1[0])
            if disparity > 1.0:
                # 计算3D坐标
                Z = (fx * baseline) / disparity
                if 0.5 < Z < 15.0:
                    X = (pt0[0] - cx) * Z / fx
                    Y = (pt0[1] - cy) * Z / fy
                    
                    points_3d.append([X, Y, Z])
                    
                    # 获取颜色
                    x, y = int(pt0[0]), int(pt0[1])
                    if 0 <= y < left_img.shape[0] and 0 <= x < left_img.shape[1]:
                        color = left_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d