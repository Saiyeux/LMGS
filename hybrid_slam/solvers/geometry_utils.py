"""
几何工具函数
包含3D几何计算相关的工具函数
"""

import numpy as np
import torch
from typing import Tuple

def backproject_depth(depth_map: np.ndarray, intrinsics: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    从深度图反投影获取3D点
    
    Args:
        depth_map: 深度图 [H, W]
        intrinsics: 相机内参矩阵 [3, 3]
        keypoints: 关键点坐标 [N, 2]
        
    Returns:
        points_3d: 3D点坐标 [N, 3]
    """
    # TODO: 实现深度反投影
    pass

def transform_points(points: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    使用旋转和平移变换3D点
    
    Args:
        points: 输入3D点 [N, 3]
        R: 旋转矩阵 [3, 3]
        T: 平移向量 [3]
        
    Returns:
        transformed_points: 变换后的3D点 [N, 3]
    """
    # TODO: 实现3D点变换
    pass

def project_points(points_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    将3D点投影到图像平面
    
    Args:
        points_3d: 3D点 [N, 3]
        intrinsics: 相机内参 [3, 3]
        
    Returns:
        points_2d: 投影的2D点 [N, 2]
    """
    # TODO: 实现3D到2D投影
    pass

def compute_reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray, 
                              R: np.ndarray, T: np.ndarray, intrinsics: np.ndarray) -> float:
    """计算重投影误差"""
    # TODO: 实现重投影误差计算
    pass