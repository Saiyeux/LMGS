"""
Stereo Vision Processing Module
传统立体视觉处理模块
"""

import cv2
import numpy as np


class StereoProcessor:
    """立体视觉处理器"""
    
    def __init__(self, camera_params=None):
        """
        初始化立体视觉处理器
        
        Args:
            camera_params: 相机参数字典
        """
        self.camera_params = camera_params or {}
        
        # 初始化立体匹配器和特征检测器
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def process_stereo_pair(self, left_img, right_img, is_mock=False):
        """
        处理立体图像对
        
        Args:
            left_img: 左图像
            right_img: 右图像
            is_mock: 是否为模拟数据
            
        Returns:
            tuple: (points_3d, colors_3d) 3D点和颜色
        """
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 计算视差图
            if is_mock:
                disparity = self._compute_mock_disparity(left_gray, right_gray)
            else:
                disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # 从视差图生成3D点云
            points_3d, colors_3d = self._disparity_to_3d(left_img, disparity)
            
            return points_3d, colors_3d
            
        except Exception as e:
            print(f"立体视觉处理失败: {e}")
            return [], []
    
    def _compute_mock_disparity(self, left_gray, right_gray):
        """计算模拟视差图"""
        # 对于模拟数据，使用简化的视差计算
        disparity = np.zeros_like(left_gray, dtype=np.float32)
        
        # 检测特征点
        kp1, desc1 = self.detector.detectAndCompute(left_gray, None)
        kp2, desc2 = self.detector.detectAndCompute(right_gray, None)
        
        if desc1 is not None and desc2 is not None:
            matches = self.matcher.match(desc1, desc2)
            
            for match in matches:
                if match.distance < 50:  # 好匹配
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    
                    # 计算视差
                    disp = abs(pt1[0] - pt2[0])
                    if disp > 1:
                        x, y = int(pt1[0]), int(pt1[1])
                        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                            disparity[y, x] = disp
        
        # 平滑视差图
        disparity = cv2.GaussianBlur(disparity, (5, 5), 1.0)
        
        return disparity
    
    def _disparity_to_3d(self, color_img, disparity):
        """从视差图生成3D点云"""
        points_3d = []
        colors_3d = []
        
        # 获取相机参数
        fx = self.camera_params.get('fx', 525.0)
        fy = self.camera_params.get('fy', 525.0)
        cx = self.camera_params.get('cx', 320.0)
        cy = self.camera_params.get('cy', 240.0)
        baseline = self.camera_params.get('baseline', 0.12)
        
        h, w = disparity.shape
        step = 6  # 采样步长
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = disparity[y, x]
                
                if d > 1.0:  # 有效视差
                    Z = (fx * baseline) / d
                    if 0.5 < Z < 12.0:  # 合理深度范围
                        X = (x - cx) * Z / fx
                        Y = (y - cy) * Z / fy
                        
                        points_3d.append([X, Y, Z])
                        
                        # 获取颜色
                        color = color_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d