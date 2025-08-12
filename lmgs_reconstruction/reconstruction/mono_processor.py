"""
Monocular Vision Processing Module
单目视觉处理模块
"""

import cv2
import numpy as np


class MonoProcessor:
    """单目视觉处理器"""
    
    def __init__(self, camera_params=None):
        """
        初始化单目视觉处理器
        
        Args:
            camera_params: 相机参数字典
        """
        self.camera_params = camera_params or {}
        
        # 初始化特征检测器和匹配器
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 存储前一帧的特征
        self.prev_kp = None
        self.prev_desc = None
    
    def process_frame(self, img):
        """
        处理单目帧
        
        Args:
            img: 输入图像
            
        Returns:
            tuple: (points_3d, colors_3d) 3D点和颜色
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测特征点
            kp, desc = self.detector.detectAndCompute(gray, None)
            
            if self.prev_desc is not None and desc is not None:
                # 匹配特征点
                matches = self.matcher.match(self.prev_desc, desc)
                good_matches = [m for m in matches if m.distance < 50]
                
                if len(good_matches) > 20:
                    # 提取匹配点
                    pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                    
                    # 估计基础矩阵
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                    
                    if F is not None:
                        # 生成3D点
                        points_3d, colors_3d = self._generate_3d_points(
                            pts1, pts2, mask, img
                        )
                        
                        # 更新前一帧信息
                        self.prev_kp = kp
                        self.prev_desc = desc
                        
                        return points_3d, colors_3d
            
            # 更新前一帧信息
            self.prev_kp = kp
            self.prev_desc = desc
            
            return [], []
            
        except Exception as e:
            print(f"单目处理失败: {e}")
            return [], []
    
    def _generate_3d_points(self, pts1, pts2, mask, img):
        """从匹配点生成3D点"""
        points_3d = []
        colors_3d = []
        
        # 获取相机参数
        fx = self.camera_params.get('fx', 525.0)
        fy = self.camera_params.get('fy', 525.0)
        cx = self.camera_params.get('cx', 320.0)
        cy = self.camera_params.get('cy', 240.0)
        
        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            if mask[i]:
                # 基于特征运动估计深度
                motion = np.linalg.norm(pt2 - pt1)
                depth = 5.0 / (motion + 0.1) if motion > 0 else 5.0
                
                if 0.5 < depth < 15.0:
                    x, y = pt2
                    X = (x - cx) * depth / fx
                    Y = (y - cy) * depth / fy
                    Z = depth
                    
                    points_3d.append([X, Y, Z])
                    
                    # 获取颜色
                    if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                        color = img[int(y), int(x)]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d
    
    def reset(self):
        """重置处理器状态"""
        self.prev_kp = None
        self.prev_desc = None