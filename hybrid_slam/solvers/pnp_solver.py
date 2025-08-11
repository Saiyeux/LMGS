"""
PnP求解器
基于OpenCV的PnP位姿估计
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

@dataclass  
class PnPResult:
    """PnP求解结果数据结构"""
    success: bool                   # 求解是否成功
    R: torch.Tensor                 # 旋转矩阵 [3, 3]
    T: torch.Tensor                 # 平移向量 [3]
    inliers: np.ndarray            # 内点索引
    num_inliers: int               # 内点数量
    reprojection_error: float      # 重投影误差
    processing_time: float         # 处理时间(ms)
    
    def is_reliable(self, min_inliers: int = 20) -> bool:
        """判断求解结果是否可靠"""
        return self.success and self.num_inliers >= min_inliers

class PnPSolver:
    """OpenCV PnP求解器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ransac_threshold = config.get('pnp_ransac_threshold', 2.0)
        self.min_inliers = config.get('pnp_min_inliers', 20)
        self.max_iterations = config.get('pnp_max_iterations', 1000)
        self.confidence = config.get('pnp_confidence', 0.99)
        
        # PnP方法选择
        self.pnp_method = config.get('pnp_method', 'SOLVEPNP_ITERATIVE')
        
    def solve_pnp_with_matches(self, matches: Dict[str, Any], 
                              ref_keyframe, current_frame,
                              camera_matrix: np.ndarray,
                              dist_coeffs: Optional[np.ndarray] = None) -> PnPResult:
        """基于特征匹配求解PnP"""
        import time
        start_time = time.time()
        
        try:
            # 获取3D-2D对应关系
            points_3d, points_2d, valid_mask = self._get_3d_2d_correspondences(
                matches, ref_keyframe, current_frame
            )
            
            if points_3d is None or len(points_3d) < 4:
                return PnPResult(
                    success=False, R=torch.eye(3), T=torch.zeros(3),
                    inliers=np.array([]), num_inliers=0, 
                    reprojection_error=float('inf'),
                    processing_time=(time.time() - start_time) * 1000
                )
            
            # 执行RANSAC PnP
            result = self._solve_pnp_ransac(
                points_3d, points_2d, camera_matrix, dist_coeffs
            )
            
            result.processing_time = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            print(f"PnP solving failed: {e}")
            return PnPResult(
                success=False, R=torch.eye(3), T=torch.zeros(3),
                inliers=np.array([]), num_inliers=0,
                reprojection_error=float('inf'),
                processing_time=(time.time() - start_time) * 1000
            )
    
    def _get_3d_2d_correspondences(self, matches: Dict[str, Any], 
                                  ref_keyframe, current_frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """获取3D-2D点对应关系"""
        try:
            # 从匹配结果获取2D点
            kpts0 = matches['keypoints0']  # 参考帧中的点
            kpts1 = matches['keypoints1']  # 当前帧中的点
            
            # 从参考帧获取3D点（通过深度图或高斯地图）
            points_3d = self._backproject_from_depth(kpts0, ref_keyframe)
            
            if points_3d is None:
                return None, None, None
            
            # 有效的3D点掩码
            valid_mask = ~np.any(np.isnan(points_3d), axis=1) & (points_3d[:, 2] > 0)
            
            if np.sum(valid_mask) < 4:
                return None, None, None
            
            return points_3d[valid_mask], kpts1[valid_mask], valid_mask
            
        except Exception as e:
            print(f"Failed to get 3D-2D correspondences: {e}")
            return None, None, None
    
    def _backproject_from_depth(self, keypoints: np.ndarray, keyframe) -> Optional[np.ndarray]:
        """从深度图反投影获取3D点"""
        try:
            # 获取相机内参
            if hasattr(keyframe, 'camera_matrix'):
                K = keyframe.camera_matrix
            else:
                # 默认相机参数（需要根据实际情况调整）
                K = np.array([[525.0, 0, 320.0],
                             [0, 525.0, 240.0], 
                             [0, 0, 1.0]])
            
            # 获取深度图
            if hasattr(keyframe, 'depth_map'):
                depth_map = keyframe.depth_map
            elif hasattr(keyframe, 'depth'):
                depth_map = keyframe.depth
            else:
                print("No depth information available")
                return None
            
            # 反投影
            points_3d = []
            for kpt in keypoints:
                x, y = int(kpt[0]), int(kpt[1])
                
                # 检查边界
                if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                    depth = depth_map[y, x]
                    
                    if depth > 0:
                        # 反投影到3D
                        z = depth
                        x_3d = (x - K[0, 2]) * z / K[0, 0]
                        y_3d = (y - K[1, 2]) * z / K[1, 1]
                        points_3d.append([x_3d, y_3d, z])
                    else:
                        points_3d.append([np.nan, np.nan, np.nan])
                else:
                    points_3d.append([np.nan, np.nan, np.nan])
            
            return np.array(points_3d)
            
        except Exception as e:
            print(f"Backprojection failed: {e}")
            return None
    
    def _solve_pnp_ransac(self, points_3d: np.ndarray, points_2d: np.ndarray,
                         camera_matrix: np.ndarray, 
                         dist_coeffs: Optional[np.ndarray] = None) -> PnPResult:
        """使用RANSAC求解PnP问题"""
        import cv2
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))
        
        # 选择PnP方法
        method_map = {
            'SOLVEPNP_ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
            'SOLVEPNP_EPNP': cv2.SOLVEPNP_EPNP,
            'SOLVEPNP_P3P': cv2.SOLVEPNP_P3P,
            'SOLVEPNP_AP3P': cv2.SOLVEPNP_AP3P
        }
        method = method_map.get(self.pnp_method, cv2.SOLVEPNP_ITERATIVE)
        
        try:
            # 使用RANSAC求解PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                camera_matrix.astype(np.float32),
                dist_coeffs.astype(np.float32),
                iterationsCount=self.max_iterations,
                reprojectionError=self.ransac_threshold,
                confidence=self.confidence,
                flags=method
            )
            
            if success and inliers is not None:
                # 转换旋转向量到旋转矩阵
                R_mat, _ = cv2.Rodrigues(rvec)
                
                # 计算重投影误差
                reprojection_error = self._compute_reprojection_error(
                    points_3d[inliers.flatten()], points_2d[inliers.flatten()],
                    R_mat, tvec.flatten(), camera_matrix, dist_coeffs
                )
                
                return PnPResult(
                    success=True,
                    R=torch.from_numpy(R_mat).float(),
                    T=torch.from_numpy(tvec.flatten()).float(),
                    inliers=inliers.flatten(),
                    num_inliers=len(inliers),
                    reprojection_error=reprojection_error,
                    processing_time=0.0
                )
            else:
                return PnPResult(
                    success=False,
                    R=torch.eye(3),
                    T=torch.zeros(3),
                    inliers=np.array([]),
                    num_inliers=0,
                    reprojection_error=float('inf'),
                    processing_time=0.0
                )
                
        except Exception as e:
            print(f"PnP RANSAC failed: {e}")
            return PnPResult(
                success=False,
                R=torch.eye(3),
                T=torch.zeros(3),
                inliers=np.array([]),
                num_inliers=0,
                reprojection_error=float('inf'),
                processing_time=0.0
            )
    
    def _compute_reprojection_error(self, points_3d: np.ndarray, points_2d: np.ndarray,
                                   R: np.ndarray, t: np.ndarray,
                                   camera_matrix: np.ndarray, 
                                   dist_coeffs: np.ndarray) -> float:
        """计算重投影误差"""
        import cv2
        
        try:
            # 将3D点投影到图像平面
            rvec, _ = cv2.Rodrigues(R)
            projected_pts, _ = cv2.projectPoints(
                points_3d.astype(np.float32),
                rvec,
                t.reshape(3, 1),
                camera_matrix,
                dist_coeffs
            )
            
            # 计算重投影误差
            projected_pts = projected_pts.reshape(-1, 2)
            errors = np.linalg.norm(points_2d - projected_pts, axis=1)
            return float(np.mean(errors))
            
        except Exception as e:
            print(f"Failed to compute reprojection error: {e}")
            return float('inf')
    
    def _refine_pose_with_initial_guess(self, points_3d: np.ndarray, points_2d: np.ndarray,
                                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                      initial_pose: torch.Tensor, 
                                      ransac_result: PnPResult) -> PnPResult:
        """使用初始位姿优化PnP结果"""
        import cv2
        
        try:
            # 使用RANSAC结果中的内点
            if len(ransac_result.inliers) < 6:
                return ransac_result
            
            inlier_3d = points_3d[ransac_result.inliers]
            inlier_2d = points_2d[ransac_result.inliers]
            
            # 使用初始位姿作为起始值进行非线性优化
            R_init = initial_pose[:3, :3].cpu().numpy()
            t_init = initial_pose[:3, 3].cpu().numpy()
            rvec_init, _ = cv2.Rodrigues(R_init)
            tvec_init = t_init.reshape(-1, 1)
            
            # 迭代优化
            success, rvec_refined, tvec_refined = cv2.solvePnP(
                inlier_3d.astype(np.float32),
                inlier_2d.astype(np.float32),
                camera_matrix.astype(np.float32),
                dist_coeffs.astype(np.float32),
                rvec=rvec_init,
                tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # 计算优化后的重投影误差
                R_refined, _ = cv2.Rodrigues(rvec_refined)
                refined_error = self._compute_reprojection_error(
                    inlier_3d, inlier_2d, R_refined, tvec_refined.flatten(),
                    camera_matrix, dist_coeffs
                )
                
                # 如果优化后的结果更好，使用优化结果
                if refined_error < ransac_result.reprojection_error:
                    return PnPResult(
                        success=True,
                        R=torch.from_numpy(R_refined).float(),
                        T=torch.from_numpy(tvec_refined.flatten()).float(),
                        inliers=ransac_result.inliers,
                        num_inliers=ransac_result.num_inliers,
                        reprojection_error=refined_error,
                        processing_time=ransac_result.processing_time
                    )
            
            return ransac_result
            
        except Exception as e:
            print(f"Pose refinement failed: {e}")
            return ransac_result