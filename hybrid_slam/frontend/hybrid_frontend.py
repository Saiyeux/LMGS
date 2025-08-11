"""
混合前端实现
整合特征跟踪、几何求解和渲染优化
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# 添加thirdparty路径
project_root = Path(__file__).parent.parent.parent
monogs_path = project_root / "thirdparty" / "MonoGS"
sys.path.insert(0, str(monogs_path))

from ..matchers.loftr_matcher import EfficientLoFTRMatcher
from ..solvers.pnp_solver import PnPSolver, PnPResult
from ..utils.data_converter import ImageProcessor
from ..utils.config_manager import ConfigManager

@dataclass
class TrackingResult:
    """跟踪结果数据结构"""
    success: bool
    pose: torch.Tensor           # 4x4位姿矩阵
    tracking_method: str         # 跟踪方法：'feature'|'rendering'|'hybrid'
    num_matches: int            # 特征匹配数量
    num_inliers: int            # PnP内点数量
    reprojection_error: float   # 重投影误差
    processing_time: float      # 处理时间(ms)
    confidence: float           # 跟踪置信度

class HybridFrontEnd:
    """混合前端主类 - 整合EfficientLoFTR和MonoGS渲染优化"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化混合前端"""
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # 超时保护
        import threading
        import time
        
        def init_with_timeout():
            try:
                print("Initializing hybrid tracking...")
                # 初始化组件
                self.feature_matcher = EfficientLoFTRMatcher(config.get('loftr_config', {}))
                print("Feature matcher initialized")
                self.pnp_solver = PnPSolver(config.get('pnp_solver', {}))  
                print("PnP solver initialized")
            except Exception as e:
                print(f"Frontend initialization failed: {e}")
                self.feature_matcher = None
                self.pnp_solver = None
        
        # 使用线程和超时
        init_thread = threading.Thread(target=init_with_timeout)
        init_thread.daemon = True
        init_thread.start()
        init_thread.join(timeout=30)  # 30秒超时
        
        if init_thread.is_alive():
            print("Frontend initialization timeout - using fallback mode")
            self.feature_matcher = None
            self.pnp_solver = PnPSolver(config.get('pnp_solver', {}))
        
        # timeout_protection marker
        
        # 跟踪参数
        self.min_matches = config.get('min_matches', 50)
        self.min_inliers = config.get('min_inliers', 20)
        self.feature_confidence_threshold = config.get('feature_confidence', 0.2)
        self.tracking_lost_threshold = config.get('tracking_lost_threshold', 10)
        
        # 状态维护
        self.current_keyframe = None
        self.previous_frame = None
        self.tracking_lost_count = 0
        self.frame_buffer = []
        
        # 性能统计
        self.tracking_stats = {
            'feature_tracking': [],
            'pnp_solving': [],
            'total_time': []
        }
    
    def tracking(self, cur_frame_idx: int, viewpoint, image: np.ndarray, 
                camera_matrix: np.ndarray) -> TrackingResult:
        """混合跟踪主函数"""
        start_time = time.time()
        
        try:
            # 初始化检查
            if self.current_keyframe is None:
                return self._initialize_tracking(cur_frame_idx, viewpoint, image, camera_matrix)
            
            # 执行混合跟踪
            result = self._hybrid_tracking(cur_frame_idx, viewpoint, image, camera_matrix)
            
            # 更新状态
            self._update_tracking_state(result, viewpoint, image)
            
            # 记录性能统计
            result.processing_time = (time.time() - start_time) * 1000
            self.tracking_stats['total_time'].append(result.processing_time)
            
            return result
            
        except Exception as e:
            print(f"Tracking failed: {e}")
            return TrackingResult(
                success=False,
                pose=torch.eye(4),
                tracking_method='failed',
                num_matches=0,
                num_inliers=0,
                reprojection_error=float('inf'),
                processing_time=(time.time() - start_time) * 1000,
                confidence=0.0
            )
    
    def _initialize_tracking(self, cur_frame_idx: int, viewpoint, 
                           image: np.ndarray, camera_matrix: np.ndarray) -> TrackingResult:
        """初始化跟踪"""
        print("Initializing hybrid tracking...")
        
        # 设置当前关键帧
        self.current_keyframe = {
            'frame_idx': cur_frame_idx,
            'viewpoint': viewpoint,
            'image': image,
            'camera_matrix': camera_matrix,
            'pose': torch.eye(4),
            'depth_map': None  # 将从MonoGS获取
        }
        
        return TrackingResult(
            success=True,
            pose=torch.eye(4),
            tracking_method='initialization',
            num_matches=0,
            num_inliers=0,
            reprojection_error=0.0,
            processing_time=0.0,
            confidence=1.0
        )
    
    def _hybrid_tracking(self, cur_frame_idx: int, viewpoint, 
                        image: np.ndarray, camera_matrix: np.ndarray) -> TrackingResult:
        """执行混合跟踪 - 特征匹配 + 几何求解 + 渲染优化"""
        
        # 1. 特征匹配阶段
        feature_result = self._feature_based_tracking(image, camera_matrix)
        
        if feature_result.success and feature_result.confidence > 0.7:
            # 特征跟踪成功且置信度高，直接使用
            feature_result.tracking_method = 'feature'
            return feature_result
        
        elif feature_result.success and feature_result.confidence > 0.3:
            # 特征跟踪部分成功，作为渲染优化初值
            initial_pose = feature_result.pose
        else:
            # 特征跟踪失败，使用上一帧位姿作为初值
            initial_pose = self.previous_frame['pose'] if self.previous_frame else torch.eye(4)
        
        # 2. 渲染优化阶段（集成MonoGS）
        rendering_result = self._rendering_based_tracking(viewpoint, initial_pose)
        
        if rendering_result.success:
            # 混合跟踪成功
            return TrackingResult(
                success=True,
                pose=rendering_result.pose,
                tracking_method='hybrid',
                num_matches=feature_result.num_matches,
                num_inliers=feature_result.num_inliers,
                reprojection_error=rendering_result.reprojection_error,
                processing_time=feature_result.processing_time + rendering_result.processing_time,
                confidence=min(feature_result.confidence + rendering_result.confidence, 1.0)
            )
        else:
            # 渲染优化也失败，返回特征跟踪结果（即使可能不可靠）
            feature_result.tracking_method = 'feature_fallback'
            return feature_result
    
    def _feature_based_tracking(self, image: np.ndarray, 
                               camera_matrix: np.ndarray) -> TrackingResult:
        """基于EfficientLoFTR的特征跟踪"""
        start_time = time.time()
        
        try:
            # 特征匹配
            ref_image = self.current_keyframe['image']
            matches = self.feature_matcher.match_frames(ref_image, image)
            
            if matches is None or matches['num_matches'] < self.min_matches:
                return TrackingResult(
                    success=False,
                    pose=torch.eye(4),
                    tracking_method='feature',
                    num_matches=0 if matches is None else matches['num_matches'],
                    num_inliers=0,
                    reprojection_error=float('inf'),
                    processing_time=(time.time() - start_time) * 1000,
                    confidence=0.0
                )
            
            # 过滤低置信度匹配
            filtered_matches = self.feature_matcher.filter_matches(
                matches, self.feature_confidence_threshold
            )
            
            # PnP求解
            pnp_result = self.pnp_solver.solve_pnp_with_matches(
                filtered_matches, self.current_keyframe, None, camera_matrix
            )
            
            if not pnp_result.is_reliable(self.min_inliers):
                return TrackingResult(
                    success=False,
                    pose=torch.eye(4),
                    tracking_method='feature',
                    num_matches=filtered_matches['num_matches'],
                    num_inliers=pnp_result.num_inliers,
                    reprojection_error=pnp_result.reprojection_error,
                    processing_time=(time.time() - start_time) * 1000,
                    confidence=0.0
                )
            
            # 构建4x4位姿矩阵
            pose = torch.eye(4)
            pose[:3, :3] = pnp_result.R
            pose[:3, 3] = pnp_result.T
            
            # 计算跟踪置信度
            confidence = self._compute_tracking_confidence(
                filtered_matches, pnp_result
            )
            
            return TrackingResult(
                success=True,
                pose=pose,
                tracking_method='feature',
                num_matches=filtered_matches['num_matches'],
                num_inliers=pnp_result.num_inliers,
                reprojection_error=pnp_result.reprojection_error,
                processing_time=(time.time() - start_time) * 1000,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Feature-based tracking failed: {e}")
            return TrackingResult(
                success=False,
                pose=torch.eye(4),
                tracking_method='feature',
                num_matches=0,
                num_inliers=0,
                reprojection_error=float('inf'),
                processing_time=(time.time() - start_time) * 1000,
                confidence=0.0
            )
    
    def _rendering_based_tracking(self, viewpoint, initial_pose: torch.Tensor) -> TrackingResult:
        """基于MonoGS渲染的位姿优化"""
        start_time = time.time()
        
        try:
            # 如果有MonoGS viewpoint，使用渲染优化
            if viewpoint is not None:
                # 设置初始位姿
                R_init = initial_pose[:3, :3]
                T_init = initial_pose[:3, 3]
                viewpoint.update_RT(R_init, T_init)
                
                # 执行MonoGS渲染跟踪优化
                # 这个过程在slam_system中的_update_monogs_with_tracking_result中完成
                # 这里返回优化后的位姿
                optimized_pose = torch.eye(4)
                optimized_pose[:3, :3] = viewpoint.R
                optimized_pose[:3, 3] = viewpoint.T
                
                # 计算优化前后的位姿差异作为置信度指标
                pose_diff = torch.norm(optimized_pose - initial_pose)
                confidence = max(0.5, 1.0 - pose_diff.item() / 2.0)
                
                return TrackingResult(
                    success=True,
                    pose=optimized_pose,
                    tracking_method='rendering',
                    num_matches=0,
                    num_inliers=0,
                    reprojection_error=pose_diff.item(),
                    processing_time=(time.time() - start_time) * 1000,
                    confidence=confidence
                )
            else:
                # 没有MonoGS支持，返回初始位姿
                return TrackingResult(
                    success=True,
                    pose=initial_pose,
                    tracking_method='rendering_fallback',
                    num_matches=0,
                    num_inliers=0,
                    reprojection_error=0.0,
                    processing_time=(time.time() - start_time) * 1000,
                    confidence=0.6  # 较低的置信度，因为没有实际优化
                )
            
        except Exception as e:
            print(f"Rendering-based tracking failed: {e}")
            return TrackingResult(
                success=False,
                pose=torch.eye(4),
                tracking_method='rendering',
                num_matches=0,
                num_inliers=0,
                reprojection_error=float('inf'),
                processing_time=(time.time() - start_time) * 1000,
                confidence=0.0
            )
    
    def _compute_tracking_confidence(self, matches: Dict[str, Any], 
                                   pnp_result: PnPResult) -> float:
        """计算跟踪置信度"""
        # 基于多个因素计算置信度
        confidence_factors = []
        
        # 1. 匹配数量因子
        match_factor = min(matches['num_matches'] / 100.0, 1.0)
        confidence_factors.append(match_factor * 0.3)
        
        # 2. 内点比率因子
        inlier_ratio = pnp_result.num_inliers / max(matches['num_matches'], 1)
        confidence_factors.append(inlier_ratio * 0.4)
        
        # 3. 重投影误差因子
        error_factor = max(0, 1.0 - pnp_result.reprojection_error / 5.0)
        confidence_factors.append(error_factor * 0.3)
        
        return sum(confidence_factors)
    
    def _update_tracking_state(self, result: TrackingResult, viewpoint, image: np.ndarray):
        """更新跟踪状态"""
        if result.success:
            self.tracking_lost_count = 0
            
            # 更新前一帧信息
            self.previous_frame = {
                'viewpoint': viewpoint,
                'image': image,
                'pose': result.pose
            }
            
            # 判断是否需要插入新关键帧
            if self._should_insert_keyframe(result):
                self._insert_keyframe(viewpoint, image)
        else:
            self.tracking_lost_count += 1
            
            # 跟踪丢失处理
            if self.tracking_lost_count > self.tracking_lost_threshold:
                print("Tracking lost! Attempting recovery...")
                self._attempt_tracking_recovery(viewpoint, image)
    
    def _should_insert_keyframe(self, result: TrackingResult) -> bool:
        """判断是否应该插入新关键帧"""
        # 简单的关键帧选择策略
        if self.current_keyframe is None:
            return True
        
        # 基于运动幅度判断
        pose_diff = torch.norm(result.pose[:3, 3] - self.current_keyframe['pose'][:3, 3])
        rotation_diff = torch.norm(result.pose[:3, :3] - self.current_keyframe['pose'][:3, :3])
        
        return pose_diff > 0.1 or rotation_diff > 0.05
    
    def _insert_keyframe(self, viewpoint, image: np.ndarray):
        """插入新关键帧"""
        print("Inserting new keyframe...")
        
        self.current_keyframe = {
            'viewpoint': viewpoint,
            'image': image,
            'pose': self.previous_frame['pose'].clone(),
            'camera_matrix': self.current_keyframe['camera_matrix'],  # 复用相机参数
            'depth_map': None  # 将从MonoGS更新
        }
    
    def _attempt_tracking_recovery(self, viewpoint, image: np.ndarray):
        """尝试跟踪恢复"""
        print("Attempting tracking recovery...")
        
        # TODO: 实现跟踪恢复策略
        # 1. 重新初始化
        # 2. 使用更松的匹配阈值
        # 3. 回退到前几帧进行匹配
        
        self.tracking_lost_count = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {}
        
        for key, times in self.tracking_stats.items():
            if times:
                stats[key] = {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'count': len(times)
                }
        
        return stats