#!/usr/bin/env python3
"""
HybridFrontEnd集成测试
测试混合前端的完整工作流程
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from hybrid_slam.frontend.hybrid_frontend import HybridFrontEnd, TrackingResult

class TestHybridFrontEndIntegration:
    """HybridFrontEnd集成测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            'device': 'cpu',
            'EfficientLoFTR': {
                'model_path': 'mock_model.ckpt',
                'device': 'cpu'
            },
            'PnPSolver': {
                'pnp_ransac_threshold': 2.0,
                'pnp_min_inliers': 20
            },
            'min_matches': 50,
            'min_inliers': 20,
            'feature_confidence': 0.2,
            'tracking_lost_threshold': 10
        }
        
        # 模拟EfficientLoFTRMatcher和PnPSolver
        with patch('hybrid_slam.frontend.hybrid_frontend.EfficientLoFTRMatcher') as mock_matcher, \
             patch('hybrid_slam.frontend.hybrid_frontend.PnPSolver') as mock_solver:
            
            self.mock_matcher_class = mock_matcher
            self.mock_solver_class = mock_solver
            
            # 创建HybridFrontEnd实例
            self.frontend = HybridFrontEnd(self.config)
    
    def test_initialization(self):
        """测试前端初始化"""
        assert self.frontend.config == self.config
        assert self.frontend.device == 'cpu'
        assert self.frontend.min_matches == 50
        assert self.frontend.min_inliers == 20
        assert self.frontend.current_keyframe is None
        assert self.frontend.previous_frame is None
        assert self.frontend.tracking_lost_count == 0
    
    def test_first_frame_initialization(self):
        """测试首帧初始化"""
        # 准备测试数据
        cur_frame_idx = 0
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        result = self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
        
        # 验证初始化结果
        assert result.success == True
        assert result.tracking_method == 'initialization'
        assert result.num_matches == 0
        assert result.num_inliers == 0
        assert result.confidence == 1.0
        
        # 验证状态更新
        assert self.frontend.current_keyframe is not None
        assert self.frontend.current_keyframe['frame_idx'] == 0
        assert torch.equal(result.pose, torch.eye(4))
    
    def test_successful_feature_tracking(self):
        """测试成功的特征跟踪"""
        # 先初始化
        self._initialize_frontend()
        
        # 模拟成功的特征匹配
        mock_matches = {
            'keypoints0': np.random.rand(100, 2) * 640,
            'keypoints1': np.random.rand(100, 2) * 640,
            'confidence': np.random.rand(100) * 0.5 + 0.5,  # 0.5-1.0
            'num_matches': 100
        }
        
        self.frontend.feature_matcher.match_frames = Mock(return_value=mock_matches)
        self.frontend.feature_matcher.filter_matches = Mock(return_value=mock_matches)
        
        # 模拟成功的PnP求解
        from hybrid_slam.solvers.pnp_solver import PnPResult
        mock_pnp_result = PnPResult(
            success=True,
            R=torch.eye(3),
            T=torch.tensor([0.1, 0.2, 0.3]),
            inliers=np.arange(80),  # 80个内点
            num_inliers=80,
            reprojection_error=1.5,
            processing_time=25.0
        )
        self.frontend.pnp_solver.solve_pnp_with_matches = Mock(return_value=mock_pnp_result)
        
        # 执行跟踪
        cur_frame_idx = 1
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        result = self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
        
        # 验证跟踪结果
        assert result.success == True
        assert result.tracking_method == 'feature'
        assert result.num_matches == 100
        assert result.num_inliers == 80
        assert result.confidence > 0.7  # 高置信度
    
    def test_hybrid_tracking_fallback(self):
        """测试混合跟踪降级机制"""
        # 先初始化
        self._initialize_frontend()
        
        # 模拟中等质量的特征匹配（需要渲染优化）
        mock_matches = {
            'keypoints0': np.random.rand(60, 2) * 640,
            'keypoints1': np.random.rand(60, 2) * 640,
            'confidence': np.random.rand(60) * 0.3 + 0.2,  # 0.2-0.5
            'num_matches': 60
        }
        
        self.frontend.feature_matcher.match_frames = Mock(return_value=mock_matches)
        self.frontend.feature_matcher.filter_matches = Mock(return_value=mock_matches)
        
        # 模拟PnP求解（中等质量）
        from hybrid_slam.solvers.pnp_solver import PnPResult
        mock_pnp_result = PnPResult(
            success=True,
            R=torch.eye(3),
            T=torch.tensor([0.1, 0.2, 0.3]),
            inliers=np.arange(25),  # 25个内点
            num_inliers=25,
            reprojection_error=2.5,
            processing_time=20.0
        )
        self.frontend.pnp_solver.solve_pnp_with_matches = Mock(return_value=mock_pnp_result)
        
        # 执行跟踪
        cur_frame_idx = 1
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        result = self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
        
        # 验证混合跟踪结果
        assert result.success == True
        assert result.tracking_method == 'hybrid'
        assert result.num_matches == 60
        assert result.num_inliers == 25
    
    def test_tracking_failure(self):
        """测试跟踪失败"""
        # 先初始化
        self._initialize_frontend()
        
        # 模拟特征匹配失败
        self.frontend.feature_matcher.match_frames = Mock(return_value=None)
        
        # 执行跟踪
        cur_frame_idx = 1
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        result = self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
        
        # 验证失败结果
        assert result.success == False
        assert result.tracking_method == 'feature_fallback'
        assert result.num_matches == 0
        assert result.confidence == 0.0
        
        # 验证失败计数增加
        assert self.frontend.tracking_lost_count == 1
    
    def test_keyframe_insertion(self):
        """测试关键帧插入逻辑"""
        # 先初始化
        self._initialize_frontend()
        
        # 创建一个有显著运动的跟踪结果
        pose = torch.eye(4)
        pose[:3, 3] = torch.tensor([0.2, 0.1, 0.15])  # 显著平移
        
        mock_result = TrackingResult(
            success=True,
            pose=pose,
            tracking_method='feature',
            num_matches=100,
            num_inliers=80,
            reprojection_error=1.0,
            processing_time=20.0,
            confidence=0.8
        )
        
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 检查是否应该插入关键帧
        should_insert = self.frontend._should_insert_keyframe(mock_result)
        assert should_insert == True
        
        # 执行关键帧插入
        self.frontend._update_tracking_state(mock_result, viewpoint, image)
        
        # 验证关键帧更新
        assert self.frontend.current_keyframe['pose'] is not None
        assert torch.allclose(self.frontend.current_keyframe['pose'], pose, atol=1e-6)
    
    def test_tracking_confidence_computation(self):
        """测试跟踪置信度计算"""
        matches = {
            'num_matches': 100
        }
        
        from hybrid_slam.solvers.pnp_solver import PnPResult
        pnp_result = PnPResult(
            success=True,
            R=torch.eye(3),
            T=torch.zeros(3),
            inliers=np.arange(80),
            num_inliers=80,
            reprojection_error=1.0,
            processing_time=0.0
        )
        
        confidence = self.frontend._compute_tracking_confidence(matches, pnp_result)
        
        # 验证置信度计算
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 好的匹配和PnP结果应该有高置信度
    
    def test_performance_statistics(self):
        """测试性能统计功能"""
        # 添加一些模拟的性能数据
        self.frontend.tracking_stats['total_time'].extend([20.5, 25.3, 18.7, 22.1])
        
        stats = self.frontend.get_performance_stats()
        
        assert 'total_time' in stats
        assert 'mean' in stats['total_time']
        assert 'std' in stats['total_time']
        assert 'min' in stats['total_time']
        assert 'max' in stats['total_time']
        assert 'count' in stats['total_time']
        
        # 验证统计值
        assert abs(stats['total_time']['mean'] - 21.65) < 0.1
        assert stats['total_time']['count'] == 4
    
    def test_tracking_recovery_attempt(self):
        """测试跟踪恢复尝试"""
        # 先初始化
        self._initialize_frontend()
        
        # 设置连续跟踪失败
        self.frontend.tracking_lost_count = self.frontend.tracking_lost_threshold + 1
        
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 执行跟踪恢复
        self.frontend._attempt_tracking_recovery(viewpoint, image)
        
        # 验证恢复后状态重置
        assert self.frontend.tracking_lost_count == 0
    
    def _initialize_frontend(self):
        """辅助方法：初始化前端状态"""
        cur_frame_idx = 0
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
    
    def test_exception_handling(self):
        """测试异常处理"""
        # 模拟特征匹配器抛出异常
        self.frontend.feature_matcher.match_frames = Mock(side_effect=Exception("Test exception"))
        
        # 先初始化
        self._initialize_frontend()
        
        # 执行跟踪
        cur_frame_idx = 1
        viewpoint = Mock()
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        camera_matrix = np.eye(3)
        
        result = self.frontend.tracking(cur_frame_idx, viewpoint, image, camera_matrix)
        
        # 验证异常被正确处理
        assert result.success == False
        assert result.tracking_method == 'failed'
        assert result.confidence == 0.0