#!/usr/bin/env python3
"""
PnPSolver单元测试
测试PnP位姿求解功能
"""

import pytest
import numpy as np
import torch
import cv2
from unittest.mock import Mock, patch
from hybrid_slam.solvers.pnp_solver import PnPSolver, PnPResult

class TestPnPSolver:
    """PnPSolver测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            'pnp_ransac_threshold': 2.0,
            'pnp_min_inliers': 10,
            'pnp_max_iterations': 1000,
            'pnp_confidence': 0.99,
            'pnp_method': 'SOLVEPNP_ITERATIVE'
        }
        self.solver = PnPSolver(self.config)
    
    def test_pnp_result_dataclass(self):
        """测试PnPResult数据结构"""
        result = PnPResult(
            success=True,
            R=torch.eye(3),
            T=torch.zeros(3),
            inliers=np.array([0, 1, 2]),
            num_inliers=3,
            reprojection_error=1.5,
            processing_time=25.0
        )
        
        assert result.success == True
        assert result.R.shape == (3, 3)
        assert result.T.shape == (3,)
        assert len(result.inliers) == 3
        assert result.num_inliers == 3
        assert result.reprojection_error == 1.5
        assert result.processing_time == 25.0
        
        # 测试可靠性判断
        assert result.is_reliable(min_inliers=2) == True  # 3 >= 2
        assert result.is_reliable(min_inliers=5) == False  # 3 < 5
    
    def test_solver_initialization(self):
        """测试求解器初始化"""
        assert self.solver.ransac_threshold == 2.0
        assert self.solver.min_inliers == 10
        assert self.solver.max_iterations == 1000
        assert self.solver.confidence == 0.99
        assert self.solver.pnp_method == 'SOLVEPNP_ITERATIVE'
    
    def test_backproject_from_depth_valid(self):
        """测试从深度图反投影3D点（有效情况）"""
        # 创建模拟关键帧
        depth_map = np.ones((480, 640)) * 2.0  # 深度为2米
        camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        keyframe = Mock()
        keyframe.depth_map = depth_map
        keyframe.camera_matrix = camera_matrix
        
        # 测试关键点
        keypoints = np.array([
            [320, 240],  # 图像中心
            [400, 300],  # 偏移点
        ])
        
        points_3d = self.solver._backproject_from_depth(keypoints, keyframe)
        
        assert points_3d is not None
        assert points_3d.shape == (2, 3)
        
        # 验证中心点的反投影（应该是(0, 0, 2)）
        assert abs(points_3d[0, 0]) < 0.01  # x ≈ 0
        assert abs(points_3d[0, 1]) < 0.01  # y ≈ 0
        assert abs(points_3d[0, 2] - 2.0) < 0.01  # z ≈ 2
    
    def test_backproject_from_depth_invalid_depth(self):
        """测试无效深度情况的反投影"""
        # 创建包含无效深度的深度图
        depth_map = np.zeros((480, 640))  # 所有深度为0（无效）
        
        keyframe = Mock()
        keyframe.depth_map = depth_map
        
        keypoints = np.array([[320, 240]])
        
        points_3d = self.solver._backproject_from_depth(keypoints, keyframe)
        
        assert points_3d is not None
        assert np.isnan(points_3d[0, 0])  # 无效深度应该返回NaN
    
    def test_backproject_from_depth_no_depth_info(self):
        """测试无深度信息的情况"""
        keyframe = Mock()
        # 没有depth_map属性
        
        keypoints = np.array([[320, 240]])
        
        points_3d = self.solver._backproject_from_depth(keypoints, keyframe)
        
        assert points_3d is None
    
    def test_get_3d_2d_correspondences_valid(self):
        """测试获取3D-2D对应关系（有效情况）"""
        # 创建模拟匹配结果（至少4个点）
        matches = {
            'keypoints0': np.array([
                [320, 240],
                [400, 300],
                [250, 180],
                [350, 350],
                [450, 200]
            ]),
            'keypoints1': np.array([
                [325, 245],  # 对应的当前帧点
                [405, 305],
                [255, 185],
                [355, 355],
                [455, 205]
            ])
        }
        
        # 创建模拟参考关键帧
        depth_map = np.ones((480, 640)) * 2.0
        ref_keyframe = Mock()
        ref_keyframe.depth_map = depth_map
        ref_keyframe.camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        # 模拟_backproject_from_depth方法
        with patch.object(self.solver, '_backproject_from_depth') as mock_backproject:
            # 返回有效的3D点（正深度）
            mock_backproject.return_value = np.array([
                [0.0, 0.0, 2.0],
                [0.3, 0.2, 2.0],
                [-0.2, -0.15, 2.0],
                [0.15, 0.25, 2.0],
                [0.35, -0.1, 2.0]
            ])
            
            points_3d, points_2d, valid_mask = self.solver._get_3d_2d_correspondences(
                matches, ref_keyframe, None
            )
            
            # 验证mock被调用
            mock_backproject.assert_called_once()
            
            assert points_3d is not None, "points_3d should not be None"
            assert points_2d is not None, "points_2d should not be None"
            assert valid_mask is not None, "valid_mask should not be None"
            assert len(points_3d) == len(points_2d), "3D and 2D points should have same length"
            assert len(points_3d) >= 4, "Should have at least 4 valid points for PnP"
            assert len(points_3d) == 5, "Should have all 5 points valid"
    
    def test_get_3d_2d_correspondences_insufficient_points(self):
        """测试3D点不足的情况"""
        matches = {
            'keypoints0': np.array([[320, 240]]),  # 只有1个点，不足4个
            'keypoints1': np.array([[325, 245]])
        }
        
        ref_keyframe = Mock()
        
        # 模拟返回有效但数量不足的3D点
        with patch.object(self.solver, '_backproject_from_depth') as mock_backproject:
            mock_backproject.return_value = np.array([[0.0, 0.0, 2.0]])
            
            points_3d, points_2d, valid_mask = self.solver._get_3d_2d_correspondences(
                matches, ref_keyframe, None
            )
            
            assert points_3d is None
            assert points_2d is None
            assert valid_mask is None
    
    @patch('cv2.solvePnPRansac')
    @patch('cv2.Rodrigues')
    def test_solve_pnp_ransac_success(self, mock_rodrigues, mock_solvepnp):
        """测试PnP RANSAC成功情况"""
        # 模拟OpenCV函数返回值
        mock_solvepnp.return_value = (
            True,  # success
            np.array([[0.1], [0.2], [0.3]]),  # rvec
            np.array([[1.0], [2.0], [3.0]]),  # tvec
            np.array([[0], [1], [2]])  # inliers
        )
        
        mock_rodrigues.return_value = (
            np.eye(3),  # rotation matrix
            None
        )
        
        # 准备测试数据
        points_3d = np.random.rand(10, 3).astype(np.float32)
        points_2d = np.random.rand(10, 2).astype(np.float32)
        camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ], dtype=np.float32)
        
        # 模拟重投影误差计算
        with patch.object(self.solver, '_compute_reprojection_error') as mock_error:
            mock_error.return_value = 1.5
            
            result = self.solver._solve_pnp_ransac(points_3d, points_2d, camera_matrix)
            
            assert result.success == True
            assert result.R.shape == (3, 3)
            assert result.T.shape == (3,)
            assert result.num_inliers == 3
            assert result.reprojection_error == 1.5
    
    @patch('cv2.solvePnPRansac')
    def test_solve_pnp_ransac_failure(self, mock_solvepnp):
        """测试PnP RANSAC失败情况"""
        # 模拟失败
        mock_solvepnp.return_value = (False, None, None, None)
        
        points_3d = np.random.rand(10, 3).astype(np.float32)
        points_2d = np.random.rand(10, 2).astype(np.float32)
        camera_matrix = np.eye(3).astype(np.float32)
        
        result = self.solver._solve_pnp_ransac(points_3d, points_2d, camera_matrix)
        
        assert result.success == False
        assert result.num_inliers == 0
        assert result.reprojection_error == float('inf')
    
    @patch('cv2.projectPoints')
    @patch('cv2.Rodrigues')
    def test_compute_reprojection_error(self, mock_rodrigues, mock_project):
        """测试重投影误差计算"""
        mock_rodrigues.return_value = (np.array([[0.1], [0.2], [0.3]]), None)
        
        # 模拟投影结果
        projected_pts = np.array([[[100, 200]], [[150, 250]]], dtype=np.float32)
        mock_project.return_value = (projected_pts, None)
        
        # 准备测试数据
        points_3d = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32)
        points_2d = np.array([[105, 205], [155, 255]], dtype=np.float32)  # 5像素误差
        R = np.eye(3)
        t = np.array([0, 0, 0])
        camera_matrix = np.eye(3).astype(np.float32)
        dist_coeffs = np.zeros((4, 1)).astype(np.float32)
        
        error = self.solver._compute_reprojection_error(
            points_3d, points_2d, R, t, camera_matrix, dist_coeffs
        )
        
        # 期望误差约为sqrt(5^2 + 5^2) = 7.07像素
        assert abs(error - 7.07) < 0.1
    
    def test_solve_pnp_with_matches_integration(self):
        """测试完整的PnP求解流程"""
        # 准备测试数据
        matches = {
            'keypoints0': np.array([
                [320, 240],
                [400, 300],
                [250, 180],
                [350, 350],
                [450, 200]
            ]),
            'keypoints1': np.array([
                [325, 245],
                [405, 305],
                [255, 185],
                [355, 355],
                [455, 205]
            ])
        }
        
        # 创建模拟关键帧
        ref_keyframe = Mock()
        ref_keyframe.depth_map = np.ones((480, 640)) * 2.0
        ref_keyframe.camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        camera_matrix = np.array([
            [525.0, 0, 320.0],
            [0, 525.0, 240.0],
            [0, 0, 1.0]
        ])
        
        # 模拟成功的PnP求解
        with patch.object(self.solver, '_solve_pnp_ransac') as mock_pnp:
            mock_result = PnPResult(
                success=True,
                R=torch.eye(3),
                T=torch.tensor([0.1, 0.2, 0.3]),
                inliers=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),  # 20个内点
                num_inliers=20,  # 满足min_inliers=20的要求
                reprojection_error=1.2,
                processing_time=0.0
            )
            mock_pnp.return_value = mock_result
            
            result = self.solver.solve_pnp_with_matches(
                matches, ref_keyframe, None, camera_matrix
            )
            
            assert result.success == True
            assert result.num_inliers >= self.solver.min_inliers