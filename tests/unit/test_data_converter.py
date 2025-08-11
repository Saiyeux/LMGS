#!/usr/bin/env python3
"""
ImageProcessor单元测试
测试图像格式转换和处理功能
"""

import pytest
import numpy as np
import torch
import cv2
from hybrid_slam.utils.data_converter import ImageProcessor

class TestImageProcessor:
    """ImageProcessor测试类"""
    
    def test_torch_to_cv2_grayscale(self):
        """测试PyTorch张量转OpenCV灰度图"""
        # 创建测试张量 [1, 1, H, W]
        H, W = 480, 640
        img_tensor = torch.rand(1, 1, H, W)
        
        # 转换
        img_cv2 = ImageProcessor.torch_to_cv2(img_tensor)
        
        # 验证结果
        assert isinstance(img_cv2, np.ndarray)
        assert img_cv2.shape == (H, W)
        assert img_cv2.dtype == np.uint8
        assert img_cv2.min() >= 0 and img_cv2.max() <= 255
    
    def test_torch_to_cv2_rgb(self):
        """测试PyTorch张量转OpenCV RGB图像"""
        # 创建测试张量 [1, 3, H, W]
        H, W = 480, 640
        img_tensor = torch.rand(1, 3, H, W)
        
        # 转换
        img_cv2 = ImageProcessor.torch_to_cv2(img_tensor)
        
        # 验证结果
        assert isinstance(img_cv2, np.ndarray)
        assert img_cv2.shape == (H, W, 3)
        assert img_cv2.dtype == np.uint8
        assert img_cv2.min() >= 0 and img_cv2.max() <= 255
    
    def test_cv2_to_torch_grayscale(self):
        """测试OpenCV灰度图转PyTorch张量"""
        # 创建测试图像
        H, W = 480, 640
        img_cv2 = np.random.randint(0, 256, (H, W), dtype=np.uint8)
        
        # 转换
        img_tensor = ImageProcessor.cv2_to_torch(img_cv2, device='cpu')
        
        # 验证结果
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (1, 1, H, W)
        assert img_tensor.dtype == torch.float32
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    
    def test_cv2_to_torch_rgb(self):
        """测试OpenCV RGB图像转PyTorch张量"""
        # 创建测试图像
        H, W = 480, 640
        img_cv2 = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        
        # 转换
        img_tensor = ImageProcessor.cv2_to_torch(img_cv2, device='cpu')
        
        # 验证结果
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (1, 3, H, W)
        assert img_tensor.dtype == torch.float32
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    
    def test_normalize_image_tensor(self):
        """测试PyTorch张量图像归一化"""
        img_tensor = torch.rand(1, 3, 480, 640)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        normalized = ImageProcessor.normalize_image(img_tensor, mean, std)
        
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == img_tensor.shape
        # 验证归一化后的分布
        assert abs(normalized.mean().item()) < 5.0  # 大致范围检查
    
    def test_normalize_image_numpy(self):
        """测试numpy数组图像归一化"""
        img_np = np.random.rand(480, 640, 3).astype(np.float32)
        mean = 0.5
        std = 0.5
        
        normalized = ImageProcessor.normalize_image(img_np, mean, std)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img_np.shape
    
    def test_resize_image_tensor(self):
        """测试PyTorch张量图像尺寸调整"""
        img_tensor = torch.rand(1, 3, 480, 640)
        target_size = (240, 320)  # (H, W)
        
        resized = ImageProcessor.resize_image(img_tensor, target_size)
        
        assert isinstance(resized, torch.Tensor)
        assert resized.shape == (1, 3, 240, 320)
    
    def test_resize_image_numpy(self):
        """测试numpy数组图像尺寸调整"""
        img_np = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        target_size = (240, 320)  # (H, W) 格式，统一接口
        
        resized = ImageProcessor.resize_image(img_np, target_size)
        
        assert isinstance(resized, np.ndarray)
        assert resized.shape == (240, 320, 3)  # (H, W, C)
    
    def test_ensure_divisible_by_32_tensor(self):
        """测试确保PyTorch张量尺寸能被32整除"""
        # 测试不能被32整除的尺寸
        img_tensor = torch.rand(1, 1, 479, 639)  # 479, 639不能被32整除
        
        processed = ImageProcessor.ensure_divisible_by_32(img_tensor)
        
        assert isinstance(processed, torch.Tensor)
        H, W = processed.shape[2], processed.shape[3]
        assert H % 32 == 0
        assert W % 32 == 0
        assert H <= 479  # 向下取整
        assert W <= 639
    
    def test_ensure_divisible_by_32_numpy(self):
        """测试确保numpy数组尺寸能被32整除"""
        # 测试不能被32整除的尺寸
        img_np = np.random.randint(0, 256, (479, 639), dtype=np.uint8)
        
        processed = ImageProcessor.ensure_divisible_by_32(img_np)
        
        assert isinstance(processed, np.ndarray)
        H, W = processed.shape[0], processed.shape[1]
        assert H % 32 == 0
        assert W % 32 == 0
        assert H <= 479
        assert W <= 639
    
    def test_ensure_divisible_by_32_already_divisible(self):
        """测试已经能被32整除的图像"""
        img_tensor = torch.rand(1, 1, 480, 640)  # 480=15*32, 640=20*32
        
        processed = ImageProcessor.ensure_divisible_by_32(img_tensor)
        
        # 应该返回原图像（形状不变）
        assert processed.shape == img_tensor.shape
    
    def test_round_trip_conversion(self):
        """测试格式转换的往返一致性"""
        # 创建原始OpenCV图像
        H, W = 480, 640
        original_img = np.random.randint(0, 256, (H, W), dtype=np.uint8)
        
        # OpenCV -> PyTorch -> OpenCV
        tensor = ImageProcessor.cv2_to_torch(original_img, device='cpu')
        reconstructed_img = ImageProcessor.torch_to_cv2(tensor)
        
        # 验证形状一致
        assert reconstructed_img.shape == original_img.shape
        
        # 验证数值近似（考虑浮点精度）
        diff = np.abs(reconstructed_img.astype(float) - original_img.astype(float))
        assert np.mean(diff) < 2.0  # 平均误差小于2个灰度值
    
    def test_invalid_tensor_dimensions(self):
        """测试无效张量维度的错误处理"""
        # 创建无效维度的张量
        invalid_tensor = torch.rand(480, 640, 3, 4)  # 4维但形状错误
        
        with pytest.raises(ValueError):
            ImageProcessor.torch_to_cv2(invalid_tensor)
    
    def test_invalid_image_dimensions(self):
        """测试无效图像维度的错误处理"""
        # 创建无效维度的图像
        invalid_img = np.random.randint(0, 256, (480, 640, 3, 4), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            ImageProcessor.ensure_divisible_by_32(invalid_img)