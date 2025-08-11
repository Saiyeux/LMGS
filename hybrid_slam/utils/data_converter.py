"""
数据格式转换工具
处理不同数据格式之间的转换
"""

import numpy as np
import torch
import cv2
from typing import Union

class ImageProcessor:
    """图像处理和格式转换"""
    
    @staticmethod
    def torch_to_cv2(img_tensor: torch.Tensor) -> np.ndarray:
        """PyTorch张量转OpenCV格式"""
        if img_tensor.dim() == 4:  # [B, C, H, W]
            img_tensor = img_tensor.squeeze(0)
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:  # [1, H, W] -> [H, W]
            img_tensor = img_tensor.squeeze(0)
        
        img_np = img_tensor.detach().cpu().numpy()
        
        if len(img_np.shape) == 2:  # Grayscale [H, W]
            return (img_np * 255).astype(np.uint8)
        elif len(img_np.shape) == 3:  # RGB [C, H, W] -> [H, W, C]
            img_np = np.transpose(img_np, (1, 2, 0))
            return (img_np * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported tensor shape: {img_np.shape}")
    
    @staticmethod  
    def cv2_to_torch(img_cv2: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """OpenCV格式转PyTorch张量"""
        # 确保输入是float32并归一化
        if img_cv2.dtype != np.float32:
            img_cv2 = img_cv2.astype(np.float32) / 255.0
        
        if len(img_cv2.shape) == 2:  # Grayscale [H, W] -> [1, 1, H, W]
            img_tensor = torch.from_numpy(img_cv2)
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif len(img_cv2.shape) == 3:  # RGB [H, W, C] -> [1, C, H, W]
            # OpenCV使用BGR，转换为RGB
            if img_cv2.shape[2] == 3:
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_cv2)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        else:
            raise ValueError(f"Unsupported image shape: {img_cv2.shape}")
            
        return img_tensor.to(device)
    
    @staticmethod
    def normalize_image(image: Union[np.ndarray, torch.Tensor], 
                       mean: Union[float, tuple] = 0.5, 
                       std: Union[float, tuple] = 0.5) -> Union[np.ndarray, torch.Tensor]:
        """图像归一化"""
        if isinstance(image, torch.Tensor):
            if isinstance(mean, (int, float)):
                mean = [mean] * image.shape[1] if image.dim() == 4 else [mean]
            if isinstance(std, (int, float)):
                std = [std] * image.shape[1] if image.dim() == 4 else [std]
            
            mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
            std = torch.tensor(std).view(-1, 1, 1).to(image.device)
            
            return (image - mean) / std
        else:
            return (image - mean) / std
    
    @staticmethod
    def resize_image(image: Union[np.ndarray, torch.Tensor], target_size: tuple, 
                    interpolation: str = 'bilinear') -> Union[np.ndarray, torch.Tensor]:
        """图像尺寸调整"""
        if isinstance(image, torch.Tensor):
            # 使用PyTorch的interpolate
            import torch.nn.functional as F
            
            if image.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                image = image.unsqueeze(0)
                squeeze_batch = True
            else:
                squeeze_batch = False
            
            # 确保target_size是(H, W)格式，PyTorch的interpolate期望(H, W)
            if len(target_size) == 2:
                target_size = (target_size[0], target_size[1])  # 保持(H, W)格式
            
            resized = F.interpolate(image, size=target_size, mode=interpolation, align_corners=False)
            
            if squeeze_batch:
                resized = resized.squeeze(0)
                
            return resized
        else:
            # 使用OpenCV resize，需要(W, H)格式
            if len(target_size) == 2:
                # target_size是(H, W)，转换为OpenCV的(W, H)
                target_size = (target_size[1], target_size[0])
            
            cv2_interp = cv2.INTER_LINEAR if interpolation == 'bilinear' else cv2.INTER_NEAREST
            return cv2.resize(image, target_size, interpolation=cv2_interp)
    
    @staticmethod
    def ensure_divisible_by_32(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """确保图像尺寸能被32整除（EfficientLoFTR要求）"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # [B, C, H, W]
                H, W = image.shape[2], image.shape[3]
            elif image.dim() == 3:  # [C, H, W]
                H, W = image.shape[1], image.shape[2]
            else:
                raise ValueError(f"Unsupported tensor dimensions: {image.dim()}")
        else:
            if len(image.shape) == 3:  # [H, W, C]
                H, W = image.shape[0], image.shape[1]
            elif len(image.shape) == 2:  # [H, W]
                H, W = image.shape
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # 计算新的尺寸（向下取整到32的倍数）
        new_H = (H // 32) * 32
        new_W = (W // 32) * 32
        
        if new_H != H or new_W != W:
            return ImageProcessor.resize_image(image, (new_H, new_W))
        
        return image