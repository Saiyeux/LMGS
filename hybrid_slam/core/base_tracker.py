"""
跟踪器基类
定义跟踪器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class BaseTracker(ABC):
    """跟踪器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def track(self, current_frame, reference_frame=None) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        跟踪方法
        
        Args:
            current_frame: 当前帧
            reference_frame: 参考帧
            
        Returns:
            R: 旋转矩阵
            T: 平移向量  
            success: 是否成功
        """
        pass
    
    @abstractmethod
    def is_tracking_reliable(self) -> bool:
        """判断跟踪是否可靠"""
        pass