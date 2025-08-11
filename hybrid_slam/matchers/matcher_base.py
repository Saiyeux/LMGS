"""
匹配器基类
定义特征匹配器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class MatcherBase(ABC):
    """特征匹配器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def match_frames(self, img0: np.ndarray, img1: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        执行帧间特征匹配
        
        Args:
            img0: 参考帧图像
            img1: 当前帧图像
            
        Returns:
            匹配结果字典，包含匹配点和置信度
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        """加载预训练模型"""
        pass
    
    def is_match_reliable(self, matches: Dict[str, Any], threshold: float = 0.5) -> bool:
        """判断匹配结果是否可靠"""
        if matches is None or 'confidence' not in matches:
            return False
        return np.mean(matches['confidence']) > threshold