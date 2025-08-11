"""
数据结构定义
定义系统中使用的通用数据结构
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

@dataclass
class StereoFrame:
    """立体帧数据结构 - 重新设计为Qt架构"""
    frame_id: int
    timestamp: float
    left_image: np.ndarray
    right_image: np.ndarray
    camera_matrices: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (K_left, K_right)
    baseline: Optional[float] = None
    depth_map: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ProcessingResult:
    """AI处理结果数据结构 - 新增"""
    frame_id: int
    timestamp: float = 0.0
    matches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    confidence: float = 0.0
    pose: Optional[np.ndarray] = None
    processing_time: float = 0.0
    visualization_data: Optional[np.ndarray] = None
    error: Optional[str] = None
    num_matches: int = 0
    method: str = "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)