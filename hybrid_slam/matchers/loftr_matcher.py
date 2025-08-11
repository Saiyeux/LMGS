"""
EfficientLoFTR特征匹配器
封装EfficientLoFTR模型进行特征匹配
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .matcher_base import MatcherBase
from ..utils.data_converter import ImageProcessor

# 添加thirdparty路径
project_root = Path(__file__).parent.parent.parent
eloftr_path = project_root / "thirdparty" / "EfficientLoFTR"
sys.path.insert(0, str(eloftr_path))

try:
    from src.loftr import LoFTR, full_default_cfg, reparameter
    from src.utils.misc import lower_config
    from copy import deepcopy
    ELOFTR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EfficientLoFTR: {e}")
    print("Please ensure EfficientLoFTR is properly installed")
    ELOFTR_AVAILABLE = False
    LoFTR = None
    full_default_cfg = None
    lower_config = None

class EfficientLoFTRMatcher(MatcherBase):
    """EfficientLoFTR匹配器 - 支持立体匹配和时序匹配"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda')
        self.model = None
        self.loftr_config = None
        
        # 匹配参数
        self.resize_to = config.get('resize_to', [640, 480])  # [width, height]
        self.match_threshold = config.get('match_threshold', 0.2)
        self.max_keypoints = config.get('max_keypoints', 2048)
        
        # 立体匹配特定参数
        self.enable_stereo_constraints = config.get('enable_stereo_constraints', True)
        self.max_disparity = config.get('max_disparity', 128)
        self.stereo_y_tolerance = config.get('stereo_y_tolerance', 2.0)
        
        # 从配置中获取模型路径
        model_path = config.get('model_path', None)
        self.load_model(model_path)
        
    def load_model(self, model_path: str = None):
        """加载EfficientLoFTR模型"""
        if not ELOFTR_AVAILABLE:
            print("EfficientLoFTR not available, skipping model loading")
            return False
            
        try:
            # 使用默认配置（与real_time_stereo_matcher.py相同方式）
            _default_cfg = deepcopy(full_default_cfg)
            
            # 初始化模型
            self.model = LoFTR(config=_default_cfg)
            
            # 加载预训练权重（与real_time_stereo_matcher.py相同方式）
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 直接加载state_dict
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                # 关键：重参数化（这是real_time_stereo_matcher.py的关键步骤）
                self.model = reparameter(self.model)
                print(f"Loaded and reparameterized EfficientLoFTR model from {model_path}")
            else:
                print("Warning: No model path provided, using random weights")
            
            # 安全的设备加载 
            try:
                if self.device == 'cuda' and not torch.cuda.is_available():
                    print("CUDA not available, falling back to CPU")
                    self.device = 'cpu'
                
                print(f"Loading model to device: {self.device}")
                self.model = self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"GPU loading failed: {e}, trying CPU...")
                self.device = 'cpu'
                self.model = self.model.to(self.device)
                self.model.eval()
            
        except Exception as e:
            print(f"Failed to load EfficientLoFTR model: {e}")
            self.model = None
    
    def match_frames(self, img0: np.ndarray, img1: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """使用EfficientLoFTR执行特征匹配"""
        if self.model is None:
            print("EfficientLoFTR model not loaded")
            return None
        
        try:
            # 预处理图像
            batch = self._preprocess_images(img0, img1)
            
            # 模型推理
            with torch.no_grad():
                self.model(batch)
            
            # 提取匹配结果
            matches = self._extract_matches(batch, img0.shape[:2])
            
            # 如果是立体匹配，应用立体约束
            if kwargs.get('stereo_matching', False) and self.enable_stereo_constraints:
                matches = self._apply_stereo_constraints(matches)
            
            return matches
            
        except Exception as e:
            print(f"Feature matching failed: {e}")
            return None
    
    def match_stereo_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> Optional[Dict[str, Any]]:
        """专门用于立体图像对匹配"""
        return self.match_frames(left_img, right_img, stereo_matching=True)
    
    def match_pair(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[Optional[list], float]:
        """兼容性方法 - 返回匹配点列表和平均置信度"""
        matches = self.match_frames(img0, img1)
        
        if matches is None or matches['num_matches'] == 0:
            print(f"[LoFTR匹配] 未找到匹配")
            return None, 0.0
        
        print(f"[LoFTR匹配] 原始匹配数: {matches['num_matches']}")
        
        # 与real_time_stereo_matcher.py相同，先获取所有匹配，再进行置信度过滤
        kpts0 = matches['keypoints0']
        kpts1 = matches['keypoints1']
        confidence = matches['confidence']
        
        # 使用更低的置信度阈值进行过滤（与real_time_stereo_matcher.py保持一致）
        conf_thresh = 0.15  # 降低阈值
        mask = confidence > conf_thresh
        
        if np.sum(mask) == 0:
            print(f"[LoFTR匹配] 置信度过滤后无匹配点（阈值: {conf_thresh}）")
            return None, 0.0
        
        # 过滤低置信度匹配
        kpts0_filtered = kpts0[mask]
        kpts1_filtered = kpts1[mask]
        conf_filtered = confidence[mask]
        
        print(f"[LoFTR匹配] 过滤后匹配数: {len(kpts0_filtered)}")
        
        match_list = []
        for i in range(len(kpts0_filtered)):
            match_list.append([kpts0_filtered[i], kpts1_filtered[i]])
        
        avg_confidence = float(np.mean(conf_filtered))
        return match_list, avg_confidence
    
    def _preprocess_images(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, torch.Tensor]:
        """预处理输入图像"""
        # 转换为灰度图
        if len(img0.shape) == 3:
            img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        else:
            img0_gray = img0.copy()
            
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1.copy()
        
        # 调整图像大小（确保能被32整除，与real_time_stereo_matcher.py相同）
        target_h, target_w = self.resize_to[1], self.resize_to[0]  # [width, height] -> [height, width]
        
        # 确保尺寸能被32整除
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32
        
        img0_resized = cv2.resize(img0_gray, (target_w, target_h))
        img1_resized = cv2.resize(img1_gray, (target_w, target_h))
        
        print(f"[LoFTR预处理] 原始尺寸: {img0_gray.shape}, 调整后: {img0_resized.shape}")
        
        # 转换为torch张量 [1, 1, H, W]
        img0_tensor = torch.from_numpy(img0_resized).float()[None, None] / 255.0
        img1_tensor = torch.from_numpy(img1_resized).float()[None, None] / 255.0
        
        # 移动到设备
        img0_tensor = img0_tensor.to(self.device)
        img1_tensor = img1_tensor.to(self.device)
        
        # 构建批次数据
        batch = {
            'image0': img0_tensor,
            'image1': img1_tensor
        }
        
        return batch
    
    def _extract_matches(self, batch: Dict[str, torch.Tensor], original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """从模型输出中提取匹配结果"""
        if 'mkpts0_f' not in batch or 'mkpts1_f' not in batch:
            return {
                'keypoints0': np.array([]).reshape(0, 2),
                'keypoints1': np.array([]).reshape(0, 2),
                'confidence': np.array([]),
                'matches0to1': np.array([]),
                'num_matches': 0
            }
        
        kpts0 = batch['mkpts0_f'].cpu().numpy()  # [N, 2]
        kpts1 = batch['mkpts1_f'].cpu().numpy()  # [N, 2]
        conf = batch['mconf'].cpu().numpy()      # [N]
        
        # 将坐标缩放回原图尺寸
        h, w = original_shape
        target_h = (self.resize_to[1] // 32) * 32  
        target_w = (self.resize_to[0] // 32) * 32
        scale_h = h / target_h
        scale_w = w / target_w
        
        kpts0[:, 0] *= scale_w
        kpts0[:, 1] *= scale_h
        kpts1[:, 0] *= scale_w
        kpts1[:, 1] *= scale_h
        
        # 限制最大匹配点数量
        if len(kpts0) > self.max_keypoints:
            # 按置信度排序，保留top-k
            top_k_indices = np.argsort(conf)[::-1][:self.max_keypoints]
            kpts0 = kpts0[top_k_indices]
            kpts1 = kpts1[top_k_indices]
            conf = conf[top_k_indices]
        
        matches = {
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'confidence': conf,
            'matches0to1': np.arange(len(kpts0)),
            'num_matches': len(kpts0)
        }
        
        return matches
    
    def filter_matches(self, matches: Dict[str, Any], 
                      min_confidence: float = 0.2) -> Dict[str, Any]:
        """根据置信度过滤匹配点"""
        confidence = matches['confidence']
        mask = confidence >= min_confidence
        
        filtered_matches = {
            'keypoints0': matches['keypoints0'][mask],
            'keypoints1': matches['keypoints1'][mask],
            'confidence': matches['confidence'][mask],
            'matches0to1': matches['matches0to1'][mask],
            'num_matches': int(np.sum(mask))
        }
        
        return filtered_matches
    
    def _apply_stereo_constraints(self, matches: Dict[str, Any]) -> Dict[str, Any]:
        """应用立体约束过滤匹配点"""
        if matches['num_matches'] == 0:
            return matches
        
        kpts0 = matches['keypoints0']
        kpts1 = matches['keypoints1']
        conf = matches['confidence']
        
        # 立体约束：左右图像点的y坐标应该接近
        y_diff = np.abs(kpts0[:, 1] - kpts1[:, 1])
        y_mask = y_diff < self.stereo_y_tolerance
        
        # 视差约束：左图点x坐标应大于右图点x坐标（正视差）
        disparity = kpts0[:, 0] - kpts1[:, 0]
        disp_mask = (disparity > 0) & (disparity < self.max_disparity)
        
        # 组合约束
        valid_mask = y_mask & disp_mask
        
        filtered_matches = {
            'keypoints0': kpts0[valid_mask],
            'keypoints1': kpts1[valid_mask],
            'confidence': conf[valid_mask],
            'matches0to1': matches['matches0to1'][valid_mask],
            'num_matches': int(np.sum(valid_mask)),
            'disparity': disparity[valid_mask]  # 额外信息
        }
        
        return filtered_matches
    
    def get_match_quality(self, matches: Dict[str, Any]) -> Dict[str, float]:
        """评估匹配质量"""
        if matches['num_matches'] == 0:
            return {'mean_confidence': 0.0, 'match_ratio': 0.0, 'disparity_consistency': 0.0}
        
        confidence = matches['confidence']
        
        quality = {
            'mean_confidence': float(np.mean(confidence)),
            'median_confidence': float(np.median(confidence)),
            'high_conf_ratio': float(np.sum(confidence > 0.8) / len(confidence)),
            'match_ratio': min(matches['num_matches'] / 1000.0, 1.0)  # 归一化匹配数量
        }
        
        # 如果有视差信息，计算视差一致性
        if 'disparity' in matches and len(matches['disparity']) > 1:
            disparity = matches['disparity']
            disparity_std = np.std(disparity)
            # 视差标准差越小，一致性越好
            quality['disparity_consistency'] = max(0.0, 1.0 - disparity_std / 50.0)
        else:
            quality['disparity_consistency'] = 0.0
        
        return quality
    
    def visualize_matches(self, img0: np.ndarray, img1: np.ndarray, 
                         matches: Dict[str, Any], max_matches: int = 100) -> np.ndarray:
        """可视化匹配结果"""
        if matches['num_matches'] == 0:
            # 返回并排显示的图像
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            h = max(h0, h1)
            combined = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
            combined[:h0, :w0] = img0 if len(img0.shape) == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            combined[:h1, w0:] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            return combined
        
        # 限制显示的匹配数量
        num_show = min(max_matches, matches['num_matches'])
        indices = np.random.choice(matches['num_matches'], num_show, replace=False)
        
        kpts0 = matches['keypoints0'][indices]
        kpts1 = matches['keypoints1'][indices]
        conf = matches['confidence'][indices]
        
        # 创建并排图像
        if len(img0.shape) == 3:
            img0_show = img0.copy()
        else:
            img0_show = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
            
        if len(img1.shape) == 3:
            img1_show = img1.copy()
        else:
            img1_show = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        h0, w0 = img0_show.shape[:2]
        h1, w1 = img1_show.shape[:2]
        h = max(h0, h1)
        
        combined = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
        combined[:h0, :w0] = img0_show
        combined[:h1, w0:] = img1_show
        
        # 绘制匹配点和连线
        for i, (pt0, pt1, c) in enumerate(zip(kpts0, kpts1, conf)):
            # 根据置信度设置颜色
            color = (int(255 * c), int(255 * (1-c)), 0)  # 高置信度偏红，低置信度偏绿
            
            # 绘制关键点
            cv2.circle(combined, (int(pt0[0]), int(pt0[1])), 3, color, -1)
            cv2.circle(combined, (int(pt1[0] + w0), int(pt1[1])), 3, color, -1)
            
            # 绘制连线
            cv2.line(combined, 
                    (int(pt0[0]), int(pt0[1])), 
                    (int(pt1[0] + w0), int(pt1[1])), 
                    color, 1)
        
        return combined