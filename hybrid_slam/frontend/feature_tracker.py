"""
特征跟踪器
基于特征匹配的跟踪实现
"""

from ..core.base_tracker import BaseTracker

class FeatureTracker(BaseTracker):
    """特征跟踪器 - 待实现"""
    
    def __init__(self, config):
        super().__init__(config)
        
    def track(self, current_frame, reference_frame=None):
        """基于特征匹配的跟踪"""
        pass
    
    def is_tracking_reliable(self):
        """判断特征跟踪是否可靠"""
        pass