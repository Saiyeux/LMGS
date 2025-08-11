"""
TUM数据集扩展
增强TUM数据集支持
"""

# TODO: 继承MonoGS的TUMDataset并扩展功能

class TUMDatasetExtended:
    """TUM数据集扩展类 - 待实现"""
    
    def __init__(self, config):
        """初始化TUM数据集"""
        # TODO: 继承并扩展thirdparty/MonoGS的TUMDataset
        pass
    
    def get_frame_with_features(self, idx: int):
        """获取带有预提取特征的帧"""
        # TODO: 实现特征预提取和缓存
        pass
    
    def get_stereo_pair(self, idx: int):
        """获取立体图像对（如果可用）"""
        # TODO: 支持双目数据
        pass