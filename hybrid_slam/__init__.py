"""
Hybrid SLAM: EfficientLoFTR + OpenCV PnP + MonoGS Integration

A robust SLAM system combining feature matching, geometric constraints,
and neural radiance field rendering for enhanced tracking performance.
"""

from .version import __version__

# Main system components (to be implemented)
# from .core.slam_system import HybridSLAMSystem
# from .frontend.hybrid_frontend import HybridFrontEnd
# from .matchers.loftr_matcher import EfficientLoFTRMatcher
# from .solvers.pnp_solver import PnPSolver
# from .utils.config_manager import ConfigManager

# Placeholder exports for now
__all__ = [
    '__version__',
    # 'HybridSLAMSystem',
    # 'HybridFrontEnd', 
    # 'EfficientLoFTRMatcher',
    # 'PnPSolver',
    # 'ConfigManager'
]

# Package metadata
__author__ = "LMGS Team"
__email__ = "team@lmgs.dev"

def get_version():
    """获取版本信息"""
    return __version__

# def create_slam_system(config_path: str):
#     """便捷的SLAM系统创建函数"""
#     config = ConfigManager.load_config(config_path)
#     return HybridSLAMSystem(config)