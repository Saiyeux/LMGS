"""
LMGS 3D Reconstruction Package

一个模块化的3D重建系统，集成EfficientLoFTR和MonoGS技术
"""

__version__ = "1.0.0"

from .camera import SmartCameraManager
from .reconstruction import HybridAdvanced3DReconstructor
from .visualization import Interactive3DViewer, UltimateVisualization
from .utils import dependencies

__all__ = [
    'SmartCameraManager',
    'HybridAdvanced3DReconstructor', 
    'Interactive3DViewer',
    'UltimateVisualization',
    'dependencies'
]