"""
Dependency management for third-party libraries
"""

import sys
from pathlib import Path

# 添加第三方库到Python路径
def setup_paths():
    """设置第三方库路径"""
    base_path = Path(__file__).parent.parent.parent
    
    # 添加EfficientLoFTR到Python路径
    loftr_path = base_path / 'thirdparty' / 'EfficientLoFTR'
    if loftr_path.exists():
        sys.path.append(str(loftr_path))
    
    # 添加MonoGS到Python路径  
    monogs_path = base_path / 'thirdparty' / 'MonoGS'
    if monogs_path.exists():
        sys.path.append(str(monogs_path))

# 设置路径
setup_paths()

# 检查EfficientLoFTR可用性
try:
    from src.loftr import LoFTR, full_default_cfg, reparameter
    LOFTR_AVAILABLE = True
except ImportError as e:
    print(f"EfficientLoFTR不可用: {e}")
    LOFTR_AVAILABLE = False

# 检查MonoGS可用性
try:
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.general_utils import safe_state
    MONOGS_AVAILABLE = True
except ImportError as e:
    print(f"MonoGS不可用: {e}")
    MONOGS_AVAILABLE = False

def check_dependencies():
    """检查所有依赖项"""
    deps = {
        'EfficientLoFTR': LOFTR_AVAILABLE,
        'MonoGS': MONOGS_AVAILABLE
    }
    
    print("依赖项检查:")
    for name, available in deps.items():
        status = "✓ 可用" if available else "✗ 不可用"
        print(f"  {name}: {status}")
    
    return deps