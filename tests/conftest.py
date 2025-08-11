"""
pytest配置文件
定义测试夹具和全局配置
"""

import pytest
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_config():
    """样例配置fixture"""
    return {
        'EfficientLoFTR': {
            'model_type': 'opt',
            'confidence_threshold': 0.2
        },
        'PnPSolver': {
            'ransac_threshold': 2.0,
            'min_inliers': 20
        }
    }

@pytest.fixture
def sample_images():
    """样例图像fixture"""
    import numpy as np
    
    # 创建样例图像
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    return img1, img2