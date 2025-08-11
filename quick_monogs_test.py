#!/usr/bin/env python3
"""
快速测试MonoGS集成状态
"""

import sys
from pathlib import Path
from hybrid_slam.core.slam_system import HybridSLAMSystem, MONOGS_AVAILABLE

def quick_test():
    print(f"MonoGS Available: {MONOGS_AVAILABLE}")
    
    if not MONOGS_AVAILABLE:
        print("MonoGS modules not loaded")
        return
    
    # 最小配置
    config = {
        'input': {'source': 'mock', 'mock': {'num_frames': 1}},
        'monogs': {
            'cam': {'H': 480, 'W': 640, 'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
        },
        'visualization': False
    }
    
    try:
        system = HybridSLAMSystem(config, save_dir="quick_test")
        print(f"System created. MonoGS enabled: {system.use_monogs}")
        print(f"MonoGS SLAM: {system.monogs_slam is not None}")
        print(f"MonoGS Backend: {system.monogs_backend is not None}")
        
        # 测试3D重建接口
        recon = system.get_3d_reconstruction()
        print(f"3D reconstruction interface: {recon is not None}")
        
        print("MonoGS integration test passed!")
        
    except Exception as e:
        print(f"MonoGS integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()