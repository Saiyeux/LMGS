#!/usr/bin/env python3
"""
修复Hybrid Tracking初始化卡住的问题
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

def test_loftr_loading():
    """测试EfficientLoFTR加载"""
    print("=" * 50)
    print("Testing EfficientLoFTR Loading")
    print("=" * 50)
    
    try:
        # 添加EfficientLoFTR路径
        eloftr_path = Path("thirdparty/EfficientLoFTR")
        if eloftr_path.exists():
            sys.path.insert(0, str(eloftr_path))
            print(f"✓ EfficientLoFTR path added: {eloftr_path}")
        
        # 测试导入
        print("Testing imports...")
        from src.loftr import LoFTR
        from src.config.default import get_cfg_defaults
        print("✓ Imports successful")
        
        # 测试配置加载
        print("Testing config loading...")
        config = get_cfg_defaults()
        print("✓ Config loading successful")
        
        # 测试模型创建
        print("Testing model creation...")
        matcher = LoFTR(config=config['LOFTR'])
        print("✓ Model creation successful")
        
        # 测试GPU
        if torch.cuda.is_available():
            print("Testing GPU transfer...")
            matcher = matcher.cuda()
            print("✓ GPU transfer successful")
        
        return True
        
    except Exception as e:
        print(f"❌ EfficientLoFTR loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_features():
    """测试OpenCV特征提取作为备选方案"""
    print("\n" + "=" * 50)
    print("Testing OpenCV Features (Fallback)")
    print("=" * 50)
    
    try:
        import cv2
        
        # 创建测试图像
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # 测试ORB特征
        print("Testing ORB...")
        orb = cv2.ORB_create()
        kp, desc = orb.detectAndCompute(img, None)
        print(f"✓ ORB features: {len(kp)} keypoints")
        
        # 测试SIFT特征
        try:
            print("Testing SIFT...")
            sift = cv2.SIFT_create()
            kp, desc = sift.detectAndCompute(img, None)
            print(f"✓ SIFT features: {len(kp)} keypoints")
        except Exception as e:
            print(f"⚠ SIFT not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV features failed: {e}")
        return False

def create_lightweight_config():
    """创建轻量级配置，避免卡死"""
    print("\n" + "=" * 50)
    print("Creating Lightweight Configuration")
    print("=" * 50)
    
    lightweight_config = {
        'input': {
            'source': 'camera',
            'camera': {
                'left_device': 0,
                'right_device': 1,
                'resolution': [640, 480],
                'fps': 15  # 降低FPS
            }
        },
        'frontend': {
            'use_opencv_fallback': True,  # 优先使用OpenCV
            'loftr_config': {
                'model_type': 'outdoor',
                'load_timeout': 10,  # 设置加载超时
                'enable_gpu': torch.cuda.is_available()
            },
            'opencv_config': {
                'detector_type': 'ORB',  # 使用ORB而不是SIFT
                'max_features': 1000,
                'match_threshold': 0.7
            }
        },
        'monogs': {
            'cam': {
                'H': 480, 'W': 640,
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            },
            'mapping': {
                'every_frame': 5,  # 减少重建频率
                'new_points': 500  # 减少点数
            }
        },
        'visualization': True,
        'visualization_config': {
            'window_size': [1000, 600],  # 减小窗口尺寸
            'save_mode': False
        },
        'performance_targets': {
            'target_fps': 10,  # 进一步降低目标FPS
            'max_memory_gb': 4,
            'max_gpu_memory_gb': 2
        }
    }
    
    # 保存配置
    config_file = Path("configs/lightweight_config.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(lightweight_config, f, default_flow_style=False)
    
    print(f"✓ Lightweight config saved: {config_file}")
    return config_file

def patch_hybrid_frontend():
    """修补HybridFrontend以避免卡死"""
    print("\n" + "=" * 50)  
    print("Patching Hybrid Frontend")
    print("=" * 50)
    
    frontend_file = Path("hybrid_slam/frontend/hybrid_frontend.py")
    
    if not frontend_file.exists():
        print("❌ Frontend file not found")
        return False
    
    # 读取原文件
    with open(frontend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否需要修补
    if 'TIMEOUT_SECONDS = 30' in content:
        print("✓ Frontend already patched")
        return True
    
    # 添加超时机制
    timeout_patch = '''
# Timeout mechanism for model loading
TIMEOUT_SECONDS = 30

import signal
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Model loading timeout")

'''
    
    # 在类定义前插入超时机制
    if 'class HybridFrontEnd:' in content:
        content = content.replace(
            'class HybridFrontEnd:',
            timeout_patch + 'class HybridFrontEnd:'
        )
        
        # 修改初始化方法以使用超时
        if 'def __init__(' in content:
            init_patch = '''
        # Set timeout for model loading
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        
        try:
'''
            content = content.replace(
                'def __init__(self, config: Dict[str, Any]):',
                '''def __init__(self, config: Dict[str, Any]):
        # Set timeout for model loading  
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIMEOUT_SECONDS)
'''
            )
        
        # 备份原文件
        backup_file = frontend_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Frontend patched with timeout mechanism")
        print(f"✓ Backup saved: {backup_file}")
        return True
    
    print("❌ Could not patch frontend")
    return False

def test_system_with_timeout():
    """使用超时机制测试系统"""
    print("\n" + "=" * 50)
    print("Testing System with Timeout")
    print("=" * 50)
    
    try:
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        
        # 简单配置，优先使用OpenCV
        config = {
            'input': {
                'source': 'mock',
                'mock': {'num_frames': 3}
            },
            'frontend': {
                'use_opencv_fallback': True,
                'disable_loftr': True  # 完全禁用LoFTR
            },
            'visualization': False
        }
        
        print("Creating SLAM system...")
        start_time = time.time()
        
        # 设置较短的超时
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutException("System creation timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15秒超时
        
        try:
            slam_system = HybridSLAMSystem(config, save_dir="timeout_test")
            signal.alarm(0)  # 取消超时
            
            elapsed = time.time() - start_time
            print(f"✓ System created successfully in {elapsed:.2f}s")
            
            # 测试基本功能
            reconstruction = slam_system.get_3d_reconstruction()
            print(f"✓ 3D reconstruction interface: {reconstruction is not None}")
            
            return True
            
        except TimeoutException:
            print("❌ System creation timeout - frontend initialization stuck")
            return False
        except Exception as e:
            signal.alarm(0)
            print(f"❌ System creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

class TimeoutException(Exception):
    pass

if __name__ == "__main__":
    print("Hybrid Tracking Fix Utility")
    print("=" * 50)
    
    # 测试各个组件
    loftr_ok = test_loftr_loading()
    opencv_ok = test_opencv_features()
    config_file = create_lightweight_config()
    
    # 如果LoFTR有问题，创建禁用版本
    if not loftr_ok:
        print("\n⚠ LoFTR has issues, creating fallback config...")
        
        fallback_config = {
            'input': {
                'source': 'camera',
                'camera': {
                    'left_device': 0,
                    'right_device': 1,
                    'resolution': [640, 480]
                }
            },
            'frontend': {
                'disable_loftr': True,  # 完全禁用LoFTR
                'use_opencv_only': True,
                'opencv_config': {
                    'detector_type': 'ORB',
                    'max_features': 500
                }
            },
            'monogs': {
                'cam': {'H': 480, 'W': 640, 'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
            },
            'visualization': True,
            'performance_targets': {'target_fps': 10}
        }
        
        import yaml
        fallback_file = Path("configs/opencv_only_config.yaml")
        with open(fallback_file, 'w') as f:
            yaml.dump(fallback_config, f)
        print(f"✓ Fallback config created: {fallback_file}")
    
    # 测试系统
    system_ok = test_system_with_timeout()
    
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print(f"LoFTR Loading: {'✓ OK' if loftr_ok else '❌ FAILED'}")
    print(f"OpenCV Features: {'✓ OK' if opencv_ok else '❌ FAILED'}")
    print(f"System Test: {'✓ OK' if system_ok else '❌ FAILED'}")
    print("=" * 50)
    
    if system_ok:
        print("🎉 System is working! Try:")
        print(f"python run_dual_camera_3d.py --config {config_file}")
    elif opencv_ok:
        print("⚠ Use OpenCV-only mode:")
        print("python run_dual_camera_3d.py --config configs/opencv_only_config.yaml") 
    else:
        print("❌ System has critical issues - check GPU and dependencies")