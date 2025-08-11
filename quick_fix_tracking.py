#!/usr/bin/env python3
"""
快速修复Hybrid Tracking卡住问题
"""

import sys
import torch
from pathlib import Path

def fix_loftr_matcher():
    """修复EfficientLoFTR matcher的GPU加载问题"""
    print("Fixing EfficientLoFTR matcher...")
    
    matcher_file = Path("hybrid_slam/matchers/loftr_matcher.py")
    
    # 读取原文件
    with open(matcher_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复GPU加载问题
    original_to_device = '''            self.model.to(self.device)
            self.model.eval()'''
    
    fixed_to_device = '''            # 安全的设备加载 
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
                self.model.eval()'''
    
    if original_to_device in content:
        content = content.replace(original_to_device, fixed_to_device)
        
        # 备份并保存
        backup_file = matcher_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        with open(matcher_file, 'w', encoding='utf-8') as f:
            content_lines = content.split('\n')
            new_content = '\n'.join(content_lines)
            f.write(new_content)
        
        print(f"Fixed matcher file, backup saved to: {backup_file}")
        return True
    
    print("Could not find the problematic code section")
    return False

def create_cpu_only_config():
    """创建仅使用CPU的配置文件"""
    print("Creating CPU-only configuration...")
    
    cpu_config = {
        'input': {
            'source': 'camera',
            'camera': {
                'left_device': 0,
                'right_device': 1,
                'resolution': [640, 480]
            }
        },
        'frontend': {
            'loftr_config': {
                'model_type': 'outdoor',
                'device': 'cpu',  # 强制使用CPU
                'resize_to': [640, 480],
                'match_threshold': 0.3
            },
            'pnp_solver': {
                'method': 'SOLVEPNP_ITERATIVE',
                'confidence': 0.99,
                'reprojection_threshold': 2.0
            }
        },
        'monogs': {
            'cam': {
                'H': 480, 'W': 640,
                'fx': 525.0, 'fy': 525.0,
                'cx': 320.0, 'cy': 240.0
            }
        },
        'visualization': True,
        'visualization_config': {
            'window_size': [1200, 800]
        },
        'performance_targets': {
            'target_fps': 10,  # 降低FPS以适应CPU处理
            'max_memory_gb': 4
        }
    }
    
    import yaml
    config_file = Path("configs/cpu_only_config.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(cpu_config, f, default_flow_style=False)
    
    print(f"CPU-only config saved: {config_file}")
    return config_file

def add_timeout_to_frontend():
    """为前端添加超时保护"""
    print("Adding timeout protection to frontend...")
    
    frontend_file = Path("hybrid_slam/frontend/hybrid_frontend.py")
    
    with open(frontend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在初始化方法中添加超时保护
    if 'def __init__(self, config: Dict[str, Any]):' in content and 'timeout_protection' not in content:
        timeout_code = '''        # 超时保护
        import threading
        import time
        
        def init_with_timeout():
            try:
                print("Initializing hybrid tracking...")
                # 初始化组件
                self.feature_matcher = EfficientLoFTRMatcher(config.get('loftr_config', {}))
                print("Feature matcher initialized")
                self.pnp_solver = PnPSolver(config.get('pnp_solver', {}))  
                print("PnP solver initialized")
            except Exception as e:
                print(f"Frontend initialization failed: {e}")
                self.feature_matcher = None
                self.pnp_solver = None
        
        # 使用线程和超时
        init_thread = threading.Thread(target=init_with_timeout)
        init_thread.daemon = True
        init_thread.start()
        init_thread.join(timeout=30)  # 30秒超时
        
        if init_thread.is_alive():
            print("Frontend initialization timeout - using fallback mode")
            self.feature_matcher = None
            self.pnp_solver = PnPSolver(config.get('pnp_solver', {}))
        
        # timeout_protection marker'''
        
        # 替换初始化代码
        original_init = '''        # 初始化组件
        self.feature_matcher = EfficientLoFTRMatcher(config.get('EfficientLoFTR', {}))
        self.pnp_solver = PnPSolver(config.get('PnPSolver', {}))'''
        
        if original_init in content:
            content = content.replace(original_init, timeout_code)
            
            # 保存修改
            with open(frontend_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("Timeout protection added to frontend")
            return True
    
    print("Frontend already has timeout protection or cannot be modified")
    return False

def quick_test():
    """快速测试修复结果"""
    print("\nTesting the fixes...")
    
    try:
        # 测试torch和CUDA
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # 测试导入
        from hybrid_slam.frontend.hybrid_frontend import HybridFrontEnd
        print("Frontend import successful")
        
        # 创建简单配置测试
        simple_config = {
            'loftr_config': {
                'device': 'cpu',
                'model_path': None
            },
            'pnp_solver': {}
        }
        
        print("Creating frontend with CPU config...")
        frontend = HybridFrontEnd(simple_config)
        print("Frontend creation successful!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Quick Fix for Hybrid Tracking")
    print("=" * 40)
    
    # 执行修复
    step1 = fix_loftr_matcher()
    step2 = create_cpu_only_config()  
    step3 = add_timeout_to_frontend()
    
    print("\n" + "=" * 40)
    print("Fix Summary:")
    print(f"Matcher fix: {'OK' if step1 else 'SKIP'}")
    print(f"CPU config: {'OK' if step2 else 'FAIL'}")
    print(f"Timeout protection: {'OK' if step3 else 'SKIP'}")
    
    # 测试修复结果
    test_ok = quick_test()
    
    print("\n" + "=" * 40)
    if test_ok:
        print("SUCCESS! Try running:")
        print("python run_dual_camera_3d.py --config configs/cpu_only_config.yaml")
    else:
        print("Still has issues. Check GPU drivers and PyTorch installation.")
    print("=" * 40)