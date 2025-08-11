#!/usr/bin/env python3
"""
环境设置脚本
检查和设置Hybrid SLAM运行环境
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} not found")
        return False

def check_cuda():
    """检查CUDA环境"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_thirdparty_modules():
    """检查第三方模块"""
    project_root = Path(__file__).parent.parent
    
    # 检查EfficientLoFTR
    eloftr_path = project_root / "thirdparty" / "EfficientLoFTR"
    if eloftr_path.exists():
        print("✅ EfficientLoFTR found")
        
        # 检查权重文件
        weights_dir = eloftr_path / "weights"
        if weights_dir.exists():
            weight_files = list(weights_dir.glob("*.ckpt"))
            if weight_files:
                print(f"   Found {len(weight_files)} model files")
            else:
                print("   ⚠️  No model files found, run download_models.py")
        else:
            print("   ❌ Weights directory not found")
    else:
        print("❌ EfficientLoFTR not found")
    
    # 检查MonoGS
    monogs_path = project_root / "thirdparty" / "MonoGS"
    if monogs_path.exists():
        print("✅ MonoGS found")
        
        # 检查CUDA扩展
        try:
            sys.path.append(str(monogs_path))
            from diff_gaussian_rasterization import _C
            print("   ✅ CUDA extensions compiled")
        except ImportError as e:
            print(f"   ❌ CUDA extensions not compiled: {e}")
    else:
        print("❌ MonoGS not found")

def main():
    """主函数"""
    print("="*60)
    print("Hybrid SLAM Environment Check")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Python版本检查
    total_checks += 1
    if check_python_version():
        checks_passed += 1
    
    print("\nChecking required packages...")
    
    # 必需包检查
    required_packages = [
        ('torch', 'torch'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('yaml', 'yaml'),
        ('tqdm', 'tqdm'),
    ]
    
    for package_name, import_name in required_packages:
        total_checks += 1
        if check_package(package_name, import_name):
            checks_passed += 1
    
    print("\nChecking CUDA environment...")
    total_checks += 1
    if check_cuda():
        checks_passed += 1
    
    print("\nChecking thirdparty modules...")
    check_thirdparty_modules()
    
    print("\n" + "="*60)
    print(f"Environment Check Summary: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("🎉 Environment setup complete! Ready to run Hybrid SLAM.")
        return 0
    else:
        print("⚠️  Some checks failed. Please install missing dependencies.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        print("\nTo compile CUDA extensions, run:")
        print("  cd thirdparty/MonoGS")
        print("  pip install ./submodules/simple-knn")
        print("  pip install ./submodules/diff-gaussian-rasterization")
        return 1

if __name__ == "__main__":
    exit(main())