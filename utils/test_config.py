#!/usr/bin/env python3
"""
测试配置系统
验证配置加载和继承功能
"""

from hybrid_slam.utils.config_manager import ConfigManager
from pathlib import Path
import json

def test_config_loading():
    """测试配置加载功能"""
    print("Testing configuration loading...")
    
    # 测试基础配置
    base_config_path = "configs/base/hybrid_slam_base.yaml"
    if Path(base_config_path).exists():
        try:
            config = ConfigManager.load_config(base_config_path)
            print(f"[OK] Base config loaded successfully")
            print(f"  Device: {config.get('device', 'N/A')}")
            print(f"  EfficientLoFTR model: {config.get('EfficientLoFTR', {}).get('model_path', 'N/A')}")
        except Exception as e:
            print(f"[FAIL] Base config loading failed: {e}")
    else:
        print(f"[FAIL] Base config file not found: {base_config_path}")
    
    # 测试主配置（含继承）
    main_config_path = "configs/hybrid_slam_config.yaml"
    if Path(main_config_path).exists():
        try:
            config = ConfigManager.load_config(main_config_path)
            print(f"[OK] Main config loaded successfully")
            print(f"  Project: {config.get('project_name', 'N/A')}")
            print(f"  Mode: {config.get('mode', 'N/A')}")
            print(f"  Inherited device: {config.get('device', 'N/A')}")
            
            # 验证配置
            if ConfigManager.validate_config(config):
                print(f"[OK] Configuration validation passed")
            else:
                print(f"[WARN] Configuration validation failed")
            
        except Exception as e:
            print(f"[FAIL] Main config loading failed: {e}")
    else:
        print(f"[FAIL] Main config file not found: {main_config_path}")
    
    # 测试数据集配置
    dataset_config_path = "configs/datasets/tum_rgbd.yaml"
    if Path(dataset_config_path).exists():
        try:
            config = ConfigManager.load_config(dataset_config_path)
            print(f"[OK] Dataset config loaded successfully")
            print(f"  Dataset: {config.get('dataset', {}).get('name', 'N/A')}")
            print(f"  Camera fx: {config.get('dataset', {}).get('camera', {}).get('fx', 'N/A')}")
        except Exception as e:
            print(f"[FAIL] Dataset config loading failed: {e}")
    else:
        print(f"[FAIL] Dataset config file not found: {dataset_config_path}")

def test_config_saving():
    """测试配置保存功能"""
    print("\nTesting configuration saving...")
    
    # 创建测试配置
    test_config = {
        'test_param': 'test_value',
        'nested': {
            'param1': 123,
            'param2': 'nested_value'
        }
    }
    
    save_path = "configs/test_output.yaml"
    try:
        ConfigManager.save_config(test_config, save_path)
        print(f"[OK] Test config saved to {save_path}")
        
        # 验证保存的配置
        loaded_config = ConfigManager.load_config(save_path)
        if loaded_config == test_config:
            print(f"[OK] Saved config verification passed")
        else:
            print(f"[FAIL] Saved config verification failed")
            
        # 清理测试文件
        Path(save_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"[FAIL] Config saving failed: {e}")

def main():
    """主测试函数"""
    print("Hybrid SLAM Configuration System Test")
    print("=" * 50)
    
    test_config_loading()
    test_config_saving()
    
    print("\nConfiguration system test completed!")

if __name__ == "__main__":
    main()