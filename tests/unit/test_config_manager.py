#!/usr/bin/env python3
"""
ConfigManager单元测试
测试配置加载、继承和验证功能
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from hybrid_slam.utils.config_manager import ConfigManager

class TestConfigManager:
    """ConfigManager测试类"""
    
    def test_load_simple_config(self):
        """测试加载简单配置文件"""
        # 创建临时配置文件
        config_data = {
            'device': 'cuda',
            'batch_size': 4,
            'model': {
                'name': 'test_model',
                'lr': 0.01
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # 加载配置
            loaded_config = ConfigManager.load_config(config_path)
            
            # 验证加载结果
            assert loaded_config['device'] == 'cuda'
            assert loaded_config['batch_size'] == 4
            assert loaded_config['model']['name'] == 'test_model'
            assert loaded_config['model']['lr'] == 0.01
            
        finally:
            # 清理临时文件
            Path(config_path).unlink(missing_ok=True)
    
    def test_config_inheritance(self):
        """测试配置继承功能"""
        # 创建父配置
        parent_config = {
            'device': 'cuda',
            'batch_size': 4,
            'model': {
                'name': 'parent_model',
                'lr': 0.01,
                'epochs': 100
            }
        }
        
        # 创建子配置（继承并覆盖部分参数）
        child_config = {
            'inherit_from': 'parent.yaml',
            'batch_size': 8,  # 覆盖父配置
            'model': {
                'lr': 0.001,  # 覆盖父配置
                'optimizer': 'adam'  # 新增参数
            },
            'new_param': 'child_value'  # 新增参数
        }
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = Path(temp_dir) / 'parent.yaml'
            child_path = Path(temp_dir) / 'child.yaml'
            
            # 保存配置文件
            with open(parent_path, 'w') as f:
                yaml.dump(parent_config, f)
            with open(child_path, 'w') as f:
                yaml.dump(child_config, f)
            
            # 加载子配置
            loaded_config = ConfigManager.load_config(str(child_path))
            
            # 验证继承结果
            assert loaded_config['device'] == 'cuda'  # 继承自父配置
            assert loaded_config['batch_size'] == 8   # 子配置覆盖
            assert loaded_config['model']['name'] == 'parent_model'  # 继承自父配置
            assert loaded_config['model']['lr'] == 0.001  # 子配置覆盖
            assert loaded_config['model']['epochs'] == 100  # 继承自父配置
            assert loaded_config['model']['optimizer'] == 'adam'  # 子配置新增
            assert loaded_config['new_param'] == 'child_value'  # 子配置新增
            assert 'inherit_from' not in loaded_config  # 继承标记应被移除
    
    def test_merge_configs(self):
        """测试配置合并功能"""
        base_config = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            }
        }
        
        override_config = {
            'b': {
                'c': 20,  # 覆盖
                'e': 4    # 新增
            },
            'f': 5  # 新增
        }
        
        merged = ConfigManager.merge_configs(base_config, override_config)
        
        assert merged['a'] == 1
        assert merged['b']['c'] == 20  # 覆盖
        assert merged['b']['d'] == 3   # 保留
        assert merged['b']['e'] == 4   # 新增
        assert merged['f'] == 5        # 新增
    
    def test_validate_config(self):
        """测试配置验证功能"""
        # 有效配置
        valid_config = {
            'EfficientLoFTR': {},
            'PnPSolver': {},
            'HybridTracking': {}
        }
        
        assert ConfigManager.validate_config(valid_config) == True
        
        # 无效配置（缺少必需部分）
        invalid_config = {
            'EfficientLoFTR': {},
            'PnPSolver': {}
            # 缺少 HybridTracking
        }
        
        assert ConfigManager.validate_config(invalid_config) == False
    
    def test_save_and_load_config(self):
        """测试配置保存和加载"""
        test_config = {
            'test_param': 'test_value',
            'nested': {
                'param1': 123,
                'param2': [1, 2, 3],
                'param3': {
                    'deep_param': 'deep_value'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # 保存配置
            ConfigManager.save_config(test_config, config_path)
            
            # 加载配置
            loaded_config = ConfigManager.load_config(config_path)
            
            # 验证一致性
            assert loaded_config == test_config
            
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config('nonexistent_config.yaml')