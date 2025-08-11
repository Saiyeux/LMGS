"""
配置管理器
统一的配置文件加载和管理
"""

import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            config: 配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 处理继承关系
        if 'inherit_from' in config:
            parent_path = config_path.parent / config['inherit_from']
            parent_config = ConfigManager.load_config(parent_path)
            config = ConfigManager.merge_configs(parent_config, config)
            del config['inherit_from']  # 移除继承标记
        
        # 处理MonoGS配置继承
        if 'MonoGS' in config and 'inherit_from' in config['MonoGS']:
            monogs_parent_path = config_path.parent.parent.parent / config['MonoGS']['inherit_from']
            if monogs_parent_path.exists():
                with open(monogs_parent_path, 'r', encoding='utf-8') as f:
                    monogs_config = yaml.safe_load(f)
                config.update(monogs_config)  # 合并MonoGS配置
        
        return config
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_sections = ['EfficientLoFTR', 'PnPSolver', 'HybridTracking']
        
        for section in required_sections:
            if section not in config:
                print(f"Warning: Missing required section '{section}' in config")
                return False
        
        # 验证EfficientLoFTR配置
        loftr_config = config.get('EfficientLoFTR', {})
        if 'model_path' in loftr_config:
            model_path = Path(loftr_config['model_path'])
            if not model_path.exists():
                print(f"Warning: EfficientLoFTR model not found: {model_path}")
        
        return True
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """保存配置到文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)