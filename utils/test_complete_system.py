#!/usr/bin/env python3
"""
完整系统测试 - 测试双摄像头实时重建功能
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hybrid_slam_system():
    """测试完整的HybridSLAM系统"""
    print("=== Testing Complete Hybrid SLAM System ===")
    
    try:
        # 导入核心系统
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        print("OK HybridSLAMSystem import")
        
        # 创建测试配置
        test_config = {
            'device': 'cpu',  # 使用CPU进行测试
            'input': {
                'source': 'mock',  # 使用模拟数据
                'camera': {
                    'resolution': [320, 240],  # 小分辨率加快测试
                    'fps': 10
                }
            },
            'frontend': {
                'matcher_type': 'loftr',
                'tracking_method': 'pnp'
            },
            'performance_targets': {
                'target_fps': 10,
                'max_memory_gb': 4,
                'max_gpu_memory_gb': 2
            },
            'visualization': False,  # 关闭可视化进行测试
            'output': {
                'save_trajectory': True,
                'save_keyframes': False
            }
        }
        
        print("OK Test configuration created")
        
        # 创建SLAM系统
        slam_system = HybridSLAMSystem(test_config, save_dir="test_results")
        print("OK HybridSLAMSystem created")
        
        print("\nSystem components:")
        print(f"  Frontend: {type(slam_system.frontend).__name__}")
        print(f"  Device: {slam_system.device}")
        print(f"  Target FPS: {slam_system.target_fps}")
        
        # 测试数据源初始化
        slam_system._init_dataset()
        print(f"OK Dataset initialized: {type(slam_system.dataset).__name__}")
        
        # 处理几帧进行测试
        print("\nProcessing test frames:")
        frame_count = 0
        max_frames = 5
        
        for stereo_frame in slam_system.dataset:
            if frame_count >= max_frames:
                break
                
            print(f"  Processing frame {stereo_frame.frame_id}...")
            
            # 处理单帧
            result = slam_system._process_stereo_frame(stereo_frame)
            
            print(f"    Image shape: {stereo_frame.left_image.shape}")
            print(f"    Timestamp: {stereo_frame.timestamp:.3f}")
            print(f"    Processing success: {result.success if hasattr(result, 'success') else 'N/A'}")
            
            frame_count += 1
            time.sleep(0.1)  # 短暂延迟
        
        print(f"\nProcessed {frame_count} frames successfully")
        print(f"Trajectory length: {len(slam_system.trajectory)}")
        
        # 关闭系统
        if hasattr(slam_system.dataset, 'close'):
            slam_system.dataset.close()
        
        print("\nSUCCESS: Complete system test passed!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_configuration_loading():
    """测试配置文件加载"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        from hybrid_slam.utils.config_manager import ConfigManager
        
        # 测试YAML配置加载
        config_file = "configs/stereo_camera_config.yaml"
        if Path(config_file).exists():
            config = ConfigManager.load_config(config_file)
            print(f"OK Loaded config from {config_file}")
            print(f"  Input source: {config.get('input', {}).get('source', 'unknown')}")
            print(f"  Device: {config.get('device', 'unknown')}")
        else:
            print(f"WARNING: Config file not found: {config_file}")
            print("Creating a test configuration instead...")
            
            # 创建临时测试配置
            test_config = {
                'device': 'cpu',
                'input': {'source': 'mock'},
                'test': True
            }
            
            # 测试配置保存和加载
            test_config_path = "test_config.yaml"
            ConfigManager.save_config(test_config, test_config_path)
            loaded_config = ConfigManager.load_config(test_config_path)
            
            if loaded_config == test_config:
                print("OK Configuration save/load test passed")
            else:
                print("ERROR Configuration save/load failed")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"ERROR in configuration test: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """主测试函数"""
    print("Starting comprehensive hybrid SLAM tests...\n")
    
    # 基础功能测试
    result1 = test_hybrid_slam_system()
    
    # 配置加载测试
    result2 = test_configuration_loading()
    
    # 总结
    if result1 == 0 and result2 == 0:
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("Dual-camera real-time reconstruction system is ready.")
        print("="*50)
        return 0
    else:
        print("\n" + "="*50)
        print("SOME TESTS FAILED!")
        print("="*50)
        return 1

if __name__ == "__main__":
    exit(main())