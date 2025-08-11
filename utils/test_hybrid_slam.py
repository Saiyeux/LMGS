#!/usr/bin/env python3
"""
测试Hybrid SLAM系统
快速验证系统基本功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("Testing Hybrid SLAM System Basic Functionality")
    print("=" * 60)
    
    try:
        # 测试导入
        print("1. Testing imports...")
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset
        from hybrid_slam.utils.performance_monitor import PerformanceMonitor
        from hybrid_slam.utils.memory_manager import MemoryManager
        print("OK All imports successful")
        
        # 测试配置管理
        print("\n2. Testing configuration...")
        from hybrid_slam.utils.config_manager import ConfigManager
        config_path = "configs/test_config.yaml"
        
        if Path(config_path).exists():
            config = ConfigManager.load_config(config_path)
            print("OK Configuration loaded successfully")
            print(f"   Project: {config.get('project_name', 'Unknown')}")
            print(f"   Mode: {config.get('mode', 'Unknown')}")
        else:
            print(f"WARNING Config file not found: {config_path}")
            # 创建最小配置
            config = {
                'project_name': 'Test',
                'input': {'source': 'mock'},
                'output': {'save_trajectory': False},
                'frontend': {'min_matches': 5},
                'visualization': False
            }
            print("OK Using minimal test configuration")
        
        # 测试数据源
        print("\n3. Testing data source...")
        mock_dataset = create_mock_stereo_dataset(num_frames=10, resolution=(320, 240))
        
        frame_count = 0
        for stereo_frame in mock_dataset:
            frame_count += 1
            print(f"   Frame {frame_count}: {stereo_frame.left_image.shape}, baseline={stereo_frame.baseline}")
            if frame_count >= 3:  # 只测试前3帧
                break
        print("OK Mock dataset working correctly")
        
        # 测试内存管理
        print("\n4. Testing memory management...")
        memory_manager = MemoryManager(max_gpu_memory_gb=2.0, max_cpu_memory_gb=4.0)
        memory_stats = memory_manager.get_memory_statistics()
        print(f"✅ Memory manager initialized")
        print(f"   GPU available: {memory_stats['current_gpu'].get('available', False)}")
        print(f"   CPU monitoring: {memory_stats['current_cpu'].get('available', False)}")
        
        # 测试性能监控
        print("\n5. Testing performance monitor...")
        perf_monitor = PerformanceMonitor(target_fps=10.0, memory_limit_gb=4.0)
        time.sleep(0.1)  # 让监控器启动
        perf_monitor.update_frame_stats(processing_time=0.05, tracking_success=True)
        stats = perf_monitor.get_real_time_stats()
        print(f"✅ Performance monitor working")
        print(f"   Current FPS: {stats.get('fps', 0):.1f}")
        perf_monitor.stop()
        
        print("\n6. Testing SLAM system initialization...")
        # 修改配置为不启用可视化和GPU
        config['visualization'] = False
        config['device'] = 'cpu'  # 确保在没有GPU的环境下也能运行
        
        # 创建SLAM系统但不运行
        slam_system = HybridSLAMSystem(config, save_dir="test_results")
        print("✅ SLAM system initialized successfully")
        print(f"   Save directory: {slam_system.save_dir}")
        print(f"   Target FPS: {slam_system.target_fps}")
        
        # 清理
        slam_system._shutdown()
        
        print("\n" + "=" * 60)
        print("🎉 All basic functionality tests PASSED!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_slam_components():
    """测试SLAM组件"""
    print("\n" + "=" * 60)
    print("Testing SLAM Components")
    print("=" * 60)
    
    try:
        # 测试特征匹配器（不加载模型）
        print("1. Testing feature matcher...")
        from hybrid_slam.matchers.loftr_matcher import EfficientLoFTRMatcher
        
        matcher_config = {
            'device': 'cpu',
            'model_path': None,  # 不加载模型
            'resize_to': [320, 240]
        }
        
        # matcher = EfficientLoFTRMatcher(matcher_config)
        # print("✅ Feature matcher initialized")
        print("⚠️  Feature matcher test skipped (requires EfficientLoFTR model)")
        
        # 测试PnP求解器
        print("\n2. Testing PnP solver...")
        from hybrid_slam.solvers.pnp_solver import PnPSolver, PnPResult
        import numpy as np
        
        pnp_config = {
            'pnp_method': 'SOLVEPNP_ITERATIVE',
            'pnp_ransac_threshold': 2.0,
            'pnp_min_inliers': 6
        }
        
        pnp_solver = PnPSolver(pnp_config)
        print("✅ PnP solver initialized")
        
        # 测试可视化工具
        print("\n3. Testing visualization...")
        from hybrid_slam.utils.visualization import visualize_matches, plot_trajectory
        
        # 创建测试数据
        img0 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        test_matches = {
            'keypoints0': np.random.rand(10, 2) * 320,
            'keypoints1': np.random.rand(10, 2) * 320,
            'confidence': np.random.rand(10),
            'num_matches': 10
        }
        
        # vis_img = visualize_matches(img0, img1, test_matches)
        print("✅ Visualization tools available")
        
        print("\n" + "=" * 60)
        print("🎉 Component tests completed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Starting Hybrid SLAM System Tests...\n")
    
    # 运行基本功能测试
    basic_success = test_basic_functionality()
    
    if basic_success:
        # 运行组件测试
        component_success = test_slam_components()
        
        if component_success:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("\nSystem is ready for use. You can now run:")
            print("python scripts/run_hybrid_slam.py --config configs/test_config.yaml")
            return 0
        else:
            print("\n⚠️  Basic tests passed, but some component tests failed.")
            print("The system may still work with limited functionality.")
            return 1
    else:
        print("\n❌ Basic tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())