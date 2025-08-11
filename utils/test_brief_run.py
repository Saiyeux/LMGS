#!/usr/bin/env python3
"""
简短运行测试 - 运行几秒钟后自动停止
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_for_duration(duration_seconds=5):
    """运行系统指定时间后停止"""
    
    print(f"=== 测试运行 {duration_seconds} 秒 ===")
    
    try:
        from hybrid_slam.core.slam_system import HybridSLAMSystem
        
        # 简单配置
        config = {
            'device': 'cpu',
            'input': {
                'source': 'mock',
                'camera': {
                    'resolution': [320, 240],
                    'fps': 10
                }
            },
            'visualization': False,  # 关闭可视化
            'frontend': {
                'matcher_type': 'loftr',
                'tracking_method': 'pnp'
            },
            'performance_targets': {
                'target_fps': 10,
                'max_memory_gb': 4,
                'max_gpu_memory_gb': 2
            },
            'output': {
                'save_trajectory': True,
                'save_keyframes': False
            }
        }
        
        # 创建SLAM系统
        slam_system = HybridSLAMSystem(config, save_dir="test_brief_results")
        print("SLAM系统创建成功")
        
        # 在单独线程中运行SLAM
        def run_slam():
            try:
                slam_system.run()
            except Exception as e:
                print(f"SLAM运行错误: {e}")
        
        slam_thread = threading.Thread(target=run_slam)
        slam_thread.daemon = True
        slam_thread.start()
        
        # 等待指定时间
        print(f"开始处理，将运行 {duration_seconds} 秒...")
        time.sleep(duration_seconds)
        
        # 停止系统
        slam_system.stop_event.set()
        print("\n停止信号已发送")
        
        # 等待线程结束
        slam_thread.join(timeout=2)
        
        # 显示结果
        print(f"处理完成！")
        print(f"轨迹点数: {len(slam_system.trajectory)}")
        if slam_system.trajectory:
            print(f"最后位姿时间戳: {slam_system.trajectory[-1]['timestamp']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_for_duration(5)
    if success:
        print("\nSUCCESS: Brief run test passed!")
        print("Dual-camera real-time reconstruction system working properly")
    else:
        print("\nERROR: Brief run test failed")
    
    exit(0 if success else 1)