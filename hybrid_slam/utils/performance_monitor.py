"""
性能监控器
实时监控系统性能指标
"""

import psutil
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, target_fps: float = 30.0, memory_limit_gb: float = 8.0):
        """
        初始化性能监控器
        
        Args:
            target_fps: 目标帧率
            memory_limit_gb: 内存限制(GB)
        """
        self.target_fps = target_fps
        self.memory_limit_gb = memory_limit_gb
        
        # 性能统计
        self.timing_stats = defaultdict(list)
        self.memory_stats = deque(maxlen=100)
        self.gpu_memory_stats = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        self.tracking_success_history = deque(maxlen=100)
        
        # 实时统计
        self.current_fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # 线程安全
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        
        # 启动监控
        self.start()
    
    def start(self):
        """启动性能监控"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
    
    def stop(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self._update_system_stats()
                time.sleep(1.0)  # 每秒更新一次系统统计
            except Exception as e:
                print(f"Performance monitor error: {e}")
                time.sleep(1.0)
    
    def _update_system_stats(self):
        """更新系统统计信息"""
        with self._lock:
            # CPU和内存统计
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            self.memory_stats.append(memory_mb)
            
            # GPU内存统计
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.gpu_memory_stats.append(gpu_memory_mb)
    
    def update_frame_stats(self, processing_time: float, tracking_success: bool):
        """更新帧处理统计"""
        current_time = time.time()
        
        with self._lock:
            # 计算帧间隔
            frame_interval = current_time - self.last_frame_time
            self.frame_times.append(frame_interval)
            self.last_frame_time = current_time
            
            # 更新FPS
            if len(self.frame_times) > 1:
                avg_interval = np.mean(list(self.frame_times))
                self.current_fps = 1.0 / max(avg_interval, 1e-6)
            
            # 记录跟踪成功率
            self.tracking_success_history.append(tracking_success)
            
            # 记录处理时间
            self.timing_stats['frame_processing'].append(processing_time)
    
    def log_tracking_time(self, duration: float, method: str):
        """记录跟踪时间"""
        with self._lock:
            self.timing_stats[f'tracking_{method}'].append(duration)
    
    def log_matching_time(self, duration: float):
        """记录特征匹配时间"""
        with self._lock:
            self.timing_stats['feature_matching'].append(duration)
    
    def log_pnp_time(self, duration: float):
        """记录PnP求解时间"""
        with self._lock:
            self.timing_stats['pnp_solving'].append(duration)
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时统计信息"""
        with self._lock:
            stats = {
                'fps': self.current_fps,
                'target_fps': self.target_fps,
                'fps_ratio': self.current_fps / self.target_fps if self.target_fps > 0 else 0.0,
            }
            
            # 内存统计
            if self.memory_stats:
                current_memory_mb = self.memory_stats[-1]
                stats.update({
                    'memory_usage_mb': current_memory_mb,
                    'memory_usage_gb': current_memory_mb / 1024,
                    'memory_usage_ratio': (current_memory_mb / 1024) / self.memory_limit_gb
                })
            
            # GPU内存统计
            if self.gpu_memory_stats:
                current_gpu_memory_mb = self.gpu_memory_stats[-1]
                stats.update({
                    'gpu_memory_usage_mb': current_gpu_memory_mb,
                    'gpu_memory_usage_gb': current_gpu_memory_mb / 1024
                })
            
            # 跟踪成功率
            if self.tracking_success_history:
                success_rate = np.mean(list(self.tracking_success_history))
                stats['tracking_success_rate'] = success_rate
            
            return stats
    
    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """获取时间统计摘要"""
        with self._lock:
            summary = {}
            
            for key, times in self.timing_stats.items():
                if len(times) > 0:
                    times_array = np.array(times)
                    summary[key] = {
                        'mean': float(np.mean(times_array)),
                        'std': float(np.std(times_array)),
                        'min': float(np.min(times_array)),
                        'max': float(np.max(times_array)),
                        'median': float(np.median(times_array)),
                        'count': len(times)
                    }
            
            return summary
    
    def check_performance_warnings(self) -> List[str]:
        """检查性能警告"""
        warnings = []
        stats = self.get_real_time_stats()
        
        # FPS警告
        if 'fps_ratio' in stats and stats['fps_ratio'] < 0.8:
            warnings.append(f"Low FPS: {stats['fps']:.1f} (target: {self.target_fps})")
        
        # 内存警告
        if 'memory_usage_ratio' in stats and stats['memory_usage_ratio'] > 0.9:
            warnings.append(f"High memory usage: {stats['memory_usage_gb']:.1f}GB")
        
        # 跟踪成功率警告
        if 'tracking_success_rate' in stats and stats['tracking_success_rate'] < 0.7:
            warnings.append(f"Low tracking success rate: {stats['tracking_success_rate']:.1%}")
        
        return warnings
    
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        with self._lock:
            report = {
                'summary': self.get_real_time_stats(),
                'timing_analysis': self.get_timing_summary(),
                'warnings': self.check_performance_warnings(),
                'system_info': self._get_system_info()
            }
            
            # 历史统计
            if self.memory_stats:
                memory_history = list(self.memory_stats)
                report['memory_history'] = {
                    'values_mb': memory_history,
                    'mean_mb': float(np.mean(memory_history)),
                    'max_mb': float(np.max(memory_history)),
                    'min_mb': float(np.min(memory_history))
                }
            
            if self.gpu_memory_stats:
                gpu_memory_history = list(self.gpu_memory_stats)
                report['gpu_memory_history'] = {
                    'values_mb': gpu_memory_history,
                    'mean_mb': float(np.mean(gpu_memory_history)),
                    'max_mb': float(np.max(gpu_memory_history)),
                    'min_mb': float(np.min(gpu_memory_history))
                }
            
            if self.fps_history:
                fps_history = list(self.fps_history)
                report['fps_history'] = {
                    'values': fps_history,
                    'mean': float(np.mean(fps_history)),
                    'std': float(np.std(fps_history)),
                    'min': float(np.min(fps_history)),
                    'max': float(np.max(fps_history))
                }
            
            return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        # GPU信息
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info.update({
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2),
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_cached_mb': torch.cuda.memory_reserved() / (1024**2)
            })
        else:
            info['gpu_available'] = False
        
        return info
    
    def print_real_time_stats(self):
        """打印实时统计信息"""
        stats = self.get_real_time_stats()
        warnings = self.check_performance_warnings()
        
        print(f"\n=== Performance Stats ===")
        print(f"FPS: {stats.get('fps', 0):.1f} / {self.target_fps}")
        print(f"Memory: {stats.get('memory_usage_gb', 0):.1f}GB")
        
        if 'gpu_memory_usage_gb' in stats:
            print(f"GPU Memory: {stats['gpu_memory_usage_gb']:.1f}GB")
        
        if 'tracking_success_rate' in stats:
            print(f"Tracking Success: {stats['tracking_success_rate']:.1%}")
        
        if warnings:
            print(f"Warnings: {', '.join(warnings)}")
        
        print("=" * 25)
    
    def __del__(self):
        """析构函数"""
        self.stop()