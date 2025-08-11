"""
内存管理器
管理GPU和CPU内存使用
"""

import gc
import time
import threading
from typing import Optional, Dict, Any
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, GPU memory management disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, CPU memory monitoring limited")

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_gpu_memory_gb: float = 6.0, max_cpu_memory_gb: float = 8.0):
        """
        初始化内存管理器
        
        Args:
            max_gpu_memory_gb: GPU内存限制(GB)
            max_cpu_memory_gb: CPU内存限制(GB)
        """
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.max_cpu_memory_gb = max_cpu_memory_gb
        
        # 内存使用历史
        self.gpu_memory_history = deque(maxlen=100)
        self.cpu_memory_history = deque(maxlen=100)
        
        # 缓存管理
        self.tensor_cache = {}
        self.cache_size_limit = 1000  # 最大缓存张量数
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 检查可用性
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.cpu_monitoring = PSUTIL_AVAILABLE
        
        print(f"Memory Manager initialized:")
        print(f"  GPU Memory Limit: {max_gpu_memory_gb:.1f}GB (Available: {self.gpu_available})")
        print(f"  CPU Memory Limit: {max_cpu_memory_gb:.1f}GB (Monitoring: {self.cpu_monitoring})")
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            free = total - reserved
            
            return {
                'available': True,
                'total_gb': total,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free,
                'utilization': allocated / total,
                'reserved_utilization': reserved / total
            }
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存信息"""
        if not self.cpu_monitoring:
            return {'available': False}
        
        try:
            mem = psutil.virtual_memory()
            return {
                'available': True,
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'free_gb': mem.available / (1024**3),
                'utilization': mem.percent / 100.0
            }
        except Exception as e:
            print(f"Error getting CPU memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def check_memory_pressure(self) -> Dict[str, bool]:
        """检查内存压力"""
        pressure = {
            'gpu_pressure': False,
            'cpu_pressure': False,
            'critical': False
        }
        
        # 检查GPU内存压力
        gpu_info = self.get_gpu_memory_info()
        if gpu_info.get('available', False):
            gpu_utilization = gpu_info.get('reserved_utilization', 0)
            if gpu_utilization > 0.85:
                pressure['gpu_pressure'] = True
            if gpu_utilization > 0.95:
                pressure['critical'] = True
        
        # 检查CPU内存压力
        cpu_info = self.get_cpu_memory_info()
        if cpu_info.get('available', False):
            cpu_utilization = cpu_info.get('utilization', 0)
            if cpu_utilization > 0.85:
                pressure['cpu_pressure'] = True
            if cpu_utilization > 0.95:
                pressure['critical'] = True
        
        return pressure
    
    def cleanup_if_needed(self, force: bool = False):
        """根据需要清理内存"""
        with self._lock:
            pressure = self.check_memory_pressure()
            
            if force or pressure['gpu_pressure'] or pressure['cpu_pressure']:
                self._perform_cleanup(force or pressure['critical'])
    
    def _perform_cleanup(self, aggressive: bool = False):
        """执行内存清理"""
        cleanup_actions = []
        
        # GPU内存清理
        if self.gpu_available:
            try:
                # 清空PyTorch缓存
                torch.cuda.empty_cache()
                cleanup_actions.append("GPU cache cleared")
                
                if aggressive:
                    # 强制垃圾回收
                    torch.cuda.synchronize()
                    cleanup_actions.append("GPU synchronized")
                    
            except Exception as e:
                cleanup_actions.append(f"GPU cleanup error: {e}")
        
        # CPU内存清理
        try:
            # 清理张量缓存
            if aggressive or len(self.tensor_cache) > self.cache_size_limit:
                cleared_count = len(self.tensor_cache)
                self.tensor_cache.clear()
                cleanup_actions.append(f"Cleared {cleared_count} cached tensors")
            
            # 强制垃圾回收
            if aggressive:
                collected = gc.collect()
                cleanup_actions.append(f"Collected {collected} objects")
                
        except Exception as e:
            cleanup_actions.append(f"CPU cleanup error: {e}")
        
        if cleanup_actions:
            print(f"Memory cleanup: {', '.join(cleanup_actions)}")
    
    def cache_tensor(self, key: str, tensor):
        """缓存张量"""
        if not TORCH_AVAILABLE:
            return
        
        with self._lock:
            # 检查缓存大小限制
            if len(self.tensor_cache) >= self.cache_size_limit:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.tensor_cache))
                del self.tensor_cache[oldest_key]
            
            # 保存张量的副本（如果是PyTorch张量）
            if hasattr(tensor, 'detach') and hasattr(tensor, 'clone'):
                self.tensor_cache[key] = tensor.detach().clone()
            else:
                self.tensor_cache[key] = tensor
    
    def get_cached_tensor(self, key: str):
        """获取缓存的张量"""
        with self._lock:
            return self.tensor_cache.get(key)
    
    def clear_cache(self):
        """清空所有缓存"""
        with self._lock:
            self.tensor_cache.clear()
            
            if self.gpu_available:
                torch.cuda.empty_cache()
    
    def monitor_memory_usage(self):
        """监控内存使用情况"""
        # 记录GPU内存使用
        gpu_info = self.get_gpu_memory_info()
        if gpu_info.get('available', False):
            self.gpu_memory_history.append(gpu_info['allocated_gb'])
        
        # 记录CPU内存使用
        cpu_info = self.get_cpu_memory_info()
        if cpu_info.get('available', False):
            self.cpu_memory_history.append(cpu_info['used_gb'])
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        stats = {
            'current_gpu': self.get_gpu_memory_info(),
            'current_cpu': self.get_cpu_memory_info(),
            'pressure': self.check_memory_pressure(),
            'cache_size': len(self.tensor_cache)
        }
        
        # 历史统计
        if self.gpu_memory_history:
            import numpy as np
            gpu_history = list(self.gpu_memory_history)
            stats['gpu_history'] = {
                'mean_gb': float(np.mean(gpu_history)),
                'max_gb': float(np.max(gpu_history)),
                'min_gb': float(np.min(gpu_history)),
                'std_gb': float(np.std(gpu_history))
            }
        
        if self.cpu_memory_history:
            import numpy as np
            cpu_history = list(self.cpu_memory_history)
            stats['cpu_history'] = {
                'mean_gb': float(np.mean(cpu_history)),
                'max_gb': float(np.max(cpu_history)),
                'min_gb': float(np.min(cpu_history)),
                'std_gb': float(np.std(cpu_history))
            }
        
        return stats
    
    def optimize_for_inference(self):
        """为推理优化内存设置"""
        if self.gpu_available:
            try:
                # 设置内存分配策略
                torch.cuda.empty_cache()
                
                # 禁用梯度计算以节省内存
                torch.set_grad_enabled(False)
                
                print("Optimized memory settings for inference")
            except Exception as e:
                print(f"Error optimizing memory for inference: {e}")
    
    def print_memory_status(self):
        """打印内存状态"""
        print("\n=== Memory Status ===")
        
        gpu_info = self.get_gpu_memory_info()
        if gpu_info.get('available', False):
            print(f"GPU Memory: {gpu_info['allocated_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB "
                  f"({gpu_info['utilization']*100:.1f}%)")
        else:
            print("GPU Memory: Not available")
        
        cpu_info = self.get_cpu_memory_info()
        if cpu_info.get('available', False):
            print(f"CPU Memory: {cpu_info['used_gb']:.1f}GB / {cpu_info['total_gb']:.1f}GB "
                  f"({cpu_info['utilization']*100:.1f}%)")
        else:
            print("CPU Memory: Monitoring not available")
        
        pressure = self.check_memory_pressure()
        if pressure['critical']:
            print("⚠️  CRITICAL MEMORY PRESSURE")
        elif pressure['gpu_pressure'] or pressure['cpu_pressure']:
            print("⚠️  Memory pressure detected")
        else:
            print("✅ Memory usage normal")
        
        print(f"Tensor Cache: {len(self.tensor_cache)} items")
        print("=" * 21)
    
    def __del__(self):
        """析构函数"""
        try:
            self.clear_cache()
        except:
            pass