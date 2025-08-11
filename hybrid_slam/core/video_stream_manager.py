"""
VideoStreamManager - 视频流管理器
负责双摄像头数据采集和同步
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Tuple, Optional, Dict, Any

try:
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
    QT_AVAILABLE = True
except ImportError:
    # 如果PyQt5不可用，提供基础类
    QT_AVAILABLE = False
    class QObject:
        pass
    def pyqtSignal(*args):
        pass
    def pyqtSlot(*args):
        def decorator(func):
            return func
        return decorator

from ..utils.data_structures import StereoFrame


class VideoStreamManager(QObject):
    """
    视频流管理器
    职责:
    - 双摄像头数据采集
    - 帧同步和时间戳管理
    - 数据预处理和格式转换
    - 帧缓存管理
    """
    
    # Qt信号定义
    if QT_AVAILABLE:
        frame_ready = pyqtSignal(object)  # StereoFrame对象
        error_occurred = pyqtSignal(str)
        stats_updated = pyqtSignal(dict)
    else:
        frame_ready = None
        error_occurred = None
        stats_updated = None
    
    def __init__(self, left_device: int = 0, right_device: int = 1, 
                 target_fps: int = 30, buffer_size: int = 30):
        super().__init__()
        
        # 摄像头配置
        self.device_ids = (left_device, right_device)
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        
        # 摄像头对象
        self.left_cap = None
        self.right_cap = None
        
        # 缓存管理
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.sync_tolerance = 0.033  # 33ms同步容差
        
        # 控制变量
        self.running = False
        self.capture_thread = None
        self.frame_counter = 0
        
        # 统计信息
        self.stats = {
            'fps': 0.0,
            'dropped_frames': 0,
            'sync_errors': 0,
            'total_frames': 0
        }
        
        # FPS计算
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def initialize_cameras(self) -> bool:
        """初始化双摄像头"""
        try:
            print(f"正在初始化摄像头 {self.device_ids[0]} 和 {self.device_ids[1]}...")
            
            # 使用更安全的摄像头初始化方式
            self.left_cap = cv2.VideoCapture(self.device_ids[0], cv2.CAP_DSHOW)  # Windows DirectShow
            time.sleep(0.1)  # 短暂等待初始化
            self.right_cap = cv2.VideoCapture(self.device_ids[1], cv2.CAP_DSHOW)
            time.sleep(0.1)
            
            # 检查摄像头是否成功打开
            left_opened = self.left_cap.isOpened()
            right_opened = self.right_cap.isOpened()
            
            print(f"左摄像头 {self.device_ids[0]}: {'成功' if left_opened else '失败'}")
            print(f"右摄像头 {self.device_ids[1]}: {'成功' if right_opened else '失败'}")
            
            if not left_opened:
                if self.left_cap:
                    self.left_cap.release()
                    self.left_cap = None
                raise RuntimeError(f"无法打开左摄像头 {self.device_ids[0]}")
                
            if not right_opened:
                if self.right_cap:
                    self.right_cap.release()
                    self.right_cap = None
                raise RuntimeError(f"无法打开右摄像头 {self.device_ids[1]}")
            
            # 设置摄像头参数
            for i, cap in enumerate([self.left_cap, self.right_cap]):
                # 设置较小的分辨率避免性能问题
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区
                
                # 验证设置
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"摄像头 {self.device_ids[i]} - 分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
                
                # 测试读取一帧来验证摄像头真正可用
                print(f"测试读取摄像头 {self.device_ids[i]}...")
                test_ret, test_frame = cap.read()
                if test_ret and test_frame is not None:
                    print(f"摄像头 {self.device_ids[i]} 测试读取成功: {test_frame.shape}")
                else:
                    raise RuntimeError(f"摄像头 {self.device_ids[i]} 无法读取帧")
            
            print("摄像头初始化完成")
            return True
            
        except Exception as e:
            error_msg = f"摄像头初始化失败: {e}"
            print(error_msg)
            if self.error_occurred:
                self.error_occurred.emit(error_msg)
            return False
    
    def start_capture(self) -> bool:
        """开始视频采集"""
        if self.running:
            print("视频采集已经在运行中")
            return True
            
        if not self.initialize_cameras():
            return False
            
        self.running = True
        self.frame_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # 重置统计信息
        self.stats = {
            'fps': 0.0,
            'dropped_frames': 0,
            'sync_errors': 0,
            'total_frames': 0
        }
        
        # 启动采集线程
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print("视频采集已启动")
        return True
    
    def stop_capture(self):
        """停止视频采集"""
        if not self.running:
            return
            
        print("正在停止视频采集...")
        self.running = False
        
        # 等待线程结束，使用更长的超时时间
        if self.capture_thread and self.capture_thread.is_alive():
            print("等待采集线程结束...")
            self.capture_thread.join(timeout=5.0)
            
            # 如果线程仍然活跃，强制释放资源
            if self.capture_thread.is_alive():
                print("警告: 采集线程未正常结束，强制释放资源")
        
        # 安全释放摄像头资源
        try:
            if self.left_cap:
                print("释放左摄像头...")
                self.left_cap.release()
                self.left_cap = None
        except Exception as e:
            print(f"释放左摄像头时出错: {e}")
        
        try:
            if self.right_cap:
                print("释放右摄像头...")
                self.right_cap.release()
                self.right_cap = None
        except Exception as e:
            print(f"释放右摄像头时出错: {e}")
        
        print("视频采集已停止")
    
    def _capture_loop(self):
        """视频采集主循环"""
        frame_time = 1.0 / self.target_fps
        consecutive_failures = 0
        max_failures = 10  # 连续失败超过10次就退出
        
        print("开始视频采集循环")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 检查摄像头是否仍然有效
                if not (self.left_cap and self.left_cap.isOpened() and 
                       self.right_cap and self.right_cap.isOpened()):
                    error_msg = "摄像头连接丢失"
                    print(error_msg)
                    if self.error_occurred:
                        self.error_occurred.emit(error_msg)
                    break
                
                # 同步读取双摄像头 - 添加超时保护
                timestamp = time.time()
                
                # 使用非阻塞方式读取
                ret_left = False
                ret_right = False
                frame_left = None
                frame_right = None
                
                try:
                    # 快速读取，如果卡住会在后面处理
                    ret_left, frame_left = self.left_cap.read()
                    if ret_left:
                        ret_right, frame_right = self.right_cap.read()
                except Exception as read_error:
                    print(f"摄像头读取异常: {read_error}")
                    ret_left = ret_right = False
                
                if ret_left and ret_right and frame_left is not None and frame_right is not None:
                    # 验证帧数据有效性
                    if frame_left.size > 0 and frame_right.size > 0:
                        # 创建立体帧对象
                        stereo_frame = StereoFrame(
                            frame_id=self.frame_counter,
                            timestamp=timestamp,
                            left_image=frame_left.copy(),
                            right_image=frame_right.copy(),
                            metadata={
                                'capture_time': timestamp,
                                'left_shape': frame_left.shape,
                                'right_shape': frame_right.shape
                            }
                        )
                        
                        # 发送到缓存和信号
                        self._enqueue_frame(stereo_frame)
                        self.frame_counter += 1
                        self.stats['total_frames'] += 1
                        
                        # 重置失败计数
                        consecutive_failures = 0
                        
                        # 更新FPS
                        self._update_fps()
                    else:
                        print("警告: 读取到空帧")
                        consecutive_failures += 1
                else:
                    # 读取失败
                    self.stats['dropped_frames'] += 1
                    consecutive_failures += 1
                    
                    if ret_left != ret_right:
                        self.stats['sync_errors'] += 1
                        print(f"摄像头同步错误: left={ret_left}, right={ret_right}")
                    
                    # 如果连续失败太多次，尝试重新初始化
                    if consecutive_failures >= max_failures:
                        error_msg = f"连续{consecutive_failures}次读取失败，停止采集"
                        print(error_msg)
                        if self.error_occurred:
                            self.error_occurred.emit(error_msg)
                        break
                    
                    # 短暂等待后继续
                    time.sleep(0.01)
                
                # FPS控制 - 避免过快循环
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, frame_time - elapsed)  # 至少等待1ms
                time.sleep(sleep_time)
                
            except Exception as e:
                consecutive_failures += 1
                error_msg = f"视频采集循环错误: {e}"
                print(error_msg)
                
                if consecutive_failures >= max_failures:
                    if self.error_occurred:
                        self.error_occurred.emit(error_msg)
                    break
                    
                # 短暂等待后重试
                time.sleep(0.1)
        
        print("视频采集循环结束")
    
    def _enqueue_frame(self, stereo_frame: StereoFrame):
        """将帧加入缓存队列"""
        try:
            # 如果队列满，丢弃最老的帧
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                    self.stats['dropped_frames'] += 1
                except Empty:
                    pass
            
            self.frame_buffer.put(stereo_frame, block=False)
            
            # 发送Qt信号
            if self.frame_ready:
                self.frame_ready.emit(stereo_frame)
                
        except Exception as e:
            error_msg = f"帧缓存错误: {e}"
            print(error_msg)
            if self.error_occurred:
                self.error_occurred.emit(error_msg)
    
    def _update_fps(self):
        """更新FPS统计"""
        self.fps_counter += 1
        
        if self.fps_counter >= 30:  # 每30帧更新一次FPS
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            
            if elapsed > 0:
                self.stats['fps'] = self.fps_counter / elapsed
                
                # 发送统计信息
                if self.stats_updated:
                    self.stats_updated.emit(self.stats.copy())
            
            # 重置计数器
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_latest_frame(self) -> Optional[StereoFrame]:
        """获取最新的帧（非阻塞）"""
        try:
            return self.frame_buffer.get_nowait()
        except Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop_capture()


# 提供非Qt版本的管理器
class SimpleVideoStreamManager:
    """简化版本，不依赖Qt"""
    
    def __init__(self, left_device: int = 0, right_device: int = 1):
        self.manager = VideoStreamManager(left_device, right_device)
        
    def start_capture(self):
        return self.manager.start_capture()
    
    def stop_capture(self):
        self.manager.stop_capture()
    
    def get_latest_frame(self):
        return self.manager.get_latest_frame()
    
    def get_stats(self):
        return self.manager.get_stats()
    
    def is_running(self):
        return self.manager.is_running()