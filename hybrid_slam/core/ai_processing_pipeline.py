"""
AIProcessingPipeline - AI处理管道
集成EfficientLoFTR、MonoGS和位姿估计
"""

import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Tuple

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

from ..utils.data_structures import StereoFrame, ProcessingResult


class AIProcessingPipeline(QObject):
    """
    AI处理管道
    职责:
    - 集成EfficientLoFTR特征匹配
    - 集成MonoGS 3D重建
    - 位姿估计和轨迹追踪
    - 处理结果缓存
    """
    
    # Qt信号定义
    if QT_AVAILABLE:
        processing_complete = pyqtSignal(object)  # ProcessingResult对象
        progress_updated = pyqtSignal(dict)
        error_occurred = pyqtSignal(str)
    else:
        processing_complete = None
        progress_updated = None
        error_occurred = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # 配置参数
        self.config = config or {
            'enable_loftr': True,
            'enable_mono_gs': False,  # 暂时关闭，先实现基础功能
            'enable_pnp': True,
            'max_features': 1000,
            'confidence_threshold': 0.1,  # 进一步降低置信度阈值
            'processing_timeout': 1.0,  # 处理超时时间(秒)
            'match_threshold': 0.05,  # 进一步降低匹配阈值
            'use_gpu': True
        }
        
        # AI模型
        self.loftr_matcher = None
        self.mono_gs = None
        self.pnp_solver = None
        
        # 处理状态
        self.processing_queue = Queue(maxsize=10)
        self.result_cache = {}
        self.is_processing = False
        self.processing_thread = None
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'successful_matches': 0,
            'failed_matches': 0,
            'current_confidence': 0.0
        }
        
        # 轨迹追踪
        self.trajectory_points = []
        self.current_pose = np.eye(4)
        
    def initialize_models(self) -> bool:
        """初始化AI模型"""
        try:
            print("正在初始化AI模型...")
            
            # 初始化EfficientLoFTR
            if self.config['enable_loftr']:
                try:
                    from ..matchers.loftr_matcher import EfficientLoFTRMatcher
                    loftr_config = {
                        'device': 'cuda' if self.config.get('use_gpu', True) else 'cpu',
                        'resize_to': [640, 480],
                        'match_threshold': self.config.get('match_threshold', 0.05),
                        'max_keypoints': 2048,
                        'model_path': 'thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt',  # 使用与real_time_stereo_matcher.py相同的权重
                        'enable_stereo_constraints': False,  # 先禁用立体约束，增加匹配数量
                        'max_disparity': 200,
                        'stereo_y_tolerance': 10.0
                    }
                    self.loftr_matcher = EfficientLoFTRMatcher(loftr_config)
                    print("OK EfficientLoFTR初始化成功")
                except Exception as e:
                    print(f"WARN EfficientLoFTR初始化失败，使用OpenCV特征匹配: {e}")
                    self.config['enable_loftr'] = False
            
            # 初始化PnP求解器
            if self.config['enable_pnp']:
                try:
                    from ..solvers.pnp_solver import PnPSolver
                    pnp_config = {
                        'method': 'EPNP',
                        'ransac_threshold': 3.0,
                        'min_inliers': 20,
                        'max_iterations': 1000,
                        'confidence': 0.99
                    }
                    self.pnp_solver = PnPSolver(pnp_config)
                    print("OK PnP求解器初始化成功")
                except Exception as e:
                    print(f"WARN PnP求解器初始化失败: {e}")
                    self.config['enable_pnp'] = False
            
            # 初始化MonoGS (暂时跳过)
            if self.config['enable_mono_gs']:
                try:
                    # TODO: 集成MonoGS
                    print("⚠ MonoGS集成待实现")
                    self.config['enable_mono_gs'] = False
                except Exception as e:
                    print(f"⚠ MonoGS初始化失败: {e}")
                    self.config['enable_mono_gs'] = False
            
            print("AI模型初始化完成")
            return True
            
        except Exception as e:
            error_msg = f"AI模型初始化失败: {e}"
            print(error_msg)
            if self.error_occurred:
                self.error_occurred.emit(error_msg)
            return False
    
    def start_processing(self) -> bool:
        """启动处理线程"""
        if self.running:
            print("AI处理线程已经在运行中")
            return True
            
        if not self.initialize_models():
            return False
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("AI处理线程已启动")
        return True
    
    def stop_processing(self):
        """停止处理线程"""
        if not self.running:
            return
            
        self.running = False
        
        # 等待线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("AI处理线程已停止")
    
    @pyqtSlot(object)
    def process_stereo_frame(self, stereo_frame: StereoFrame):
        """处理立体帧（Qt槽函数）"""
        self.enqueue_frame(stereo_frame)
    
    def enqueue_frame(self, stereo_frame: StereoFrame):
        """将帧加入处理队列"""
        if not self.running:
            return
            
        try:
            # 如果队列满，丢弃最老的帧
            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except Empty:
                    pass
            
            self.processing_queue.put(stereo_frame, block=False)
            
        except Exception as e:
            error_msg = f"处理队列错误: {e}"
            print(error_msg)
            if self.error_occurred:
                self.error_occurred.emit(error_msg)
    
    def _processing_loop(self):
        """AI处理主循环"""
        print("AI处理循环开始")
        
        while self.running:
            try:
                # 从队列获取帧
                stereo_frame = self.processing_queue.get(timeout=1.0)
                
                if not self.is_processing:
                    self.is_processing = True
                    start_time = time.time()
                    
                    # 执行处理管道
                    result = self._execute_pipeline(stereo_frame)
                    
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time * 1000  # 转换为毫秒
                    
                    # 更新统计信息
                    self._update_stats(result)
                    
                    # 发送结果
                    if self.processing_complete:
                        self.processing_complete.emit(result)
                    
                    self.is_processing = False
                
            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                error_msg = f"处理循环错误: {e}"
                print(error_msg)
                if self.error_occurred:
                    self.error_occurred.emit(error_msg)
                self.is_processing = False
        
        print("AI处理循环结束")
    
    def _execute_pipeline(self, stereo_frame: StereoFrame) -> ProcessingResult:
        """执行完整的AI处理管道"""
        result = ProcessingResult(
            frame_id=stereo_frame.frame_id,
            timestamp=stereo_frame.timestamp
        )
        
        try:
            # 步骤1: 特征匹配
            matches, confidence = self._perform_feature_matching(
                stereo_frame.left_image, 
                stereo_frame.right_image
            )
            
            result.matches = matches
            result.confidence = confidence
            result.num_matches = len(matches) if matches else 0
            
            print(f"[处理管道] 帧ID: {stereo_frame.frame_id}, 匹配数: {result.num_matches}, 置信度: {result.confidence:.3f}")
            
            # 步骤2: 位姿估计
            if self.pnp_solver and matches and len(matches) > 4:
                try:
                    print(f"[位姿估计] 尝试位姿估计，匹配数: {len(matches)}")
                    pose = self._estimate_pose(matches, stereo_frame)
                    result.pose = pose
                    
                    # 更新轨迹
                    if pose is not None:
                        translation = pose[:3, 3]
                        self.trajectory_points.append(translation)
                        self.current_pose = pose
                        print(f"[位姿估计] 成功，位置: {translation}")
                    else:
                        print(f"[位姿估计] 失败")
                        
                except Exception as e:
                    result.error = f"位姿估计失败: {e}"
                    print(f"[位姿估计] 错误: {e}")
            else:
                print(f"[位姿估计] 跳过，PnP求解器: {self.pnp_solver is not None}, 匹配数: {len(matches) if matches else 0}")
            
            # 步骤3: 3D重建 (MonoGS - 暂时跳过)
            if self.config['enable_mono_gs'] and self.mono_gs:
                # TODO: 集成MonoGS
                pass
            
            # 步骤4: 生成可视化数据
            # 设置方法名称
            if self.loftr_matcher and self.loftr_matcher.model:
                result.method = 'EfficientLoFTR'
            else:
                result.method = 'OpenCV'
            
            result.visualization_data = self._create_visualization(
                stereo_frame, result
            )
            print(f"[可视化] 可视化数据生成完成")
            
        except Exception as e:
            result.error = f"处理管道错误: {e}"
            print(f"[处理管道] 错误: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _perform_feature_matching(self, left_image: np.ndarray, 
                                right_image: np.ndarray) -> Tuple[Optional[List], float]:
        """执行特征匹配"""
        try:
            if self.loftr_matcher:
                # 使用EfficientLoFTR
                matches, confidence = self.loftr_matcher.match_pair(
                    left_image, right_image
                )
                return matches, confidence
            else:
                # 使用OpenCV作为后备
                return self._opencv_feature_matching(left_image, right_image)
                
        except Exception as e:
            print(f"特征匹配失败: {e}")
            return None, 0.0
    
    def _opencv_feature_matching(self, left_image: np.ndarray, 
                               right_image: np.ndarray) -> Tuple[Optional[List], float]:
        """OpenCV特征匹配作为后备"""
        try:
            import cv2
            
            # 转换为灰度图
            gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            
            # 使用SIFT特征检测
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray_left, None)
            kp2, des2 = sift.detectAndCompute(gray_right, None)
            
            if des1 is None or des2 is None:
                return None, 0.0
            
            # 使用FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # 应用Lowe's ratio test（放宽阈值）
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.9 * n.distance:  # 进一步放宽ratio test阈值
                        good_matches.append(m)
            
            # 转换为坐标对
            if len(good_matches) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # 使用RANSAC过滤（放宽阈值）
                _, mask = cv2.findHomography(src_pts, dst_pts, 
                                           cv2.RANSAC, 12.0)  # 进一步增加RANSAC阈值
                
                if mask is not None:
                    good_matches = [m for i, m in enumerate(good_matches) if mask[i]]
                    src_pts = src_pts[mask.ravel() == 1]
                    dst_pts = dst_pts[mask.ravel() == 1]
                
                matches_list = list(zip(src_pts, dst_pts))
                confidence = min(1.0, len(matches_list) / 100.0)  # 简单置信度计算
                
                return matches_list, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"OpenCV特征匹配失败: {e}")
            return None, 0.0
    
    def _estimate_pose(self, matches: List, stereo_frame: StereoFrame) -> Optional[np.ndarray]:
        """位姿估计"""
        try:
            if self.pnp_solver:
                return self.pnp_solver.estimate_pose(matches)
            return None
        except Exception as e:
            print(f"位姿估计失败: {e}")
            return None
    
    def _create_visualization(self, stereo_frame: StereoFrame, 
                            result: ProcessingResult) -> np.ndarray:
        """创建可视化数据"""
        try:
            import cv2
            
            # 创建可视化画布
            left_img = stereo_frame.left_image.copy()
            right_img = stereo_frame.right_image.copy()
            
            # 如果有匹配，绘制匹配点
            if result.matches and len(result.matches) > 0:
                print(f"[可视化] 绘制 {len(result.matches)} 个匹配点")
                # 在左右图像上绘制匹配点
                for i, (pt1, pt2) in enumerate(result.matches[:50]):  # 限制显示前50个
                    if len(pt1) >= 2 and len(pt2) >= 2:
                        # 检查是否为有效数值（非NaN）
                        if (not np.isnan(pt1[0]) and not np.isnan(pt1[1]) and 
                            not np.isnan(pt2[0]) and not np.isnan(pt2[1])):
                            pt1_int = (int(pt1[0]), int(pt1[1]))
                            pt2_int = (int(pt2[0]), int(pt2[1]))
                            
                            # 绘制匹配点（增加大小和颜色对比度）
                            cv2.circle(left_img, pt1_int, 5, (0, 255, 0), -1)  # 增大半径
                            cv2.circle(right_img, pt2_int, 5, (0, 255, 0), -1)
                            # 添加白色边框增加可见度
                            cv2.circle(left_img, pt1_int, 6, (255, 255, 255), 1)
                            cv2.circle(right_img, pt2_int, 6, (255, 255, 255), 1)
                            
                            print(f"[可视化] 匹配{i+1}: ({pt1[0]:.1f},{pt1[1]:.1f}) -> ({pt2[0]:.1f},{pt2[1]:.1f})")
            else:
                print(f"[可视化] 无匹配点可显示")
            
            # 拼接左右图像
            combined = cv2.hconcat([left_img, right_img])
            
            # 在拼接图上绘制连线（如果有匹配）
            if result.matches and len(result.matches) > 0:
                for i, (pt1, pt2) in enumerate(result.matches[:20]):  # 限制连线数量
                    if (len(pt1) >= 2 and len(pt2) >= 2 and 
                        not np.isnan(pt1[0]) and not np.isnan(pt1[1]) and 
                        not np.isnan(pt2[0]) and not np.isnan(pt2[1])):
                        
                        pt1_int = (int(pt1[0]), int(pt1[1]))
                        pt2_int = (int(pt2[0] + left_img.shape[1]), int(pt2[1]))  # 右图偏移
                        
                        # 绘制连线（使用亮黄色）
                        cv2.line(combined, pt1_int, pt2_int, (0, 255, 255), 2)
            
            # 添加信息文本（增加背景和更大字体）
            info_text = [
                f"Frame: {result.frame_id}",
                f"Matches: {result.num_matches}",
                f"Confidence: {result.confidence:.3f}" if not np.isnan(result.confidence) else "Confidence: N/A",
                f"Method: {getattr(result, 'method', 'Unknown')}",
                f"Time: {result.processing_time:.1f}ms" if not np.isnan(result.processing_time) else "Time: N/A"
            ]
            
            for i, text in enumerate(info_text):
                # 添加黑色背景
                cv2.rectangle(combined, (5, 10 + i * 30), (400, 35 + i * 30), (0, 0, 0), -1)
                # 添加文本
                cv2.putText(combined, text, (10, 30 + i * 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            return combined
            
        except Exception as e:
            print(f"可视化创建失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回原始图像拼接
            try:
                return cv2.hconcat([stereo_frame.left_image, stereo_frame.right_image])
            except:
                # 如果连拼接都失败，返回空图像
                return np.zeros((480, 1280, 3), dtype=np.uint8)
    
    def _update_stats(self, result: ProcessingResult):
        """更新统计信息"""
        self.stats['total_processed'] += 1
        
        # 更新平均处理时间
        total_time = (self.stats['avg_processing_time'] * 
                     (self.stats['total_processed'] - 1) + result.processing_time)
        self.stats['avg_processing_time'] = total_time / self.stats['total_processed']
        
        # 更新匹配统计
        if result.num_matches > 0:
            self.stats['successful_matches'] += 1
        else:
            self.stats['failed_matches'] += 1
        
        # 更新当前置信度
        self.stats['current_confidence'] = result.confidence
        
        # 发送进度更新
        if self.progress_updated:
            self.progress_updated.emit(self.stats.copy())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def get_trajectory(self) -> List[np.ndarray]:
        """获取轨迹点"""
        return self.trajectory_points.copy()
    
    def is_running_processing(self) -> bool:
        """检查处理线程是否在运行"""
        return self.running
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop_processing()


# 提供非Qt版本的处理管道
class SimpleAIProcessingPipeline:
    """简化版本，不依赖Qt"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.pipeline = AIProcessingPipeline(config)
        
    def initialize_models(self):
        return self.pipeline.initialize_models()
    
    def start_processing(self):
        return self.pipeline.start_processing()
    
    def stop_processing(self):
        self.pipeline.stop_processing()
    
    def enqueue_frame(self, stereo_frame):
        self.pipeline.enqueue_frame(stereo_frame)
    
    def get_stats(self):
        return self.pipeline.get_stats()
    
    def get_trajectory(self):
        return self.pipeline.get_trajectory()
    
    def is_running_processing(self):
        return self.pipeline.is_running_processing()