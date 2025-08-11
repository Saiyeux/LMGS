"""
QtDisplayWidget - Qt显示组件
统一的视频流和AI处理结果显示界面
"""

import time
import numpy as np
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                                QGroupBox, QTextEdit, QFrame)
    from PyQt5.QtCore import Qt, pyqtSlot, QTimer
    from PyQt5.QtGui import QPixmap, QImage, QFont
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    print("PyQt5不可用，请安装: pip install PyQt5")

from ..utils.data_structures import StereoFrame, ProcessingResult


if QT_AVAILABLE:
    class QtDisplayWidget(QWidget):
        """
        Qt显示组件
        职责:
        - 原始视频流显示
        - AI处理结果可视化
        - 多窗口布局管理
        - 实时性能监控
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            # 显示组件
            self.left_video_label = QLabel()
            self.right_video_label = QLabel()
            self.result_display_label = QLabel()
            self.info_panel = QTextEdit()
            
            # 状态变量
            self.current_stereo_frame = None
            self.current_result = None
            
            # FPS计算
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.current_fps = 0.0
            
            # 统计信息
            self.total_frames = 0
            self.total_results = 0
            
            # 初始化界面
            self.setup_layout()
            self.setup_style()
            
            # 定时器用于更新显示
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self._update_displays)
            self.update_timer.start(33)  # 30 FPS 更新
        
        def setup_layout(self):
            """设置界面布局"""
            # 主布局
            main_layout = QVBoxLayout()
            main_layout.setSpacing(10)
            main_layout.setContentsMargins(10, 10, 10, 10)
            
            # 标题
            title_label = QLabel("Hybrid SLAM - Qt+OpenCV+AI视频处理系统")
            title_label.setAlignment(Qt.AlignCenter)
            title_font = QFont()
            title_font.setPointSize(16)
            title_font.setBold(True)
            title_label.setFont(title_font)
            
            # 顶部：原始视频显示区域
            video_layout = QHBoxLayout()
            video_layout.setSpacing(10)
            
            # 左摄像头显示
            left_group = QGroupBox("左摄像头")
            left_layout = QVBoxLayout()
            
            self.left_video_label.setMinimumSize(640, 480)
            self.left_video_label.setMaximumSize(640, 480)
            self.left_video_label.setStyleSheet(
                "border: 2px solid #333; background-color: #000;"
            )
            self.left_video_label.setAlignment(Qt.AlignCenter)
            self.left_video_label.setText("等待左摄像头...")
            
            left_layout.addWidget(self.left_video_label)
            left_group.setLayout(left_layout)
            
            # 右摄像头显示
            right_group = QGroupBox("右摄像头")
            right_layout = QVBoxLayout()
            
            self.right_video_label.setMinimumSize(640, 480)
            self.right_video_label.setMaximumSize(640, 480)
            self.right_video_label.setStyleSheet(
                "border: 2px solid #333; background-color: #000;"
            )
            self.right_video_label.setAlignment(Qt.AlignCenter)
            self.right_video_label.setText("等待右摄像头...")
            
            right_layout.addWidget(self.right_video_label)
            right_group.setLayout(right_layout)
            
            video_layout.addWidget(left_group)
            video_layout.addWidget(right_group)
            
            # 底部：处理结果显示区域
            result_layout = QHBoxLayout()
            result_layout.setSpacing(10)
            
            # AI处理结果可视化
            result_group = QGroupBox("AI处理结果")
            result_inner_layout = QVBoxLayout()
            
            self.result_display_label.setMinimumSize(800, 400)
            self.result_display_label.setMaximumSize(1200, 400)
            self.result_display_label.setStyleSheet(
                "border: 2px solid #333; background-color: #000;"
            )
            self.result_display_label.setAlignment(Qt.AlignCenter)
            self.result_display_label.setText("等待AI处理结果...")
            
            result_inner_layout.addWidget(self.result_display_label)
            result_group.setLayout(result_inner_layout)
            
            # 系统信息面板
            info_group = QGroupBox("系统信息")
            info_layout = QVBoxLayout()
            
            self.info_panel.setMaximumWidth(350)
            self.info_panel.setMinimumHeight(400)
            self.info_panel.setReadOnly(True)
            self.info_panel.setStyleSheet(
                "background-color: #1e1e1e; color: #00ff00; "
                "font-family: 'Consolas', 'Monaco', monospace; "
                "font-size: 11px; border: 1px solid #333;"
            )
            self.info_panel.setText("系统启动中...\n\n等待数据...")
            
            info_layout.addWidget(self.info_panel)
            info_group.setLayout(info_layout)
            
            result_layout.addWidget(result_group)
            result_layout.addWidget(info_group)
            
            # 添加到主布局
            main_layout.addWidget(title_label)
            main_layout.addLayout(video_layout)
            main_layout.addLayout(result_layout)
            
            self.setLayout(main_layout)
        
        def setup_style(self):
            """设置组件样式"""
            # 整体样式
            self.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555;
                    border-radius: 8px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    color: #ffffff;
                }
            """)
        
        @pyqtSlot(object)
        def update_video_display(self, stereo_frame: StereoFrame):
            """更新视频显示"""
            self.current_stereo_frame = stereo_frame
            self.total_frames += 1
            
            # 更新FPS计算
            self._update_fps()
            
            # 注意：实际的图像更新在定时器回调中进行，避免线程冲突
        
        @pyqtSlot(object)
        def update_result_display(self, processing_result: ProcessingResult):
            """更新AI处理结果显示"""
            self.current_result = processing_result
            self.total_results += 1
        
        def _update_displays(self):
            """定时器回调：更新所有显示内容"""
            try:
                # 更新视频显示
                if self.current_stereo_frame:
                    self._update_video_frames()
                
                # 更新结果显示
                if self.current_result:
                    self._update_result_visualization()
                
                # 更新信息面板
                self._update_info_panel()
                
            except Exception as e:
                print(f"显示更新错误: {e}")
        
        def _update_video_frames(self):
            """更新视频帧显示"""
            if not self.current_stereo_frame:
                return
                
            # 转换并显示左摄像头
            left_pixmap = self._cv2_to_qpixmap(self.current_stereo_frame.left_image)
            if left_pixmap:
                scaled_left = left_pixmap.scaled(
                    self.left_video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.left_video_label.setPixmap(scaled_left)
            
            # 转换并显示右摄像头
            right_pixmap = self._cv2_to_qpixmap(self.current_stereo_frame.right_image)
            if right_pixmap:
                scaled_right = right_pixmap.scaled(
                    self.right_video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.right_video_label.setPixmap(scaled_right)
        
        def _update_result_visualization(self):
            """更新结果可视化显示"""
            if not self.current_result or self.current_result.visualization_data is None:
                return
                
            # 显示处理结果可视化
            vis_pixmap = self._cv2_to_qpixmap(self.current_result.visualization_data)
            if vis_pixmap:
                scaled_vis = vis_pixmap.scaled(
                    self.result_display_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.result_display_label.setPixmap(scaled_vis)
        
        def _cv2_to_qpixmap(self, cv_image: np.ndarray) -> Optional[QPixmap]:
            """OpenCV图像转Qt Pixmap"""
            try:
                if cv_image is None or cv_image.size == 0:
                    return None
                    
                # 确保图像是3通道BGR格式
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                    height, width, channel = cv_image.shape
                    bytes_per_line = 3 * width
                    
                    # OpenCV是BGR，Qt需要RGB，所以转换
                    rgb_image = cv_image[:, :, ::-1]  # BGR to RGB
                    
                    q_image = QImage(
                        rgb_image.data.tobytes(), width, height, 
                        bytes_per_line, QImage.Format_RGB888
                    )
                    return QPixmap.fromImage(q_image)
                elif len(cv_image.shape) == 2:
                    # 灰度图像
                    height, width = cv_image.shape
                    q_image = QImage(
                        cv_image.data.tobytes(), width, height, 
                        width, QImage.Format_Grayscale8
                    )
                    return QPixmap.fromImage(q_image)
                    
                return None
                
            except Exception as e:
                print(f"图像转换错误: {e}")
                return None
        
        def _update_info_panel(self):
            """更新信息显示面板"""
            try:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 基础信息
                info_text = f"""═══════════════════════════════════════
🎥 Hybrid SLAM 系统状态
═══════════════════════════════════════
⏰ 时间: {current_time}
📊 总帧数: {self.total_frames}
🔄 处理数: {self.total_results}
📈 显示FPS: {self.current_fps:.1f}

───────────────────────────────────────
"""
                
                # 视频流信息
                if self.current_stereo_frame:
                    info_text += f"""📹 视频流信息:
• 帧ID: {self.current_stereo_frame.frame_id}
• 时间戳: {self.current_stereo_frame.timestamp:.3f}
• 左图尺寸: {self.current_stereo_frame.left_image.shape}
• 右图尺寸: {self.current_stereo_frame.right_image.shape}

───────────────────────────────────────
"""
                
                # AI处理结果信息
                if self.current_result:
                    info_text += f"""🤖 AI处理结果:
• 处理帧ID: {self.current_result.frame_id}
• 处理方法: {self.current_result.method}
• 特征匹配数: {self.current_result.num_matches}
• 匹配置信度: {self.current_result.confidence:.3f}
• 处理时间: {self.current_result.processing_time:.1f}ms

"""
                    # 位姿信息
                    if self.current_result.pose is not None:
                        pose = self.current_result.pose
                        if pose.shape == (4, 4):
                            tx, ty, tz = pose[:3, 3]
                            info_text += f"""🎯 位姿信息:
• X: {tx:.3f}m
• Y: {ty:.3f}m  
• Z: {tz:.3f}m

"""
                    
                    # 错误信息
                    if self.current_result.error:
                        info_text += f"""⚠️  错误信息:
{self.current_result.error}

"""
                else:
                    info_text += """🤖 AI处理结果: 等待中...

"""
                
                info_text += "═══════════════════════════════════════"
                
                self.info_panel.setText(info_text)
                
                # 自动滚动到底部
                self.info_panel.moveCursor(self.info_panel.textCursor().End)
                
            except Exception as e:
                print(f"信息面板更新错误: {e}")
        
        def _update_fps(self):
            """更新FPS计算"""
            self.fps_counter += 1
            
            if self.fps_counter >= 30:  # 每30帧更新一次FPS
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                
                if elapsed > 0:
                    self.current_fps = self.fps_counter / elapsed
                
                # 重置计数器
                self.fps_counter = 0
                self.fps_start_time = current_time
        
        def reset_display(self):
            """重置显示状态"""
            self.current_stereo_frame = None
            self.current_result = None
            self.total_frames = 0
            self.total_results = 0
            self.current_fps = 0.0
            
            # 重置显示内容
            self.left_video_label.setText("等待左摄像头...")
            self.right_video_label.setText("等待右摄像头...")
            self.result_display_label.setText("等待AI处理结果...")
            self.info_panel.setText("系统启动中...\n\n等待数据...")
        
        def get_display_stats(self) -> Dict[str, Any]:
            """获取显示统计信息"""
            return {
                'total_frames': self.total_frames,
                'total_results': self.total_results,
                'current_fps': self.current_fps
            }

else:
    # PyQt5不可用时的占位类
    class QtDisplayWidget:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyQt5不可用，请安装: pip install PyQt5")