"""
MainWindow - 主应用程序窗口
整合所有组件的统一界面
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt5.QtWidgets import (QMainWindow, QApplication, QMenuBar, QStatusBar,
                                QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
                                QLabel, QLineEdit, QPushButton, QSpinBox,
                                QCheckBox, QGroupBox, QDialogButtonBox)
    from PyQt5.QtCore import Qt, pyqtSlot, QTimer
    from PyQt5.QtGui import QIcon
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    print("PyQt5不可用，请安装: pip install PyQt5")

if QT_AVAILABLE:
    from .qt_display_widget import QtDisplayWidget
    from ..core.video_stream_manager import VideoStreamManager
    from ..core.ai_processing_pipeline import AIProcessingPipeline


if QT_AVAILABLE:
    class ConfigDialog(QDialog):
        """配置对话框"""
        
        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config.copy()
            self.setWindowTitle("系统配置")
            self.setModal(True)
            self.setup_ui()
        
        def setup_ui(self):
            layout = QVBoxLayout()
            
            # 摄像头配置
            camera_group = QGroupBox("摄像头配置")
            camera_layout = QVBoxLayout()
            
            # 左摄像头设备ID
            left_layout = QHBoxLayout()
            left_layout.addWidget(QLabel("左摄像头设备ID:"))
            self.left_device_spin = QSpinBox()
            self.left_device_spin.setRange(0, 10)
            self.left_device_spin.setValue(self.config.get('left_device', 0))
            left_layout.addWidget(self.left_device_spin)
            camera_layout.addLayout(left_layout)
            
            # 右摄像头设备ID
            right_layout = QHBoxLayout()
            right_layout.addWidget(QLabel("右摄像头设备ID:"))
            self.right_device_spin = QSpinBox()
            self.right_device_spin.setRange(0, 10)
            self.right_device_spin.setValue(self.config.get('right_device', 1))
            right_layout.addWidget(self.right_device_spin)
            camera_layout.addLayout(right_layout)
            
            # 目标FPS
            fps_layout = QHBoxLayout()
            fps_layout.addWidget(QLabel("目标FPS:"))
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(10, 60)
            self.fps_spin.setValue(self.config.get('target_fps', 30))
            fps_layout.addWidget(self.fps_spin)
            camera_layout.addLayout(fps_layout)
            
            camera_group.setLayout(camera_layout)
            layout.addWidget(camera_group)
            
            # AI模型配置
            ai_group = QGroupBox("AI模型配置")
            ai_layout = QVBoxLayout()
            
            self.enable_loftr_check = QCheckBox("启用EfficientLoFTR特征匹配")
            self.enable_loftr_check.setChecked(
                self.config.get('enable_loftr', True)
            )
            ai_layout.addWidget(self.enable_loftr_check)
            
            self.enable_pnp_check = QCheckBox("启用PnP位姿估计")
            self.enable_pnp_check.setChecked(
                self.config.get('enable_pnp', True)
            )
            ai_layout.addWidget(self.enable_pnp_check)
            
            self.enable_mono_gs_check = QCheckBox("启用MonoGS 3D重建")
            self.enable_mono_gs_check.setChecked(
                self.config.get('enable_mono_gs', False)
            )
            ai_layout.addWidget(self.enable_mono_gs_check)
            
            # 置信度阈值
            conf_layout = QHBoxLayout()
            conf_layout.addWidget(QLabel("置信度阈值:"))
            self.confidence_spin = QSpinBox()
            self.confidence_spin.setRange(50, 95)
            self.confidence_spin.setValue(
                int(self.config.get('confidence_threshold', 0.8) * 100)
            )
            self.confidence_spin.setSuffix("%")
            conf_layout.addWidget(self.confidence_spin)
            ai_layout.addLayout(conf_layout)
            
            ai_group.setLayout(ai_layout)
            layout.addWidget(ai_group)
            
            # 按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)
            
            self.setLayout(layout)
        
        def get_config(self) -> Dict[str, Any]:
            """获取配置"""
            return {
                'left_device': self.left_device_spin.value(),
                'right_device': self.right_device_spin.value(),
                'target_fps': self.fps_spin.value(),
                'enable_loftr': self.enable_loftr_check.isChecked(),
                'enable_pnp': self.enable_pnp_check.isChecked(),
                'enable_mono_gs': self.enable_mono_gs_check.isChecked(),
                'confidence_threshold': self.confidence_spin.value() / 100.0
            }


    class MainWindow(QMainWindow):
        """
        主应用程序窗口
        职责:
        - 整合所有组件
        - 菜单和工具栏
        - 系统控制逻辑
        - 配置管理
        """
        
        def __init__(self):
            super().__init__()
            
            # 默认配置
            self.config = {
                'left_device': 0,
                'right_device': 1,
                'target_fps': 30,
                'enable_loftr': True,
                'enable_pnp': True,
                'enable_mono_gs': False,
                'confidence_threshold': 0.8,
                'buffer_size': 30
            }
            
            # 核心组件
            self.video_manager = None
            self.ai_pipeline = None
            self.display_widget = None
            
            # 状态变量
            self.is_running = False
            
            # 配置文件路径
            self.config_file = Path("config/qt_slam_config.json")
            self.config_file.parent.mkdir(exist_ok=True)
            
            # 初始化界面
            self.init_ui()
            self.setup_connections()
            self.setup_menus()
            
            # 加载配置
            self.load_config()
            
            # 状态更新定时器
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self.update_status)
            self.status_timer.start(1000)  # 每秒更新状态
        
        def init_ui(self):
            """初始化用户界面"""
            self.setWindowTitle("Hybrid SLAM - Qt+OpenCV+AI视频处理系统")
            self.setGeometry(100, 100, 1800, 1200)
            
            # 创建显示组件
            self.display_widget = QtDisplayWidget()
            self.setCentralWidget(self.display_widget)
            
            # 状态栏
            self.status_bar = self.statusBar()
            self.status_bar.showMessage("就绪")
            
            # 设置窗口图标（如果有的话）
            try:
                self.setWindowIcon(QIcon("icons/slam.ico"))
            except:
                pass
        
        def setup_connections(self):
            """建立信号连接"""
            # 这里会在start_system中建立连接，因为组件是动态创建的
            pass
        
        def setup_menus(self):
            """设置菜单栏"""
            menubar = self.menuBar()
            
            # 文件菜单
            file_menu = menubar.addMenu('文件(&F)')
            file_menu.addAction('保存配置(&S)', self.save_config, 'Ctrl+S')
            file_menu.addAction('加载配置(&L)', self.load_config, 'Ctrl+O')
            file_menu.addSeparator()
            file_menu.addAction('退出(&X)', self.close, 'Alt+F4')
            
            # 控制菜单
            control_menu = menubar.addMenu('控制(&C)')
            self.start_action = control_menu.addAction('开始(&S)', self.start_system, 'F5')
            self.stop_action = control_menu.addAction('停止(&T)', self.stop_system, 'F6')
            self.stop_action.setEnabled(False)
            control_menu.addSeparator()
            control_menu.addAction('重置显示(&R)', self.reset_display, 'Ctrl+R')
            
            # 设置菜单
            settings_menu = menubar.addMenu('设置(&E)')
            settings_menu.addAction('系统配置(&C)', self.show_config_dialog, 'Ctrl+P')
            settings_menu.addSeparator()
            settings_menu.addAction('摄像头诊断(&D)', self.diagnose_cameras, 'Ctrl+D')
            
            # 帮助菜单
            help_menu = menubar.addMenu('帮助(&H)')
            help_menu.addAction('关于(&A)', self.show_about_dialog, 'F1')
        
        @pyqtSlot()
        def start_system(self):
            """启动系统"""
            try:
                self.status_bar.showMessage("正在启动系统...")
                
                # 创建视频管理器
                self.video_manager = VideoStreamManager(
                    left_device=self.config['left_device'],
                    right_device=self.config['right_device'],
                    target_fps=self.config['target_fps'],
                    buffer_size=self.config['buffer_size']
                )
                
                # 创建AI处理管道
                ai_config = {
                    'enable_loftr': self.config['enable_loftr'],
                    'enable_pnp': self.config['enable_pnp'],
                    'enable_mono_gs': self.config['enable_mono_gs'],
                    'confidence_threshold': self.config['confidence_threshold']
                }
                self.ai_pipeline = AIProcessingPipeline(ai_config)
                
                # 建立信号连接
                self.setup_runtime_connections()
                
                # 初始化并启动AI模型
                if not self.ai_pipeline.initialize_models():
                    QMessageBox.warning(self, "警告", "AI模型初始化失败，但系统将继续运行")
                
                if not self.ai_pipeline.start_processing():
                    QMessageBox.critical(self, "错误", "AI处理线程启动失败")
                    return
                
                # 启动视频采集
                if not self.video_manager.start_capture():
                    QMessageBox.critical(self, "错误", "视频采集启动失败")
                    return
                
                self.is_running = True
                self.start_action.setEnabled(False)
                self.stop_action.setEnabled(True)
                self.status_bar.showMessage("系统运行中...")
                
                print("系统启动成功")
                
            except Exception as e:
                error_msg = f"系统启动失败: {e}"
                print(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
                self.cleanup_system()
        
        def setup_runtime_connections(self):
            """建立运行时信号连接"""
            if self.video_manager and self.ai_pipeline and self.display_widget:
                # 视频流 -> 显示
                self.video_manager.frame_ready.connect(
                    self.display_widget.update_video_display
                )
                
                # 视频流 -> AI处理
                self.video_manager.frame_ready.connect(
                    self.ai_pipeline.process_stereo_frame
                )
                
                # AI处理结果 -> 显示
                self.ai_pipeline.processing_complete.connect(
                    self.display_widget.update_result_display
                )
                
                # 错误处理
                self.video_manager.error_occurred.connect(self.handle_error)
                self.ai_pipeline.error_occurred.connect(self.handle_error)
        
        @pyqtSlot()
        def stop_system(self):
            """停止系统"""
            try:
                self.status_bar.showMessage("正在停止系统...")
                
                self.cleanup_system()
                
                self.is_running = False
                self.start_action.setEnabled(True)
                self.stop_action.setEnabled(False)
                self.status_bar.showMessage("系统已停止")
                
                print("系统停止成功")
                
            except Exception as e:
                error_msg = f"系统停止失败: {e}"
                print(error_msg)
                self.handle_error(error_msg)
        
        def cleanup_system(self):
            """清理系统资源"""
            try:
                if self.video_manager:
                    self.video_manager.stop_capture()
                    self.video_manager = None
                
                if self.ai_pipeline:
                    self.ai_pipeline.stop_processing()
                    self.ai_pipeline = None
                
                if self.display_widget:
                    self.display_widget.reset_display()
                    
            except Exception as e:
                print(f"系统清理错误: {e}")
        
        @pyqtSlot()
        def reset_display(self):
            """重置显示"""
            if self.display_widget:
                self.display_widget.reset_display()
                self.status_bar.showMessage("显示已重置")
        
        @pyqtSlot(str)
        def handle_error(self, error_msg: str):
            """错误处理"""
            print(f"系统错误: {error_msg}")
            QMessageBox.warning(self, "系统错误", error_msg)
            self.status_bar.showMessage(f"错误: {error_msg}")
        
        @pyqtSlot()
        def show_config_dialog(self):
            """显示配置对话框"""
            dialog = ConfigDialog(self.config, self)
            if dialog.exec_() == QDialog.Accepted:
                new_config = dialog.get_config()
                
                # 如果系统正在运行，询问是否重启
                if self.is_running:
                    reply = QMessageBox.question(
                        self, "配置更改", 
                        "配置已更改。是否重启系统以应用新配置？",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        self.stop_system()
                        self.config.update(new_config)
                        self.start_system()
                    else:
                        self.config.update(new_config)
                else:
                    self.config.update(new_config)
                
                self.save_config()
                self.status_bar.showMessage("配置已更新")
        
        @pyqtSlot()
        def diagnose_cameras(self):
            """摄像头诊断"""
            try:
                import cv2
                msg = "摄像头诊断结果:\n\n"
                
                # 检测可用摄像头
                available_cameras = []
                for i in range(6):  # 检查前6个设备
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        # 尝试读取一帧
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(i)
                            msg += f"摄像头 {i}: 可用 ({frame.shape[1]}x{frame.shape[0]})\n"
                        else:
                            msg += f"摄像头 {i}: 打开但无法读取\n"
                        cap.release()
                    else:
                        msg += f"摄像头 {i}: 不可用\n"
                
                msg += f"\n找到 {len(available_cameras)} 个可用摄像头: {available_cameras}\n"
                msg += f"\n当前配置:"
                msg += f"\n  左摄像头: {self.config['left_device']}"
                msg += f"\n  右摄像头: {self.config['right_device']}"
                
                if len(available_cameras) >= 2:
                    msg += f"\n\n建议配置:"
                    msg += f"\n  左摄像头: {available_cameras[0]}"
                    msg += f"\n  右摄像头: {available_cameras[1]}"
                else:
                    msg += f"\n\nWARN: 可用摄像头不足，需要至少2个摄像头进行立体SLAM"
                
                QMessageBox.information(self, "摄像头诊断", msg)
                
            except Exception as e:
                QMessageBox.critical(self, "诊断错误", f"摄像头诊断失败: {e}")
        
        @pyqtSlot()
        def show_about_dialog(self):
            """显示关于对话框"""
            about_text = """
<h3>Hybrid SLAM 系统</h3>
<p>版本: 1.0.0</p>
<p>基于Qt+OpenCV+AI的实时视频处理系统</p>

<p><b>主要功能:</b></p>
<ul>
<li>双摄像头实时视频采集</li>
<li>EfficientLoFTR特征匹配</li>
<li>PnP位姿估计</li>
<li>MonoGS 3D重建 (实验性)</li>
<li>统一Qt界面显示</li>
</ul>

<p><b>技术栈:</b></p>
<ul>
<li>PyQt5 - GUI框架</li>
<li>OpenCV - 计算机视觉</li>
<li>PyTorch - 深度学习</li>
<li>NumPy - 数值计算</li>
</ul>
            """
            
            QMessageBox.about(self, "关于 Hybrid SLAM", about_text)
        
        def save_config(self):
            """保存配置到文件"""
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                print(f"配置已保存到: {self.config_file}")
            except Exception as e:
                print(f"保存配置失败: {e}")
        
        def load_config(self):
            """从文件加载配置"""
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        self.config.update(loaded_config)
                    print(f"配置已加载: {self.config_file}")
            except Exception as e:
                print(f"加载配置失败: {e}")
        
        def update_status(self):
            """更新状态栏"""
            if self.is_running:
                stats_info = []
                
                # 获取视频管理器统计
                if self.video_manager:
                    video_stats = self.video_manager.get_stats()
                    stats_info.append(f"视频FPS: {video_stats['fps']:.1f}")
                
                # 获取AI处理统计
                if self.ai_pipeline:
                    ai_stats = self.ai_pipeline.get_stats()
                    stats_info.append(f"处理: {ai_stats['total_processed']}")
                
                # 获取显示统计
                if self.display_widget:
                    display_stats = self.display_widget.get_display_stats()
                    stats_info.append(f"显示FPS: {display_stats['current_fps']:.1f}")
                
                if stats_info:
                    status_text = f"运行中 - {' | '.join(stats_info)}"
                else:
                    status_text = "运行中"
                
                self.status_bar.showMessage(status_text)
        
        def closeEvent(self, event):
            """窗口关闭事件"""
            if self.is_running:
                reply = QMessageBox.question(
                    self, "退出确认", 
                    "系统正在运行，确定要退出吗？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.stop_system()
                    self.save_config()
                    event.accept()
                else:
                    event.ignore()
            else:
                self.save_config()
                event.accept()


    def main():
        """主函数"""
        app = QApplication(sys.argv)
        
        # 设置应用信息
        app.setApplicationName("Hybrid SLAM")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("SLAM Research Lab")
        
        # 创建主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用
        sys.exit(app.exec_())


    if __name__ == "__main__":
        main()

else:
    def main():
        print("PyQt5不可用，请安装: pip install PyQt5")
        return 1