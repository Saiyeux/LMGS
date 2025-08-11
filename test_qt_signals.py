#!/usr/bin/env python3
"""
测试Qt信号连接和显示更新
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

from hybrid_slam.utils.data_structures import StereoFrame, ProcessingResult
from hybrid_slam.gui.qt_display_widget import QtDisplayWidget

class TestSignalEmitter(QObject):
    """测试信号发射器"""
    frame_ready = pyqtSignal(object)
    processing_complete = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        
    def emit_test_frame(self):
        """发射测试帧"""
        # 创建测试立体帧
        left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些可识别的特征
        # 在左图画一个绿色圆圈
        import cv2
        cv2.circle(left_img, (200, 200), 30, (0, 255, 0), -1)
        cv2.circle(right_img, (180, 200), 30, (0, 255, 0), -1)  # 模拟视差
        
        # 添加文字标识
        cv2.putText(left_img, f"Frame {self.frame_counter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(right_img, f"Frame {self.frame_counter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        stereo_frame = StereoFrame(
            frame_id=self.frame_counter,
            timestamp=time.time(),
            left_image=left_img,
            right_image=right_img
        )
        
        print(f"[信号测试] 发射立体帧 {self.frame_counter}")
        self.frame_ready.emit(stereo_frame)
        
        self.frame_counter += 1
    
    def emit_test_result(self):
        """发射测试处理结果"""
        # 创建测试处理结果
        result = ProcessingResult(
            frame_id=self.frame_counter - 1,
            timestamp=time.time()
        )
        
        # 模拟匹配结果
        matches = [
            ([200.0, 200.0], [180.0, 200.0]),  # 模拟匹配点
            ([300.0, 150.0], [285.0, 150.0]),
            ([400.0, 300.0], [380.0, 300.0])
        ]
        
        result.matches = matches
        result.num_matches = len(matches)
        result.confidence = 0.85
        result.processing_time = 150.0
        result.method = "TestMatcher"
        
        # 创建可视化数据
        vis_data = np.zeros((400, 800, 3), dtype=np.uint8)
        # 在可视化数据上画一些东西
        import cv2
        cv2.rectangle(vis_data, (100, 100), (300, 200), (0, 255, 0), 2)
        cv2.putText(vis_data, f"Test Result {result.frame_id}", (150, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_data, f"Matches: {result.num_matches}", (150, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        result.visualization_data = vis_data
        
        print(f"[信号测试] 发射处理结果 {result.frame_id}, 匹配数: {result.num_matches}")
        self.processing_complete.emit(result)

class TestMainWindow(QMainWindow):
    """测试主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Qt信号连接测试")
        self.setGeometry(100, 100, 1800, 1000)
        
        # 创建显示组件
        self.display_widget = QtDisplayWidget()
        self.setCentralWidget(self.display_widget)
        
        # 创建信号发射器
        self.signal_emitter = TestSignalEmitter()
        
        # 建立信号连接
        self.signal_emitter.frame_ready.connect(
            self.display_widget.update_video_display
        )
        self.signal_emitter.processing_complete.connect(
            self.display_widget.update_result_display
        )
        
        # 创建定时器来定期发射信号
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.emit_test_signals)
        
        print("Qt信号连接测试窗口初始化完成")
    
    def emit_test_signals(self):
        """发射测试信号"""
        try:
            # 先发射视频帧
            self.signal_emitter.emit_test_frame()
            
            # 稍后发射处理结果
            QTimer.singleShot(500, self.signal_emitter.emit_test_result)
            
        except Exception as e:
            print(f"信号发射错误: {e}")
            import traceback
            traceback.print_exc()
    
    def start_test(self):
        """开始测试"""
        print("开始Qt信号测试...")
        self.test_timer.start(3000)  # 每3秒发射一次测试信号

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 创建测试窗口
    window = TestMainWindow()
    window.show()
    
    # 开始测试
    window.start_test()
    
    print("Qt信号测试应用已启动")
    print("应该看到:")
    print("1. 左右摄像头显示区域出现测试图像")
    print("2. AI处理结果区域显示可视化")
    print("3. 信息面板显示处理统计")
    print("4. 每3秒更新一次")
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()