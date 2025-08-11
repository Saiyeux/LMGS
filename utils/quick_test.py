#!/usr/bin/env python3
"""
快速测试脚本 - 测试摄像头和基础功能
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_cameras():
    """快速测试摄像头"""
    print("=== 摄像头快速测试 ===")
    
    available = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
                print(f"摄像头 {i}: OK ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"摄像头 {i}: 可打开但无法读取")
            cap.release()
        else:
            print(f"摄像头 {i}: 不可用")
    
    print(f"\n可用摄像头: {available}")
    return available

def test_qt():
    """测试Qt"""
    print("\n=== Qt测试 ===")
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        print("PyQt5: OK")
        app.quit()
        return True
    except ImportError:
        print("PyQt5: 不可用")
        return False

def test_dependencies():
    """测试依赖"""
    print("\n=== 依赖测试 ===")
    
    deps = {
        'OpenCV': 'cv2',
        'NumPy': 'numpy', 
        'PyTorch': 'torch'
    }
    
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"{name}: OK")
        except ImportError:
            print(f"{name}: 不可用")

def main():
    print("Hybrid SLAM 快速测试")
    print("=" * 30)
    
    # 测试依赖
    test_dependencies()
    
    # 测试摄像头
    cameras = test_cameras()
    
    # 测试Qt
    qt_ok = test_qt()
    
    print("\n" + "=" * 30)
    print("测试总结:")
    print(f"可用摄像头: {len(cameras)} 个")
    print(f"PyQt5: {'可用' if qt_ok else '不可用'}")
    
    if len(cameras) >= 2 and qt_ok:
        print("\n SUCCESS: 可以启动完整系统")
        print(f"建议命令: python run_qt_slam.py --left-cam {cameras[0]} --right-cam {cameras[1]}")
    elif len(cameras) >= 1:
        print(f"\n WARN: 只有1个摄像头，可以测试单摄像头模式")
        print(f"建议命令: python run_qt_slam.py --left-cam {cameras[0]} --right-cam {cameras[0]} --no-ai")
    else:
        print("\n ERROR: 没有可用摄像头，无法启动系统")
    
    return 0

if __name__ == "__main__":
    main()