#!/usr/bin/env python3
"""
最简单的OpenCV可视化测试
"""

import cv2
import numpy as np
import time

def test_simple_visualization():
    """测试OpenCV可视化"""
    print("Testing simple OpenCV visualization...")
    
    # 创建统一窗口
    cv2.namedWindow('Test Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Test Visualization', 1200, 600)
    
    try:
        for i in range(30):  # 30帧测试
            print(f"Frame {i+1}/30")
            
            # 创建画布
            canvas = np.zeros((600, 1200, 3), dtype=np.uint8)
            
            # 生成测试图像
            left_img = np.random.randint(0, 255, (240, 300, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (240, 300, 3), dtype=np.uint8)
            depth_img = np.random.randint(0, 255, (240, 300, 3), dtype=np.uint8)
            
            # 放置图像到画布
            canvas[10:250, 10:310] = left_img  # 左上
            canvas[10:250, 320:620] = right_img  # 右上
            canvas[270:510, 10:310] = depth_img  # 左下
            
            # 添加标签
            cv2.putText(canvas, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(canvas, "Right Camera", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(canvas, "Depth Map", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(canvas, f"Frame {i+1}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 绘制分割线
            cv2.line(canvas, (310, 0), (310, 600), (100, 100, 100), 2)  # 垂直线
            cv2.line(canvas, (0, 260), (620, 260), (100, 100, 100), 2)  # 水平线
            
            # 显示
            cv2.imshow('Test Visualization', canvas)
            
            # 检查退出键
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("User quit")
                break
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_simple_visualization()