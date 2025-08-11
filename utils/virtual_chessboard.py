import cv2
import numpy as np
import tkinter as tk


def get_monitor_dpi():
    """获取显示器DPI"""
    root = tk.Tk()
    root.withdraw()  # 隐藏窗口
    dpi = root.winfo_fpixels('1i')  # 获取每英寸像素数
    root.destroy()
    return dpi


def create_chessboard(squares_width=12, squares_height=9, square_size_cm=1.0):
    """
    创建虚拟标定板
    
    参数:
    - squares_width: 棋盘格宽度方向的正方形个数（默认12）
    - squares_height: 棋盘格高度方向的正方形个数（默认9）
    - square_size_cm: 每个正方形的边长（厘米，默认1.0）
    """
    
    # 获取显示器DPI
    dpi = get_monitor_dpi()
    print(f"显示器DPI: {dpi:.2f}")
    
    # 计算每厘米的像素数（1英寸 = 2.54厘米）
    pixels_per_cm = dpi / 2.54
    print(f"每厘米像素数: {pixels_per_cm:.2f}")
    
    # 计算每个正方形的像素大小
    square_size_pixels = int(square_size_cm * pixels_per_cm)
    print(f"每个正方形边长: {square_size_pixels} 像素")
    
    # 计算棋盘格的总尺寸
    board_width = squares_width * square_size_pixels
    board_height = squares_height * square_size_pixels
    print(f"棋盘格总尺寸: {board_width} x {board_height} 像素")
    
    # 创建棋盘格图像
    chessboard = np.zeros((board_height, board_width), dtype=np.uint8)
    
    # 填充棋盘格模式
    for i in range(squares_height):
        for j in range(squares_width):
            # 计算当前正方形的位置
            y1 = i * square_size_pixels
            y2 = (i + 1) * square_size_pixels
            x1 = j * square_size_pixels
            x2 = (j + 1) * square_size_pixels
            
            # 交替填充黑白色（棋盘格模式）
            if (i + j) % 2 == 0:
                chessboard[y1:y2, x1:x2] = 255  # 白色
            # 黑色正方形保持为0（已经初始化为0）
    
    return chessboard, square_size_pixels


def display_fullscreen_chessboard():
    """在4K屏幕下全屏显示标定板"""
    
    # 创建棋盘格
    chessboard, square_size_pixels = create_chessboard()
    
    print("\n操作说明:")
    print("- 按 'ESC' 或 'q' 键退出全屏")
    print("- 按 's' 键保存标定板图像")
    print("- 按 'f' 键切换全屏/窗口模式")
    
    # 创建窗口
    window_name = "Virtual Chessboard - Camera Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 设置为全屏
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 获取屏幕尺寸
    screen_width = 3840  # 4K宽度
    screen_height = 2160  # 4K高度
    
    # 创建与屏幕尺寸相同的背景图像（黑色背景）
    background = np.zeros((screen_height, screen_width), dtype=np.uint8)
    
    # 计算棋盘格在屏幕中的居中位置
    board_height, board_width = chessboard.shape
    start_y = (screen_height - board_height) // 2
    start_x = (screen_width - board_width) // 2
    
    # 将棋盘格放置在背景中央
    background[start_y:start_y + board_height, start_x:start_x + board_width] = chessboard
    
    is_fullscreen = True
    
    while True:
        # 显示图像
        cv2.imshow(window_name, background)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC或q键退出
            break
        elif key == ord('s'):  # s键保存图像
            filename = f"chessboard_12x9_{square_size_pixels}px_per_square.png"
            cv2.imwrite(filename, chessboard)
            print(f"标定板图像已保存为: {filename}")
        elif key == ord('f'):  # f键切换全屏模式
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("切换到全屏模式")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, board_width, board_height)
                print("切换到窗口模式")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("虚拟标定板生成器")
    print("=" * 50)
    print("配置信息:")
    print("- 棋盘格尺寸: 12 x 9 个正方形")
    print("- 正方形边长: 1.0 cm")
    print("- 目标显示: 4K全屏 (3840x2160)")
    print("=" * 50)
    
    try:
        display_fullscreen_chessboard()
    except KeyboardInterrupt:
        print("\n程序被用户终止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()