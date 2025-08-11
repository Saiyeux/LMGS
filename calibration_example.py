"""
棋盘格相机标定示例程序
演示如何使用 ChessboardCalibrator 进行相机标定
"""

import cv2
import numpy as np
import os
from camera_calibration import ChessboardCalibrator


def create_sample_chessboard_images():
    """
    创建一些示例棋盘格图像用于演示
    注意：实际使用时需要用真实相机拍摄的棋盘格图像
    """
    print("创建示例棋盘格图像...")
    
    # 创建目录
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 创建一个虚拟的棋盘格图像
    board_size = (9, 6)
    square_size = 50  # 像素
    
    # 计算图像尺寸
    img_width = (board_size[0] + 1) * square_size
    img_height = (board_size[1] + 1) * square_size
    
    for i in range(5):  # 创建5张示例图像
        # 创建棋盘格图像
        img = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        for row in range(board_size[1] + 1):
            for col in range(board_size[0] + 1):
                if (row + col) % 2 == 0:
                    y1 = row * square_size
                    y2 = (row + 1) * square_size
                    x1 = col * square_size
                    x2 = (col + 1) * square_size
                    img[y1:y2, x1:x2] = 0
        
        # 添加一些随机变换来模拟不同角度
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-10, 10), 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        
        # 转换为彩色图像
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 保存图像
        filename = os.path.join(sample_dir, f"chessboard_{i+1:02d}.jpg")
        cv2.imwrite(filename, img_color)
        
    print(f"已创建 {5} 张示例图像在目录: {sample_dir}")
    return sample_dir


def basic_calibration_example():
    """
    基本标定示例
    """
    print("\n=== 基本标定示例 ===")
    
    # 创建示例图像
    image_dir = create_sample_chessboard_images()
    
    # 创建标定器
    calibrator = ChessboardCalibrator(
        board_size=(9, 6),  # 棋盘格内角点数量
        square_size=25.0    # 方块大小 (mm)
    )
    
    # 批量处理图像
    success_count = calibrator.collect_images(image_dir, "*.jpg")
    
    if success_count >= 3:
        # 执行标定
        if calibrator.calibrate_camera():
            # 保存结果
            calibrator.save_calibration("example_calibration.json")
            
            # 显示结果
            calibrator.print_calibration_results()
            
            return calibrator
        else:
            print("标定失败!")
    else:
        print(f"需要至少3张成功检测角点的图像，当前只有 {success_count} 张")
        
    return None


def single_image_example():
    """
    单张图像处理示例
    """
    print("\n=== 单张图像处理示例 ===")
    
    # 创建示例图像
    image_dir = create_sample_chessboard_images()
    
    # 创建标定器
    calibrator = ChessboardCalibrator(board_size=(9, 6), square_size=25.0)
    
    # 处理单张图像
    test_image = os.path.join(image_dir, "chessboard_01.jpg")
    if os.path.exists(test_image):
        print(f"处理单张图像: {test_image}")
        success = calibrator.detect_corners(test_image, visualize=True)
        print(f"角点检测结果: {'成功' if success else '失败'}")
    else:
        print(f"测试图像不存在: {test_image}")


def load_and_undistort_example():
    """
    加载标定结果并矫正图像示例
    """
    print("\n=== 加载标定结果并矫正图像示例 ===")
    
    # 创建新的标定器实例
    calibrator = ChessboardCalibrator()
    
    # 尝试加载已保存的标定结果
    calibration_file = "example_calibration.json"
    if os.path.exists(calibration_file):
        calibrator.load_calibration(calibration_file)
        
        # 矫正示例图像
        image_dir = "sample_images"
        test_image = os.path.join(image_dir, "chessboard_01.jpg")
        
        if os.path.exists(test_image):
            print(f"矫正图像: {test_image}")
            calibrator.undistort_image(test_image)
        else:
            print(f"测试图像不存在: {test_image}")
    else:
        print(f"标定文件不存在: {calibration_file}")
        print("请先运行基本标定示例")


def interactive_calibration():
    """
    交互式标定程序
    """
    print("\n=== 交互式标定程序 ===")
    
    # 获取用户输入
    while True:
        try:
            board_width = int(input("请输入棋盘格内角点宽度数量 (默认9): ") or "9")
            board_height = int(input("请输入棋盘格内角点高度数量 (默认6): ") or "6")
            square_size = float(input("请输入棋盘格方块大小/mm (默认25.0): ") or "25.0")
            break
        except ValueError:
            print("输入无效，请重新输入")
    
    image_dir = input("请输入图像目录路径 (默认使用示例图像): ").strip()
    if not image_dir:
        image_dir = create_sample_chessboard_images()
    
    # 创建标定器
    calibrator = ChessboardCalibrator(
        board_size=(board_width, board_height),
        square_size=square_size
    )
    
    # 处理图像
    print(f"开始处理目录中的图像: {image_dir}")
    success_count = calibrator.collect_images(image_dir)
    
    if success_count >= 3:
        # 执行标定
        if calibrator.calibrate_camera():
            # 保存结果
            save_path = input("请输入保存文件名 (默认interactive_calibration.json): ").strip()
            if not save_path:
                save_path = "interactive_calibration.json"
                
            calibrator.save_calibration(save_path)
            calibrator.print_calibration_results()
            
            # 询问是否矫正图像
            if input("是否要矫正一张图像? (y/n): ").lower() == 'y':
                test_image = input("请输入要矫正的图像路径: ").strip()
                if test_image and os.path.exists(test_image):
                    calibrator.undistort_image(test_image)
                else:
                    print("图像路径无效")
        else:
            print("标定失败!")
    else:
        print(f"成功检测角点的图像数量不足 ({success_count} < 3)")


def main():
    """
    主示例函数
    """
    print("棋盘格相机标定示例程序")
    print("=" * 40)
    
    while True:
        print("\n请选择示例:")
        print("1. 基本标定示例")
        print("2. 单张图像处理示例")
        print("3. 加载标定结果并矫正图像")
        print("4. 交互式标定")
        print("0. 退出")
        
        choice = input("请输入选择 (0-4): ").strip()
        
        if choice == "1":
            basic_calibration_example()
        elif choice == "2":
            single_image_example()
        elif choice == "3":
            load_and_undistort_example()
        elif choice == "4":
            interactive_calibration()
        elif choice == "0":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")
        
        input("\n按Enter键继续...")


if __name__ == "__main__":
    main()