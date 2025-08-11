import cv2
import numpy as np
import os
import glob
import json
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class ChessboardCalibrator:
    """
    棋盘格相机标定类
    """
    
    def __init__(self, board_size: Tuple[int, int] = (12, 9), square_size: float = 10.0):
        """
        初始化标定器
        
        Args:
            board_size: 棋盘格内角点数量 (width, height)
            square_size: 棋盘格方块的实际大小 (单位: mm或任意单位)
        """
        self.board_size = board_size
        self.square_size = square_size
        
        # 准备3D对象点
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 存储标定数据的列表
        self.objpoints = []  # 3D世界坐标点
        self.imgpoints = []  # 2D图像坐标点
        self.image_size = None
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
    def detect_corners(self, image_path: str, visualize: bool = False) -> bool:
        """
        在单张图像中检测棋盘格角点
        
        Args:
            image_path: 图像路径
            visualize: 是否显示角点检测结果
            
        Returns:
            bool: 是否成功检测到角点
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 设置图像尺寸
        if self.image_size is None:
            self.image_size = gray.shape[::-1]
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # 提高角点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 保存对象点和图像点
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            
            if visualize:
                # 绘制并显示角点
                cv2.drawChessboardCorners(img, self.board_size, corners, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)
                
            return True
        else:
            print(f"未在图像中找到棋盘格: {image_path}")
            return False
    
    def collect_images(self, image_dir: str, image_pattern: str = "*.jpg") -> int:
        """
        批量处理目录中的图像
        
        Args:
            image_dir: 图像目录路径
            image_pattern: 图像文件模式
            
        Returns:
            int: 成功检测到角点的图像数量
        """
        image_paths = glob.glob(os.path.join(image_dir, image_pattern))
        
        if not image_paths:
            print(f"在目录 {image_dir} 中未找到匹配的图像文件")
            return 0
            
        success_count = 0
        print(f"开始处理 {len(image_paths)} 张图像...")
        
        for i, image_path in enumerate(image_paths):
            print(f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            if self.detect_corners(image_path):
                success_count += 1
                
        cv2.destroyAllWindows()
        print(f"成功检测到角点的图像数量: {success_count}/{len(image_paths)}")
        return success_count
    
    def calibrate_camera(self) -> bool:
        """
        执行相机标定
        
        Returns:
            bool: 标定是否成功
        """
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("错误: 没有可用的标定数据")
            return False
            
        if len(self.objpoints) < 3:
            print("警告: 建议使用至少3张图像进行标定")
            
        print(f"开始相机标定，使用 {len(self.objpoints)} 张图像...")
        
        # 执行标定
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )
        
        if ret:
            # 计算重投影误差
            self.calibration_error = self._calculate_reprojection_error()
            print(f"标定成功! 平均重投影误差: {self.calibration_error:.3f} 像素")
            return True
        else:
            print("标定失败!")
            return False
    
    def _calculate_reprojection_error(self) -> float:
        """
        计算重投影误差
        
        Returns:
            float: 平均重投影误差
        """
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            total_points += 1
            
        return total_error / total_points
    
    def save_calibration(self, save_path: str = "camera_calibration.json"):
        """
        保存标定结果
        
        Args:
            save_path: 保存路径
        """
        if self.camera_matrix is None:
            print("错误: 没有标定结果可保存")
            return
            
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'image_size': list(self.image_size),
            'board_size': list(self.board_size),
            'square_size': self.square_size,
            'reprojection_error': float(self.calibration_error),
            'num_images': len(self.objpoints)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)
            
        print(f"标定结果已保存到: {save_path}")
    
    def load_calibration(self, load_path: str = "camera_calibration.json"):
        """
        加载标定结果
        
        Args:
            load_path: 加载路径
        """
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
                
            self.camera_matrix = np.array(calibration_data['camera_matrix'])
            self.dist_coeffs = np.array(calibration_data['distortion_coefficients'])
            self.image_size = tuple(calibration_data['image_size'])
            self.board_size = tuple(calibration_data['board_size'])
            self.square_size = calibration_data['square_size']
            self.calibration_error = calibration_data['reprojection_error']
            
            print(f"标定结果已从 {load_path} 加载")
            self.print_calibration_results()
            
        except FileNotFoundError:
            print(f"错误: 文件 {load_path} 不存在")
        except json.JSONDecodeError:
            print(f"错误: 无法解析 {load_path} 文件")
    
    def print_calibration_results(self):
        """
        打印标定结果
        """
        if self.camera_matrix is None:
            print("没有可用的标定结果")
            return
            
        print("\n=== 相机标定结果 ===")
        print(f"图像尺寸: {self.image_size}")
        print(f"使用图像数量: {len(self.objpoints) if self.objpoints else '未知'}")
        print(f"重投影误差: {self.calibration_error:.3f} 像素")
        print(f"\n相机内参矩阵:")
        print(self.camera_matrix)
        print(f"\n畸变系数:")
        print(self.dist_coeffs.ravel())
        
        # 计算焦距和光心
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        print(f"\n焦距: fx={fx:.2f}, fy={fy:.2f}")
        print(f"光心: cx={cx:.2f}, cy={cy:.2f}")
    
    def undistort_image(self, image_path: str, save_path: Optional[str] = None):
        """
        矫正图像畸变
        
        Args:
            image_path: 输入图像路径
            save_path: 输出图像路径，如果为None则显示图像
        """
        if self.camera_matrix is None:
            print("错误: 没有标定结果")
            return
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
            
        # 矫正畸变
        undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs)
        
        if save_path:
            cv2.imwrite(save_path, undistorted)
            print(f"矫正后的图像已保存到: {save_path}")
        else:
            # 显示对比图像
            comparison = np.hstack((img, undistorted))
            cv2.imshow('Original vs Undistorted', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """
    主函数，提供命令行接口
    """
    parser = argparse.ArgumentParser(description='棋盘格相机标定程序')
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='包含棋盘格图像的目录路径')
    parser.add_argument('--board_width', type=int, default=9, 
                        help='棋盘格内角点宽度数量 (默认: 9)')
    parser.add_argument('--board_height', type=int, default=6, 
                        help='棋盘格内角点高度数量 (默认: 6)')
    parser.add_argument('--square_size', type=float, default=1.0, 
                        help='棋盘格方块大小 (默认: 1.0)')
    parser.add_argument('--save_file', type=str, default='camera_calibration.json', 
                        help='标定结果保存文件 (默认: camera_calibration.json)')
    parser.add_argument('--image_pattern', type=str, default='*.jpg', 
                        help='图像文件模式 (默认: *.jpg)')
    parser.add_argument('--visualize', action='store_true', 
                        help='显示角点检测过程')
    
    args = parser.parse_args()
    
    # 创建标定器
    calibrator = ChessboardCalibrator(
        board_size=(args.board_width, args.board_height),
        square_size=args.square_size
    )
    
    # 收集图像并检测角点
    success_count = calibrator.collect_images(args.image_dir, args.image_pattern)
    
    if success_count > 0:
        # 执行标定
        if calibrator.calibrate_camera():
            # 保存结果
            calibrator.save_calibration(args.save_file)
            calibrator.print_calibration_results()
        else:
            print("标定失败!")
    else:
        print("没有成功检测到角点的图像，无法进行标定")


if __name__ == "__main__":
    main()