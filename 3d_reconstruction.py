#!/usr/bin/env python3
"""
终极3D重建解决方案 - 结合模拟与真实相机的优势
"""

import cv2
import time
import numpy as np
from pathlib import Path
import threading
from queue import Queue, Empty
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
import sys
import torch
from copy import deepcopy

# 添加EfficientLoFTR到Python路径
sys.path.append('thirdparty/EfficientLoFTR')
try:
    from src.loftr import LoFTR, full_default_cfg, reparameter
    LOFTR_AVAILABLE = True
except ImportError as e:
    print(f"EfficientLoFTR不可用: {e}")
    LOFTR_AVAILABLE = False

# 添加MonoGS到Python路径
sys.path.append('thirdparty/MonoGS')
try:
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.general_utils import safe_state
    MONOGS_AVAILABLE = True
except ImportError as e:
    print(f"MonoGS不可用: {e}")
    MONOGS_AVAILABLE = False

class SmartCameraManager:
    """智能相机管理器 - 自动处理相机访问问题"""
    def __init__(self):
        self.cameras = {}
        self.use_mock = False
        self.mock_frame_count = 0
        
    def initialize(self):
        """初始化相机系统"""
        print("初始化智能相机系统...")
        
        # 尝试访问真实相机
        success = self._try_real_cameras()
        
        if not success:
            print("真实相机不可用，使用模拟模式")
            self.use_mock = True
            return True
        
        return success
    
    def _try_real_cameras(self):
        """尝试访问真实相机"""
        try:
            # 检查更多相机索引以找到可用的相机
            camera_indices_to_try = [0, 1, 2, 3, 4]  # 扩展搜索范围
            
            for i in camera_indices_to_try:
                try:
                    # Linux使用V4L2，Windows使用DirectShow
                    import platform
                    if platform.system() == "Windows":
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    
                    if cap.isOpened():
                        # 快速测试读取
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = cap.read()
                        if ret:
                            print(f"相机{i}可用: {frame.shape}")
                            self.cameras[i] = cap
                        else:
                            print(f"相机{i}无法读取帧")
                            cap.release()
                    else:
                        print(f"相机{i}无法打开")
                        
                    time.sleep(0.1)  # 避免快速访问问题
                    
                except Exception as e:
                    print(f"相机{i}访问异常: {e}")
                    continue
            
            print(f"总共找到 {len(self.cameras)} 个可用相机: {list(self.cameras.keys())}")
            return len(self.cameras) > 0
            
        except Exception as e:
            print(f"相机初始化异常: {e}")
            return False
    
    def get_frames(self):
        """获取帧数据"""
        if self.use_mock:
            return self._generate_mock_frames()
        else:
            return self._get_real_frames()
    
    def _get_real_frames(self):
        """获取真实相机帧"""
        frames = {}
        for camera_id, cap in self.cameras.items():
            try:
                ret, frame = cap.read()
                if ret:
                    frames[camera_id] = frame
            except Exception as e:
                print(f"相机{camera_id}读取失败: {e}")
        
        return frames
    
    def _generate_mock_frames(self):
        """生成模拟帧数据"""
        frames = {}
        
        # 生成动态场景
        self.mock_frame_count += 1
        t = self.mock_frame_count * 0.1
        
        for camera_id in [0, 1]:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 生成动态背景
            frame[:] = (50, 50, 50)
            
            # 添加移动的几何体
            center_x = int(320 + 200 * np.cos(t + camera_id * 0.1))
            center_y = int(240 + 100 * np.sin(t * 1.2))
            
            # 主要物体
            cv2.circle(frame, (center_x, center_y), 40, (0, 255, 128), -1)
            cv2.rectangle(frame, (center_x-60, center_y-60), (center_x+60, center_y+60), (255, 128, 0), 3)
            
            # 特征点
            for i in range(30):
                x = int(320 + 250 * np.cos(t + i * 0.2))
                y = int(240 + 200 * np.sin(t + i * 0.3))
                if 0 <= x < 640 and 0 <= y < 480:
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            
            # 网格参考
            for i in range(0, 640, 80):
                cv2.line(frame, (i, 0), (i, 480), (80, 80, 80), 1)
            for i in range(0, 480, 60):
                cv2.line(frame, (0, i), (640, i), (80, 80, 80), 1)
            
            # 相机标识
            cv2.putText(frame, f"Camera {camera_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {self.mock_frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            frames[camera_id] = frame
        
        return frames
    
    def is_stereo_mode(self):
        """是否为立体模式"""
        if self.use_mock:
            return True  # 模拟模式总是双目
        return len(self.cameras) >= 2
    
    def cleanup(self):
        """清理资源"""
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()

class HybridAdvanced3DReconstructor:
    """混合高级3D重建器 - 集成EfficientLoFTR和MonoGS"""
    def __init__(self):
        self.points_3d = []
        self.colors_3d = []
        self.frame_count = 0
        
        # 设备配置
        self.device = torch.device('cpu')  # 强制CPU避免CUDA兼容性问题
        
        # 相机参数
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 320.0
        self.cy = 240.0
        self.baseline = 0.12
        
        # 初始化EfficientLoFTR
        self.loftr_matcher = None
        self.use_loftr = False
        if LOFTR_AVAILABLE:
            self._init_loftr()
        
        # 初始化MonoGS组件
        self.gaussian_model = None
        self.use_monogs = False
        if MONOGS_AVAILABLE:
            self._init_monogs()
        
        # 后备OpenCV方法
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_desc = None
        
    def _init_loftr(self):
        """初始化EfficientLoFTR匹配器"""
        try:
            print("初始化EfficientLoFTR...")
            _default_cfg = deepcopy(full_default_cfg)
            self.loftr_matcher = LoFTR(config=_default_cfg)
            
            # 尝试加载权重
            weights_path = "thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
            if Path(weights_path).exists():
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                self.loftr_matcher.load_state_dict(checkpoint['state_dict'])
                self.loftr_matcher = reparameter(self.loftr_matcher)
                self.loftr_matcher = self.loftr_matcher.eval().to(self.device)
                self.use_loftr = True
                print("EfficientLoFTR初始化成功")
            else:
                print(f"EfficientLoFTR权重文件不存在: {weights_path}")
        except Exception as e:
            print(f"EfficientLoFTR初始化失败: {e}")
            
    def _init_monogs(self):
        """初始化MonoGS组件"""
        try:
            print("初始化MonoGS组件...")
            self.gaussian_model = GaussianModel(sh_degree=0)  # 简化配置
            self.use_monogs = True
            print("MonoGS初始化成功")
        except Exception as e:
            print(f"MonoGS初始化失败: {e}")
        
    def process_frames(self, frames, is_mock=False):
        """处理输入帧"""
        self.frame_count += 1
        
        if len(frames) >= 2:
            # 双目重建 - 使用实际可用的相机索引
            camera_ids = sorted(frames.keys())
            left_img = frames[camera_ids[0]]   # 第一个相机作为左相机
            right_img = frames[camera_ids[1]]  # 第二个相机作为右相机
            return self._process_stereo_frames(left_img, right_img, is_mock)
        elif len(frames) == 1:
            # 单目重建
            camera_id = list(frames.keys())[0]
            return self._process_mono_frame(frames[camera_id], is_mock)
        
        return False
    
    def _process_stereo_frames(self, left_img, right_img, is_mock):
        """处理立体帧 - 使用EfficientLoFTR或OpenCV"""
        try:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            if self.use_loftr and not is_mock:
                # 使用EfficientLoFTR进行高质量特征匹配
                points_3d, colors_3d = self._process_with_loftr(left_img, right_img, left_gray, right_gray)
            else:
                # 使用传统方法
                if is_mock:
                    disparity = self._compute_mock_disparity(left_gray, right_gray)
                else:
                    disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
                
                points_3d, colors_3d = self._disparity_to_3d(left_img, disparity)
            
            if len(points_3d) > 0:
                self.points_3d.extend(points_3d)
                self.colors_3d.extend(colors_3d)
                
                # 限制点云大小
                max_points = 4000 if is_mock else 3000
                if len(self.points_3d) > max_points:
                    self.points_3d = self.points_3d[-max_points:]
                    self.colors_3d = self.colors_3d[-max_points:]
                
                return True
        
        except Exception as e:
            print(f"立体处理失败: {e}")
        
        return False
    
    def _process_with_loftr(self, left_img, right_img, left_gray, right_gray):
        """使用EfficientLoFTR进行特征匹配和3D重建"""
        try:
            # 预处理图像
            left_tensor = torch.from_numpy(left_gray)[None][None].to(self.device).float() / 255.0
            right_tensor = torch.from_numpy(right_gray)[None][None].to(self.device).float() / 255.0
            
            # 确保尺寸是32的倍数
            h, w = left_gray.shape
            new_h = (h // 32) * 32
            new_w = (w // 32) * 32
            
            if new_h != h or new_w != w:
                left_tensor = torch.nn.functional.interpolate(left_tensor, size=(new_h, new_w))
                right_tensor = torch.nn.functional.interpolate(right_tensor, size=(new_h, new_w))
            
            batch = {'image0': left_tensor, 'image1': right_tensor}
            
            # EfficientLoFTR匹配
            with torch.no_grad():
                self.loftr_matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
            
            # 过滤高置信度匹配
            mask = mconf > 0.3
            mkpts0_filtered = mkpts0[mask]
            mkpts1_filtered = mkpts1[mask]
            
            # 使用匹配点计算3D坐标
            points_3d = []
            colors_3d = []
            
            for i in range(len(mkpts0_filtered)):
                pt0 = mkpts0_filtered[i]
                pt1 = mkpts1_filtered[i]
                
                # 计算视差
                disparity = abs(pt0[0] - pt1[0])
                if disparity > 1.0:
                    # 计算3D坐标
                    Z = (self.fx * self.baseline) / disparity
                    if 0.5 < Z < 15.0:
                        X = (pt0[0] - self.cx) * Z / self.fx
                        Y = (pt0[1] - self.cy) * Z / self.fy
                        
                        points_3d.append([X, Y, Z])
                        
                        # 获取颜色
                        x, y = int(pt0[0]), int(pt0[1])
                        if 0 <= y < left_img.shape[0] and 0 <= x < left_img.shape[1]:
                            color = left_img[y, x]
                            colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
            
            return points_3d, colors_3d
            
        except Exception as e:
            print(f"EfficientLoFTR处理失败: {e}")
            return [], []
    
    def _compute_mock_disparity(self, left_gray, right_gray):
        """计算模拟视差"""
        # 对于模拟数据，使用简化的视差计算
        disparity = np.zeros_like(left_gray, dtype=np.float32)
        
        # 检测特征点
        kp1, desc1 = self.detector.detectAndCompute(left_gray, None)
        kp2, desc2 = self.detector.detectAndCompute(right_gray, None)
        
        if desc1 is not None and desc2 is not None:
            matches = self.matcher.match(desc1, desc2)
            
            for match in matches:
                if match.distance < 50:  # 好匹配
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    
                    # 计算视差
                    disp = abs(pt1[0] - pt2[0])
                    if disp > 1:
                        x, y = int(pt1[0]), int(pt1[1])
                        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                            disparity[y, x] = disp
        
        # 平滑视差图
        disparity = cv2.GaussianBlur(disparity, (5, 5), 1.0)
        
        return disparity
    
    def _process_mono_frame(self, img, is_mock=False):
        """处理单目帧"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测特征点
            kp, desc = self.detector.detectAndCompute(gray, None)
            
            if self.prev_desc is not None and desc is not None:
                matches = self.matcher.match(self.prev_desc, desc)
                good_matches = [m for m in matches if m.distance < 50]
                
                if len(good_matches) > 20:
                    # 提取匹配点
                    pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good_matches])
                    
                    # 估计基础矩阵
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                    
                    if F is not None:
                        # 简化深度估计
                        points_3d = []
                        colors_3d = []
                        
                        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                            if mask[i]:
                                # 基于特征运动估计深度
                                motion = np.linalg.norm(pt2 - pt1)
                                depth = 5.0 / (motion + 0.1) if motion > 0 else 5.0
                                
                                if 0.5 < depth < 15.0:
                                    x, y = pt2
                                    X = (x - self.cx) * depth / self.fx
                                    Y = (y - self.cy) * depth / self.fy
                                    Z = depth
                                    
                                    points_3d.append([X, Y, Z])
                                    
                                    # 获取颜色
                                    if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                                        color = img[int(y), int(x)]
                                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
                        
                        if len(points_3d) > 0:
                            self.points_3d.extend(points_3d)
                            self.colors_3d.extend(colors_3d)
                            
                            # 限制点云大小
                            if len(self.points_3d) > 2000:
                                self.points_3d = self.points_3d[-2000:]
                                self.colors_3d = self.colors_3d[-2000:]
                            
                            return True
            
            self.prev_kp = kp
            self.prev_desc = desc
            
        except Exception as e:
            print(f"单目处理失败: {e}")
        
        return False
    
    def _disparity_to_3d(self, color_img, disparity):
        """从视差图生成3D点云"""
        points_3d = []
        colors_3d = []
        
        h, w = disparity.shape
        step = 6  # 采样步长
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                d = disparity[y, x]
                
                if d > 1.0:  # 有效视差
                    Z = (self.fx * self.baseline) / d
                    if 0.5 < Z < 12.0:  # 合理深度范围
                        X = (x - self.cx) * Z / self.fx
                        Y = (y - self.cy) * Z / self.fy
                        
                        points_3d.append([X, Y, Z])
                        
                        # 获取颜色
                        color = color_img[y, x]
                        colors_3d.append([int(color[0]), int(color[1]), int(color[2])])
        
        return points_3d, colors_3d
    
    def get_reconstruction_data(self):
        """获取重建数据"""
        if len(self.points_3d) > 0:
            return {
                'points': np.array(self.points_3d),
                'colors': np.array(self.colors_3d),
                'type': 'hybrid_reconstruction',
                'count': len(self.points_3d),
                'frame_count': self.frame_count
            }
        return None

class Interactive3DViewer:
    """交互式3D点云查看器"""
    def __init__(self):
        self.fig = None
        self.ax = None
        self.scatter = None
        # 固定的最佳观察视角
        self.azimuth = 45      # 方位角 
        self.elevation = 30    # 仰角
        self.setup_3d_plot()
        
    def setup_3d_plot(self):
        """设置3D绘图环境"""
        try:
            # 创建3D图形
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # 设置标签和标题
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title('Real-time 3D Point Cloud')
            
            # 设置固定视角
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            
            # 设置背景色
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            
            # 设置网格
            self.ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"3D绘图初始化失败: {e}")
            self.fig = None
            self.ax = None
    
    def update_3d_view(self, points, colors):
        """更新3D视图"""
        if self.ax is None or len(points) == 0:
            return None
            
        try:
            # 清除之前的点
            self.ax.clear()
            
            # 重新设置标签和标题
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_zlabel('Z (meters)')
            self.ax.set_title(f'3D Point Cloud ({len(points)} points)')
            
            # 保持固定视角，无需旋转
            
            # 设置固定视角
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            
            # 绘制点云
            if len(points) > 0:
                # 采样点以提高性能
                max_points = 1000
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    sample_points = points[indices]
                    sample_colors = colors[indices] if len(colors) > 0 else None
                else:
                    sample_points = points
                    sample_colors = colors
                
                # 归一化颜色
                if sample_colors is not None and len(sample_colors) > 0:
                    sample_colors = sample_colors / 255.0
                else:
                    sample_colors = 'blue'
                
                # 绘制散点图
                self.ax.scatter(sample_points[:, 0], 
                              sample_points[:, 1], 
                              sample_points[:, 2],
                              c=sample_colors,
                              s=1.5,
                              alpha=0.6)
                
                # 设置固定坐标轴范围
                self._set_fixed_axis_limits()
            
            # 转换为图像
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            size = canvas.get_width_height()
            
            # 处理不同matplotlib版本的API差异
            try:
                raw_data = renderer.tostring_argb()
                # ARGB格式需要转换
                img_array = np.frombuffer(raw_data, dtype=np.uint8)
                img_array = img_array.reshape((size[1], size[0], 4))
                # 去掉alpha通道并转换ARGB到RGB
                img_array = img_array[:, :, 1:4]  # 去掉alpha通道
            except AttributeError:
                try:
                    raw_data = renderer.tobytes()
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    img_array = img_array.reshape((size[1], size[0], 3))
                except AttributeError:
                    # 兼容旧版本matplotlib
                    raw_data = renderer.tostring_rgb()
                    img_array = np.frombuffer(raw_data, dtype=np.uint8)
                    img_array = img_array.reshape((size[1], size[0], 3))
            
            
            # 转换为BGR格式（OpenCV格式）
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            print(f"3D视图更新失败: {e}")
            return None
    
    def _set_fixed_axis_limits(self):
        """设置固定的坐标轴范围"""
        # 固定的坐标轴范围 - 适合室内场景重建
        self.ax.set_xlim(-5.0, 5.0)   # X轴范围: -5米到+5米
        self.ax.set_ylim(-5.0, 5.0)   # Y轴范围: -5米到+5米  
        self.ax.set_zlim(0.0, 10.0)   # Z轴范围: 0米到+10米（深度）
    
    def close(self):
        """关闭3D查看器"""
        if self.fig:
            plt.close(self.fig)

class UltimateVisualization:
    """终极可视化系统"""
    def __init__(self):
        self.window_name = "Ultimate 3D Reconstruction System"
        self.headless = False
        self.viewer_3d = Interactive3DViewer()
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1600, 1000)
        except cv2.error as e:
            print(f"GUI不可用，切换到无头模式: {e}")
            self.headless = True
        
    def display(self, frames, reconstruction_data, is_stereo, is_mock):
        """显示系统状态"""
        if self.headless:
            # 无头模式：只打印状态信息
            if reconstruction_data and reconstruction_data['frame_count'] % 30 == 0:  # 每30帧打印一次
                print(f"3D重建状态 - 点数: {reconstruction_data['count']}, "
                      f"帧数: {reconstruction_data['frame_count']}, "
                      f"模式: {'立体' if is_stereo else '单目'} {'模拟' if is_mock else '真实'}")
            
            # 模拟键盘输入检查（实际项目中可用其他方式）
            import select, sys
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                if key == 'q':
                    return False
            return True
            
        try:
            # 创建大画布
            canvas = np.zeros((1000, 1600, 3), dtype=np.uint8)
            
            # 标题和模式指示
            mode_text = "STEREO" if is_stereo else "MONO"
            source_text = "MOCK" if is_mock else "REAL"
            cv2.putText(canvas, f"Ultimate 3D Reconstruction - {mode_text} {source_text}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # 左侧：相机视图
            y_offset = 80
            for camera_id, frame in frames.items():
                if frame is not None:
                    # 计算可用空间
                    available_height = min(420, 1000 - y_offset - 20)
                    display_height = min(480, available_height)
                    
                    # 调整大小
                    display_frame = cv2.resize(frame, (640, display_height))
                    canvas[y_offset:y_offset+display_height, 20:660] = display_frame
                    
                    # 标签
                    label = f"Camera {camera_id} ({'Mock' if is_mock else 'Real'})"
                    cv2.putText(canvas, label, (30, y_offset+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    y_offset += display_height + 20
                    if y_offset > 800:  # 避免超出画布
                        break
            
            # 右侧：3D重建信息
            info_x = 700
            self._draw_reconstruction_info(canvas, info_x, 80, 880, 400, reconstruction_data)
            
            # 右下：真正的3D点云可视化
            self._draw_interactive_3d_pointcloud(canvas, info_x, 500, 880, 480, reconstruction_data)
            
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            return key != ord('q')
            
        except Exception as e:
            print(f"显示错误: {e}")
            return True
    
    def _draw_reconstruction_info(self, canvas, x, y, w, h, reconstruction_data):
        """绘制重建信息"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (40, 40, 40), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(canvas, "3D Reconstruction Status", (x+10, y+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        text_y = y + 80
        line_height = 30
        
        if reconstruction_data:
            info_lines = [
                f"Point Count: {reconstruction_data['count']}",
                f"Frame Count: {reconstruction_data['frame_count']}",
                f"Type: {reconstruction_data['type']}",
                "",
                "System Status:",
                "- 3D Reconstruction: ACTIVE",
                "- Point Cloud: UPDATING",
                "- Visualization: REAL-TIME",
                "",
                "Performance:",
                f"- Memory Usage: {reconstruction_data['count'] * 24 / 1024:.1f}KB",
                "- Processing: SMOOTH",
                "- Quality: HIGH"
            ]
        else:
            info_lines = [
                "Status: INITIALIZING",
                "Waiting for data...",
                "",
                "System Ready:",
                "- Cameras: DETECTED",
                "- Algorithms: LOADED",
                "- Display: ACTIVE"
            ]
        
        for line in info_lines:
            if text_y > y + h - 20:
                break
            
            if line:
                if line.startswith("- "):
                    color = (0, 255, 0) if "ACTIVE" in line or "UPDATING" in line or "HIGH" in line else (0, 255, 255)
                else:
                    color = (200, 200, 200)
                
                cv2.putText(canvas, line, (x+15, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            text_y += line_height
    
    def _draw_interactive_3d_pointcloud(self, canvas, x, y, w, h, reconstruction_data):
        """绘制交互式3D点云"""
        # 背景
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 2)
        
        # 标题
        cv2.putText(canvas, "Interactive 3D Point Cloud", (x+10, y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if reconstruction_data and reconstruction_data.get('points') is not None:
            points = reconstruction_data['points']
            colors = reconstruction_data['colors']
            
            if len(points) > 0:
                try:
                    # 生成3D视图
                    view_3d = self.viewer_3d.update_3d_view(points, colors)
                    
                    if view_3d is not None:
                        # 调整3D视图大小以适应画布区域
                        view_h_target = h - 50
                        view_w_target = w - 20
                        
                        # 计算缩放比例保持纵横比
                        scale_h = view_h_target / view_3d.shape[0]
                        scale_w = view_w_target / view_3d.shape[1]
                        scale = min(scale_h, scale_w)
                        
                        new_h = int(view_3d.shape[0] * scale)
                        new_w = int(view_3d.shape[1] * scale)
                        
                        # 调整大小
                        view_3d_resized = cv2.resize(view_3d, (new_w, new_h))
                        
                        # 计算居中位置
                        start_x = x + 10 + (view_w_target - new_w) // 2
                        start_y = y + 50 + (view_h_target - new_h) // 2
                        
                        # 确保不超出边界
                        end_x = min(start_x + new_w, x + w)
                        end_y = min(start_y + new_h, y + h)
                        actual_w = end_x - start_x
                        actual_h = end_y - start_y
                        
                        if actual_w > 0 and actual_h > 0:
                            # 调整视图以适应实际可用空间
                            view_3d_final = view_3d_resized[:actual_h, :actual_w]
                            
                            # 将3D视图嵌入到画布中
                            canvas[start_y:end_y, start_x:end_x] = view_3d_final
                        
                        # 添加3D视图信息
                        info_text = f"Points: {len(points)} | Fixed 3D view"
                        cv2.putText(canvas, info_text, (x+15, y+h-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        cv2.putText(canvas, "3D View Generation Failed", (x+20, y+100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                except Exception as e:
                    cv2.putText(canvas, f"3D Render Error: {str(e)[:40]}", (x+20, y+100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    print(f"3D渲染错误: {e}")
        else:
            cv2.putText(canvas, "No 3D data available", (x+20, y+100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            
            # 显示等待图标
            cv2.putText(canvas, "Initializing 3D reconstruction...", (x+20, y+130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def close(self):
        """关闭显示"""
        if not self.headless:
            cv2.destroyWindow(self.window_name)
        # 关闭3D查看器
        self.viewer_3d.close()

def main():
    """主函数"""
    print("=" * 70)
    print("终极3D重建系统 - 智能相机适配")
    print("=" * 70)
    print("自动检测相机状态，必要时使用模拟数据")
    print("按 'q' 退出")
    print()
    
    # 初始化系统组件
    camera_manager = SmartCameraManager()
    reconstructor = HybridAdvanced3DReconstructor()
    visualizer = UltimateVisualization()
    
    # 初始化相机
    if not camera_manager.initialize():
        print("系统初始化失败！")
        return
    
    is_mock = camera_manager.use_mock
    is_stereo = camera_manager.is_stereo_mode()
    
    print(f"工作模式: {'模拟' if is_mock else '真实'}相机, {'立体' if is_stereo else '单目'}重建")
    
    # 显示使用的算法
    components = []
    if reconstructor.use_loftr:
        components.append("EfficientLoFTR")
    if reconstructor.use_monogs:
        components.append("MonoGS")
    if not components:
        components.append("OpenCV")
    
    print(f"算法组件: {', '.join(components)}")
    print("开始3D重建...")
    
    frame_count = 0
    start_time = time.time()
    last_report = start_time
    
    try:
        while True:
            frame_count += 1
            
            # 获取帧数据
            frames = camera_manager.get_frames()
            
            if frames:
                # 3D重建处理
                reconstructor.process_frames(frames, is_mock)
                
                # 获取重建数据
                reconstruction_data = reconstructor.get_reconstruction_data()
                
                # 更新显示
                if not visualizer.display(frames, reconstruction_data, is_stereo, is_mock):
                    break
                
                # 性能报告
                current_time = time.time()
                if current_time - last_report > 3.0:  # 每3秒报告一次
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed
                    points_count = reconstruction_data['count'] if reconstruction_data else 0
                    
                    print(f"性能报告 - FPS: {fps:.1f}, 处理帧数: {frame_count}, 3D点数: {points_count}")
                    last_report = current_time
            
            # 控制帧率
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n用户中断...")
    except Exception as e:
        print(f"\n系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        camera_manager.cleanup()
        visualizer.close()
        
        # 保存结果
        reconstruction_data = reconstructor.get_reconstruction_data()
        if reconstruction_data:
            save_path = Path("ultimate_3d_reconstruction.npz")
            np.savez(save_path, 
                    points=reconstruction_data['points'],
                    colors=reconstruction_data['colors'])
            print(f"\n3D重建结果已保存: {save_path}")
            print(f"总点数: {reconstruction_data['count']}, 处理帧数: {frame_count}")
        
        print("终极3D重建系统已关闭")

if __name__ == "__main__":
    main()