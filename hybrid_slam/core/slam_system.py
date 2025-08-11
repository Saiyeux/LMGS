"""
混合SLAM系统核心实现
整合EfficientLoFTR、OpenCV PnP和MonoGS
支持双摄像头实时重建
"""

import sys
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from threading import Thread, Event
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

# 添加thirdparty路径
project_root = Path(__file__).parent.parent.parent
monogs_path = project_root / "thirdparty" / "MonoGS"
sys.path.insert(0, str(monogs_path))

# 导入MonoGS核心组件
try:
    from slam import SLAM
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.slam_backend import BackEnd
    from utils.slam_frontend import FrontEnd
    MONOGS_AVAILABLE = True
    print("MonoGS modules loaded successfully")
except ImportError as e:
    print(f"Warning: MonoGS modules not available: {e}")
    print("Using fallback implementation")
    SLAM = object
    BackEnd = object
    FrontEnd = object
    MONOGS_AVAILABLE = False

# 导入hybrid_slam组件
from ..frontend.hybrid_frontend import HybridFrontEnd, TrackingResult
# 避免循环导入，在函数内导入
from ..utils.config_manager import ConfigManager
from ..utils.performance_monitor import PerformanceMonitor
from ..utils.visualization import RealtimeVisualizer, RealtimeVisualizerExtended
from ..utils.memory_manager import MemoryManager

# StereoFrame已移至utils.data_structures
from ..utils.data_structures import StereoFrame

class HybridSLAMSystem:
    """融合SLAM系统主类 - 支持双摄像头实时重建"""
    
    def __init__(self, config: Dict[str, Any], save_dir: Optional[str] = None):
        """初始化混合SLAM系统"""
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / "results"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备和性能配置
        self.device = config.get('device', 'cuda')
        self.target_fps = config.get('performance_targets', {}).get('target_fps', 20)
        
        # 初始化核心组件
        self._init_logging()
        self._init_core_components()
        self._init_visualization()
        
        # 线程控制
        self.stop_event = Event()
        self.processing_thread = None
        self.visualization_thread = None
        
        # 状态追踪
        self.current_frame_id = 0
        self.is_initialized = False
        self.last_pose = torch.eye(4)
        self.trajectory = []
        
        print(f"HybridSLAMSystem initialized, saving to {self.save_dir}")
    
    def _init_logging(self):
        """初始化日志系统"""
        log_file = self.save_dir / "slam.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HybridSLAM')
    
    def _init_core_components(self):
        """初始化核心组件"""
        # 混合前端
        self.frontend = HybridFrontEnd(self.config.get('frontend', {}))
        
        # 性能监控
        self.perf_monitor = PerformanceMonitor(
            target_fps=self.target_fps,
            memory_limit_gb=self.config.get('performance_targets', {}).get('max_memory_gb', 8)
        )
        
        # 内存管理
        self.memory_manager = MemoryManager(
            max_gpu_memory_gb=self.config.get('performance_targets', {}).get('max_gpu_memory_gb', 6)
        )
        
        # 数据源（将在run时初始化）
        self.dataset = None
        
        # MonoGS后端集成
        self.monogs_slam = None
        self.monogs_backend = None
        self.use_monogs = False
        
        if MONOGS_AVAILABLE:
            try:
                # 初始化MonoGS后端组件
                self._init_monogs_backend()
            except Exception as e:
                self.logger.warning(f"Failed to initialize MonoGS backend: {e}")
                self.use_monogs = False
        
        # 创建数据适配器用于存储3D重建数据
        from ..utils.monogs_adapter import MonoGSAdapter
        self.monogs_adapter = MonoGSAdapter(self.config.get('monogs_adapter', {}))
    
    def _init_visualization(self):
        """初始化可视化组件"""
        if self.config.get('visualization', True):
            vis_config = self.config.get('visualization_config', {})
            # 设置默认值
            default_config = {
                'window_size': (1200, 800),
                'show_trajectory': True,
                'show_pointcloud': True
            }
            # 合并配置，vis_config中的值会覆盖默认值
            final_config = {**default_config, **vis_config}
            
            # 转换window_size为tuple（如果是list）
            if 'window_size' in final_config and isinstance(final_config['window_size'], list):
                final_config['window_size'] = tuple(final_config['window_size'])
            
            # 使用扩展的可视化器以支持3D重建
            self.visualizer = RealtimeVisualizerExtended(**final_config)
        else:
            self.visualizer = None
    
    def run(self):
        """运行SLAM系统主循环"""
        try:
            self.logger.info("Starting Hybrid SLAM System...")
            
            # 初始化数据源
            self._init_dataset()
            
            # 启动可视化线程
            if self.visualizer:
                self.visualization_thread = Thread(target=self._visualization_loop)
                self.visualization_thread.start()
            
            # 主处理循环
            print("Starting main processing loop...")
            self._main_processing_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"SLAM system error: {e}")
            raise
        finally:
            self._shutdown()
    
    def _init_monogs_backend(self):
        """初始化MonoGS后端组件"""
        if not MONOGS_AVAILABLE:
            return
        
        try:
            # 直接使用MonoGS的后端组件，避开完整SLAM初始化
            self.use_monogs = True
            self.monogs_slam = None  # 不使用完整SLAM对象
            
            # 直接创建后端用于3D重建
            from gaussian_splatting.scene.gaussian_model import GaussianModel
            
            # 创建高斯模型用于3D重建
            self.gaussian_model = GaussianModel(sh_degree=0)
            
            # 模拟MonoGS后端的基本功能
            class SimplifiedMonoGSBackend:
                def __init__(self):
                    self.gaussians = GaussianModel(sh_degree=0)
                    self.point_cloud = []
                    self.initialized = False
                
                def add_points(self, points, colors=None):
                    """添加3D点到点云"""
                    if colors is None:
                        colors = [[128, 128, 128]] * len(points)
                    
                    for i in range(len(points)):
                        self.point_cloud.append({
                            'position': points[i][:3].tolist(),
                            'color': colors[i][:3].tolist() if len(colors[i]) >= 3 else [128, 128, 128]
                        })
                
                def get_reconstruction(self):
                    """获取3D重建结果"""
                    if not self.point_cloud:
                        return None
                    
                    points = []
                    colors = []
                    for point_data in self.point_cloud:
                        points.append(point_data['position'])
                        colors.append(point_data['color'])
                    
                    return {
                        'points': np.array(points),
                        'colors': np.array(colors),
                        'type': 'gaussian_splatting'
                    }
            
            self.monogs_backend = SimplifiedMonoGSBackend()
            self.logger.info("MonoGS backend initialized successfully (simplified mode)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MonoGS backend: {e}")
            self.use_monogs = False
    
    def _prepare_monogs_config(self) -> Dict[str, Any]:
        """准备MonoGS配置"""
        # 基础MonoGS配置结构
        monogs_config = {
            'Results': {
                'save_results': True,
                'save_dir': str(self.save_dir),
                'save_trj': True,
                'save_trj_kf_intv': 10,
                'use_gui': False,  # 禁用GUI用于集成模式
                'eval_rendering': False,
                'use_wandb': False
            },
            'Dataset': {
                'type': 'realsense',  # 使用实时相机类型，绕过文件检查
                'dataset_path': str(self.save_dir),  # 临时路径
                'sensor_type': 'monocular',
                'pcd_downsample': 64,
                'pcd_downsample_init': 32,
                'adaptive_pointsize': True,
                'point_size': 0.01,
                'Calibration': {
                    'fx': 525.0,
                    'fy': 525.0,
                    'cx': 320.0,
                    'cy': 240.0,
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0,
                    'k3': 0.0,
                    'width': 640,
                    'height': 480,
                    'depth_scale': 5000.0,
                    'distorted': False
                }
            },
            'Training': {
                'init_itr_num': 500,  # 减少迭代以加快速度
                'init_gaussian_update': 50,
                'init_gaussian_reset': 200,
                'init_gaussian_th': 0.005,
                'init_gaussian_extent': 30,
                'tracking_itr_num': 50,
                'mapping_itr_num': 75,
                'gaussian_update_every': 150,
                'gaussian_update_offset': 50,
                'gaussian_th': 0.7,
                'gaussian_extent': 1.0,
                'gaussian_reset': 2001,
                'size_threshold': 20,
                'kf_interval': 5,
                'window_size': 8,
                'pose_window': 3,
                'edge_threshold': 1.1,
                'rgb_boundary_threshold': 0.01,
                'kf_translation': 0.08,
                'kf_min_translation': 0.05,
                'kf_overlap': 0.9,
                'kf_cutoff': 0.3,
                'prune_mode': 'slam',
                'single_thread': True,  # 单线程模式用于集成
                'spherical_harmonics': False,
                'lr': {
                    'cam_rot_delta': 0.003,
                    'cam_trans_delta': 0.001
                }
            },
            'opt_params': {
                'iterations': 15000,  # 减少迭代
                'position_lr_init': 0.0016,
                'position_lr_final': 0.0000016,
                'position_lr_delay_mult': 0.01,
                'position_lr_max_steps': 15000,
                'feature_lr': 0.0025,
                'opacity_lr': 0.05,
                'scaling_lr': 0.001,
                'rotation_lr': 0.001,
                'percent_dense': 0.01,
                'lambda_dssim': 0.2,
                'densification_interval': 100,
                'opacity_reset_interval': 3000,
                'densify_from_iter': 500,
                'densify_until_iter': 7500,
                'densify_grad_threshold': 0.0002
            },
            'model_params': {
                'sh_degree': 0,
                'source_path': "",
                'model_path': "",
                'resolution': -1,
                'white_background': False,
                'data_device': "cuda"
            },
            'pipeline_params': {
                'convert_SHs_python': False,
                'compute_cov3D_python': False
            }
        }
        
        # 合并用户配置
        user_monogs_config = self.config.get('monogs', {})
        if user_monogs_config:
            # 更新相机参数
            if 'cam' in user_monogs_config:
                cam_config = user_monogs_config['cam']
                calibration = monogs_config['Dataset']['Calibration']
                calibration.update({
                    'fx': cam_config.get('fx', calibration['fx']),
                    'fy': cam_config.get('fy', calibration['fy']),
                    'cx': cam_config.get('cx', calibration['cx']),
                    'cy': cam_config.get('cy', calibration['cy']),
                    'width': cam_config.get('W', calibration['width']),
                    'height': cam_config.get('H', calibration['height'])
                })
            
            # 深度合并其他配置
            self._deep_merge_dict(monogs_config, user_monogs_config)
        
        return monogs_config
    
    def _deep_merge_dict(self, base_dict: Dict, update_dict: Dict):
        """深度合并字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _init_dataset(self):
        """初始化数据源"""
        input_config = self.config.get('input', {})
        source_type = input_config.get('source', 'camera')
        
        if source_type == 'camera':
            # 双摄像头实时输入
            from ..datasets.dataset_factory import create_stereo_camera_dataset
            camera_config = input_config.get('camera', {})
            self.dataset = create_stereo_camera_dataset(
                left_device=camera_config.get('left_device', 0),
                right_device=camera_config.get('right_device', 1),
                resolution=tuple(camera_config.get('resolution', [640, 480])),
                fps=camera_config.get('fps', 30),
                calibration_file=camera_config.get('calibration_file')
            )
        elif source_type == 'mock':
            # 使用模拟数据
            camera_config = input_config.get('camera', {})
            from ..datasets.dataset_factory import create_mock_stereo_dataset
            self.dataset = create_mock_stereo_dataset(
                num_frames=1000,
                resolution=tuple(camera_config.get('resolution', [640, 480]))
            )
        elif source_type == 'dataset':
            # 离线数据集
            dataset_config = input_config.get('dataset_config')
            config_manager = ConfigManager(dataset_config)
            dataset_params = config_manager.load_config()
            
            try:
                from ..datasets.tum_dataset import TUMStereoDataset
                self.dataset = TUMStereoDataset(**dataset_params)
            except ImportError:
                print("TUM dataset not available, using mock dataset")
                from ..datasets.dataset_factory import create_mock_stereo_dataset
                self.dataset = create_mock_stereo_dataset()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        self.logger.info(f"Initialized dataset: {type(self.dataset).__name__}")
    
    def _main_processing_loop(self):
        """主处理循环"""
        frame_time_budget = 1.0 / self.target_fps
        frame_count = 0
        
        print(f"Processing frames with target FPS: {self.target_fps}")
        
        for stereo_frame in self.dataset:
            if self.stop_event.is_set():
                break
            
            frame_count += 1
            if frame_count % 10 == 1:  # 每10帧输出一次
                print(f"Processing frame {frame_count}")
            
            # 测试模式：只处理前50帧
            if frame_count > 50:
                print("Test mode: stopping after 50 frames")
                break
            
            loop_start = time.time()
            
            try:
                # 处理立体帧
                result = self._process_stereo_frame(stereo_frame)
                
                # 更新可视化（无论跟踪是否成功都显示数据）
                if self.visualizer:
                    self._update_visualization(stereo_frame, result)
                
                # 保存中间结果
                self._save_intermediate_results(stereo_frame, result)
                
                # 性能监控
                self.perf_monitor.update_frame_stats(
                    processing_time=time.time() - loop_start,
                    tracking_success=result.success
                )
                
                # 内存管理
                self.memory_manager.cleanup_if_needed()
                
                # 帧率控制
                elapsed = time.time() - loop_start
                if elapsed < frame_time_budget:
                    time.sleep(frame_time_budget - elapsed)
                    
            except Exception as e:
                self.logger.error(f"Frame processing failed: {e}")
                continue
        
        self.logger.info("Main processing loop completed")
    
    def _process_stereo_frame(self, stereo_frame: StereoFrame) -> TrackingResult:
        """处理立体帧"""
        # 计算立体深度（如果需要）
        if stereo_frame.depth_map is None:
            stereo_frame.depth_map = self._compute_stereo_depth(
                stereo_frame.left_image, 
                stereo_frame.right_image,
                stereo_frame.camera_matrices,
                stereo_frame.baseline
            )
        
        # 创建MonoGS兼容的viewpoint（如果MonoGS可用）
        viewpoint = None
        if self.monogs_slam:
            viewpoint = self._create_monogs_viewpoint(stereo_frame)
        
        # 混合前端跟踪（使用左图像）
        tracking_result = self.frontend.tracking(
            cur_frame_idx=stereo_frame.frame_id,
            viewpoint=viewpoint,
            image=stereo_frame.left_image,
            camera_matrix=stereo_frame.camera_matrices[0]
        )
        
        # 将结果传递给MonoGS进行3D重建
        if self.monogs_slam and tracking_result.success:
            self._update_monogs_with_tracking_result(stereo_frame, tracking_result, viewpoint)
        elif self.monogs_adapter:
            # 如果没有MonoGS，使用适配器存储数据
            self.monogs_adapter.add_frame(stereo_frame, tracking_result)
        
        # 更新轨迹
        if tracking_result.success:
            self.trajectory.append({
                'timestamp': stereo_frame.timestamp,
                'frame_id': stereo_frame.frame_id,
                'pose': tracking_result.pose.clone(),
                'confidence': tracking_result.confidence
            })
            self.last_pose = tracking_result.pose.clone()
        
        return tracking_result
    
    def _compute_stereo_depth(self, left_img: np.ndarray, right_img: np.ndarray,
                            camera_matrices: Tuple[np.ndarray, np.ndarray], 
                            baseline: float) -> np.ndarray:
        """计算立体深度图"""
        # 使用OpenCV立体匹配
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 转换为灰度图
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # 计算视差图
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # 转换为深度图
        K_left = camera_matrices[0]
        focal_length = K_left[0, 0]  # 假设fx = fy
        
        depth_map = np.zeros_like(disparity)
        valid_pixels = disparity > 0
        depth_map[valid_pixels] = (focal_length * baseline) / disparity[valid_pixels]
        
        return depth_map
    
    def _create_monogs_viewpoint(self, stereo_frame: StereoFrame):
        """创建MonoGS兼容的viewpoint对象"""
        try:
            # 导入MonoGS的Camera类
            from thirdparty.MonoGS.utils.camera_utils import Camera
            
            # 获取左摄像头参数
            K_left = stereo_frame.camera_matrices[0]
            fx, fy = K_left[0, 0], K_left[1, 1]
            cx, cy = K_left[0, 2], K_left[1, 2]
            
            # 计算视场角
            width, height = stereo_frame.left_image.shape[1], stereo_frame.left_image.shape[0]
            fovx = 2 * torch.atan(width / (2 * fx))
            fovy = 2 * torch.atan(height / (2 * fy))
            
            # 创建初始位姿矩阵（单位矩阵）
            gt_pose = torch.eye(4, device=self.device)
            
            # 将图像转换为tensor
            color_tensor = torch.from_numpy(stereo_frame.left_image).float().permute(2, 0, 1) / 255.0
            color_tensor = color_tensor.to(self.device)
            
            # 深度图转换（如果可用）
            if stereo_frame.depth_map is not None:
                depth_tensor = torch.from_numpy(stereo_frame.depth_map).float().unsqueeze(0)
                depth_tensor = depth_tensor.to(self.device)
            else:
                depth_tensor = torch.zeros((1, height, width), device=self.device)
            
            # 创建投影矩阵
            projection_matrix = torch.tensor([
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, -1, -1],
                [0, 0, 0, 1]
            ], dtype=torch.float32, device=self.device)
            
            # 创建Camera对象
            viewpoint = Camera(
                uid=stereo_frame.frame_id,
                color=color_tensor,
                depth=depth_tensor,
                gt_T=gt_pose,
                projection_matrix=projection_matrix,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                fovx=fovx,
                fovy=fovy,
                image_height=height,
                image_width=width,
                device=self.device
            )
            
            return viewpoint
            
        except Exception as e:
            self.logger.warning(f"Failed to create MonoGS viewpoint: {e}")
            return None
    
    def _update_monogs_with_tracking_result(self, stereo_frame: StereoFrame, 
                                          tracking_result: TrackingResult, 
                                          viewpoint):
        """将跟踪结果更新到MonoGS系统"""
        try:
            if not viewpoint or not self.monogs_frontend:
                return
            
            # 更新viewpoint的位姿
            R = tracking_result.pose[:3, :3].to(self.device)
            T = tracking_result.pose[:3, 3].to(self.device)
            viewpoint.update_RT(R, T)
            
            # 如果这是第一帧，进行初始化
            if not self.is_initialized:
                if hasattr(self.monogs_frontend, 'initialize'):
                    self.monogs_frontend.initialize(stereo_frame.frame_id, viewpoint)
                self.is_initialized = True
                self.logger.info("MonoGS system initialized")
            else:
                # 执行MonoGS跟踪
                if hasattr(self.monogs_frontend, 'tracking'):
                    render_pkg = self.monogs_frontend.tracking(stereo_frame.frame_id, viewpoint)
                    
                    # 可选：更新gaussian场景
                    if self.monogs_backend and hasattr(self.monogs_backend, 'push_to_backend'):
                        # 检查是否是关键帧
                        is_keyframe = self.monogs_frontend.is_keyframe(
                            stereo_frame.frame_id,
                            self.monogs_frontend.kf_indices[-1] if self.monogs_frontend.kf_indices else 0,
                            render_pkg.get("visibility_filter", None),
                            {}  # occ_aware_visibility
                        )
                        
                        if is_keyframe:
                            self.monogs_backend.push_to_backend(
                                stereo_frame.frame_id, 
                                viewpoint, 
                                render_pkg, 
                                stereo_frame.left_image
                            )
            
        except Exception as e:
            self.logger.error(f"Failed to update MonoGS with tracking result: {e}")
    
    def _update_visualization(self, stereo_frame: StereoFrame, result: TrackingResult):
        """更新可视化"""
        if not self.visualizer:
            return
        
        # 获取3D重建数据
        reconstruction_data = self.get_3d_reconstruction()
        
        vis_data = {
            'left_image': stereo_frame.left_image,
            'right_image': stereo_frame.right_image,
            'depth_map': stereo_frame.depth_map,
            'current_pose': result.pose,
            'trajectory': self.trajectory,
            '3d_reconstruction': reconstruction_data,  # 添加3D重建数据
            'tracking_info': {
                'method': result.tracking_method,
                'confidence': result.confidence,
                'num_matches': result.num_matches,
                'processing_time': result.processing_time
            }
        }
        
        self.visualizer.update(vis_data)
    
    def _save_intermediate_results(self, stereo_frame: StereoFrame, result: TrackingResult):
        """保存中间结果"""
        output_config = self.config.get('output', {})
        
        # 保存轨迹
        if output_config.get('save_trajectory', False) and result.success:
            traj_file = self.save_dir / "trajectory.txt"
            pose_str = self._pose_to_tum_format(stereo_frame.timestamp, result.pose)
            
            with open(traj_file, 'a') as f:
                f.write(pose_str + '\n')
        
        # 保存关键帧图像
        if output_config.get('save_keyframes', False):
            if stereo_frame.frame_id % 10 == 0:  # 每10帧保存一次
                keyframe_dir = self.save_dir / "keyframes"
                keyframe_dir.mkdir(exist_ok=True)
                
                cv2.imwrite(
                    str(keyframe_dir / f"left_{stereo_frame.frame_id:06d}.jpg"),
                    stereo_frame.left_image
                )
                cv2.imwrite(
                    str(keyframe_dir / f"right_{stereo_frame.frame_id:06d}.jpg"),
                    stereo_frame.right_image
                )
    
    def _pose_to_tum_format(self, timestamp: float, pose: torch.Tensor) -> str:
        """将位姿转换为TUM格式"""
        # 提取位移和四元数
        t = pose[:3, 3].cpu().numpy()
        R = pose[:3, :3].cpu().numpy()
        
        # 转换为四元数 (w, x, y, z)
        from scipy.spatial.transform import Rotation as R_scipy
        quat = R_scipy.from_matrix(R).as_quat()  # (x, y, z, w)
        quat = [quat[3], quat[0], quat[1], quat[2]]  # 转换为(w, x, y, z)
        
        return f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {quat[0]:.6f}"
    
    def _visualization_loop(self):
        """可视化线程循环"""
        while not self.stop_event.is_set():
            if self.visualizer:
                self.visualizer.render()
            time.sleep(1/60)  # 60 FPS 可视化
    
    def _shutdown(self):
        """关闭系统"""
        self.logger.info("Shutting down Hybrid SLAM System...")
        
        # 保存最终结果
        print("Saving final results...")
        self._save_final_results()
        
        # 停止所有线程
        self.stop_event.set()
        
        if self.visualization_thread:
            self.visualization_thread.join(timeout=1.0)
        
        # 关闭数据源
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
        
        # 关闭可视化
        if self.visualizer:
            self.visualizer.close()
        
        self.logger.info("Shutdown completed")
    
    def _save_final_results(self):
        """保存最终结果"""
        output_config = self.config.get('output', {})
        
        # 保存完整轨迹
        if output_config.get('save_trajectory', True):
            formats = output_config.get('formats', {}).get('trajectory', ['tum'])
            
            for fmt in formats:
                if fmt == 'tum':
                    self._save_trajectory_tum()
                elif fmt == 'kitti':
                    self._save_trajectory_kitti()
        
        # 保存性能报告
        print("Generating performance report...")
        try:
            perf_report = self.perf_monitor.generate_report()
            report_file = self.save_dir / "performance_report.json"
            
            import json
            with open(report_file, 'w') as f:
                json.dump(perf_report, f, indent=2)
            print(f"Performance report saved to {report_file}")
        except Exception as e:
            print(f"Failed to save performance report: {e}")
        
        # 保存MonoGS适配器数据（如果可用）
        print(f"MonoGS adapter available: {self.monogs_adapter is not None}")
        if self.monogs_adapter:
            print("Saving MonoGS adapter data...")
            adapter_dir = self.save_dir / "reconstruction_data"
            self.monogs_adapter.save_reconstruction_data(adapter_dir)
            
            # 导出可视化数据
            vis_files = self.monogs_adapter.export_for_visualization(adapter_dir)
            self.logger.info(f"3D reconstruction data exported: {vis_files}")
            
            # 保存统计信息
            stats = self.monogs_adapter.get_statistics()
            stats_file = self.save_dir / "reconstruction_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        
        self.logger.info(f"Final results saved to {self.save_dir}")
    
    def _save_trajectory_tum(self):
        """保存TUM格式轨迹"""
        traj_file = self.save_dir / "trajectory_complete.txt"
        
        with open(traj_file, 'w') as f:
            for traj_point in self.trajectory:
                pose_str = self._pose_to_tum_format(
                    traj_point['timestamp'], 
                    traj_point['pose']
                )
                f.write(pose_str + '\n')
    
    def _save_trajectory_kitti(self):
        """保存KITTI格式轨迹"""
        traj_file = self.save_dir / "trajectory_kitti.txt"
        
        with open(traj_file, 'w') as f:
            for traj_point in self.trajectory:
                pose = traj_point['pose'].cpu().numpy()
                # KITTI格式：12个数字表示3x4变换矩阵（按行排列）
                pose_flat = pose[:3, :].flatten()
                pose_str = ' '.join([f"{x:.6f}" for x in pose_flat])
                f.write(pose_str + '\n')
    
    def _create_monogs_viewpoint(self, stereo_frame: StereoFrame):
        """创建MonoGS兼容的viewpoint对象"""
        if not MONOGS_AVAILABLE or not self.use_monogs:
            return None
        
        try:
            # 导入MonoGS的相机类
            from utils.camera_utils import Camera
            
            # 获取相机内参
            K = stereo_frame.camera_matrices[0]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # 创建Camera对象
            height, width = stereo_frame.left_image.shape[:2]
            
            # 初始姿态（如果没有则使用单位矩阵）
            pose = self.last_pose.cpu().numpy() if hasattr(self.last_pose, 'cpu') else np.eye(4)
            R = pose[:3, :3]
            T = pose[:3, 3]
            
            camera = Camera(
                colmap_id=stereo_frame.frame_id,
                R=R, T=T,
                FoVx=2 * np.arctan(width / (2 * fx)),
                FoVy=2 * np.arctan(height / (2 * fy)),
                image=torch.from_numpy(stereo_frame.left_image).float().permute(2, 0, 1) / 255.0,
                gt_alpha_mask=None,
                image_name=f"frame_{stereo_frame.frame_id:06d}",
                uid=stereo_frame.frame_id,
                depth=torch.from_numpy(stereo_frame.depth_map).float() if stereo_frame.depth_map is not None else None
            )
            
            return camera
            
        except Exception as e:
            self.logger.warning(f"Failed to create MonoGS viewpoint: {e}")
            return None
    
    def _update_monogs_with_tracking_result(self, stereo_frame: StereoFrame, 
                                          tracking_result: TrackingResult, viewpoint):
        """将跟踪结果传递给MonoGS进行3D重建"""
        if not self.use_monogs or not self.monogs_backend:
            return
        
        try:
            # 从立体视觉提取3D点
            if stereo_frame.depth_map is not None:
                points_3d = self._extract_3d_points_from_stereo(
                    stereo_frame.left_image,
                    stereo_frame.depth_map,
                    stereo_frame.camera_matrices[0],
                    tracking_result.pose
                )
                
                if len(points_3d) > 0:
                    # 提取点和颜色
                    points = np.array([p['position'] for p in points_3d])
                    colors = np.array([p['color'] for p in points_3d])
                    
                    # 添加到MonoGS后端
                    self.monogs_backend.add_points(points, colors)
                    
                    self.logger.debug(f"Added {len(points)} 3D points to MonoGS backend")
            
        except Exception as e:
            self.logger.warning(f"Failed to update MonoGS with tracking result: {e}")
    
    def _extract_3d_points_from_stereo(self, image: np.ndarray, depth_map: np.ndarray, 
                                     camera_matrix: np.ndarray, pose: torch.Tensor) -> List[Dict]:
        """从立体视觉提取3D点用于MonoGS"""
        points_3d = []
        
        # 相机内参
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # 世界坐标系变换矩阵
        T_world_cam = pose.cpu().numpy()
        
        # 采样策略：每隔N个像素提取一个点
        step = 4  # 更密集的采样
        height, width = depth_map.shape
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                
                if depth > 0.1 and depth < 8.0:  # 有效深度范围
                    # 相机坐标系3D点
                    X_cam = (x - cx) * depth / fx
                    Y_cam = (y - cy) * depth / fy
                    Z_cam = depth
                    
                    # 转换到世界坐标系
                    point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                    point_world = T_world_cam @ point_cam
                    
                    # 获取颜色
                    if len(image.shape) == 3:
                        color = image[y, x].astype(float)
                    else:
                        gray = image[y, x]
                        color = [gray, gray, gray]
                    
                    points_3d.append({
                        'position': point_world[:3],
                        'color': color,
                        'confidence': min(1.0, 1.0 / depth)  # 距离越近置信度越高
                    })
        
        return points_3d
    
    def get_3d_reconstruction(self):
        """获取3D重建结果"""
        if self.use_monogs and self.monogs_backend:
            # 从MonoGS后端获取3D重建数据
            try:
                reconstruction = self.monogs_backend.get_reconstruction()
                if reconstruction:
                    return reconstruction
            except Exception as e:
                self.logger.warning(f"Failed to get 3D reconstruction from MonoGS backend: {e}")
        
        # 回退到适配器的点云数据
        if self.monogs_adapter and len(self.monogs_adapter.point_cloud_data) > 0:
            points = []
            colors = []
            for point_data in self.monogs_adapter.point_cloud_data:
                points.append(point_data['position'])
                colors.append(point_data['color'])
            return {
                'points': np.array(points),
                'colors': np.array(colors),
                'type': 'traditional_stereo'
            }
        
        return None
    
    @classmethod
    def from_config_file(cls, config_path: str, save_dir: Optional[str] = None):
        """从配置文件创建SLAM系统"""
        config = ConfigManager.load_config(config_path)
        
        return cls(config, save_dir)