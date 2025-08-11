"""
MonoGS适配器 - 用于在没有MonoGS时存储和管理3D重建数据
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from .data_structures import StereoFrame
from ..frontend.hybrid_frontend import TrackingResult


class MonoGSAdapter:
    """MonoGS数据适配器，用于存储和管理3D重建相关数据"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化适配器"""
        self.config = config
        self.save_intermediate = config.get('save_intermediate', True)
        self.max_frames = config.get('max_frames', 1000)
        
        # 存储数据结构
        self.frames_data = []
        self.keyframes_data = []
        self.point_cloud_data = []
        self.camera_poses = []
        
        # 统计信息
        self.total_frames = 0
        self.successful_tracks = 0
        self.keyframe_count = 0
        
    def add_frame(self, stereo_frame: StereoFrame, tracking_result: TrackingResult):
        """添加帧数据"""
        self.total_frames += 1
        
        if tracking_result.success:
            self.successful_tracks += 1
            
            # 构建帧数据
            frame_data = {
                'frame_id': stereo_frame.frame_id,
                'timestamp': stereo_frame.timestamp,
                'pose': tracking_result.pose.cpu().numpy().tolist(),
                'tracking_method': tracking_result.tracking_method,
                'confidence': tracking_result.confidence,
                'num_matches': tracking_result.num_matches,
                'num_inliers': tracking_result.num_inliers,
                'reprojection_error': tracking_result.reprojection_error,
                'camera_matrix': stereo_frame.camera_matrices[0].tolist(),
                'baseline': stereo_frame.baseline
            }
            
            # 添加深度信息（如果可用）
            if stereo_frame.depth_map is not None:
                depth_stats = self._compute_depth_statistics(stereo_frame.depth_map)
                frame_data['depth_stats'] = depth_stats
            
            self.frames_data.append(frame_data)
            
            # 简单的关键帧选择策略
            if self._should_be_keyframe(tracking_result):
                self._add_keyframe(stereo_frame, tracking_result)
            
            # 限制存储的帧数
            if len(self.frames_data) > self.max_frames:
                self.frames_data.pop(0)
    
    def _should_be_keyframe(self, tracking_result: TrackingResult) -> bool:
        """判断是否应该作为关键帧"""
        if len(self.keyframes_data) == 0:
            return True
        
        # 基于运动幅度的简单策略
        if len(self.camera_poses) > 0:
            last_pose = torch.tensor(self.camera_poses[-1])
            current_pose = tracking_result.pose
            
            translation_diff = torch.norm(current_pose[:3, 3] - last_pose[:3, 3])
            rotation_diff = torch.norm(current_pose[:3, :3] - last_pose[:3, :3])
            
            return translation_diff > 0.1 or rotation_diff > 0.05
        
        return True
    
    def _add_keyframe(self, stereo_frame: StereoFrame, tracking_result: TrackingResult):
        """添加关键帧"""
        self.keyframe_count += 1
        
        keyframe_data = {
            'keyframe_id': self.keyframe_count,
            'frame_id': stereo_frame.frame_id,
            'timestamp': stereo_frame.timestamp,
            'pose': tracking_result.pose.cpu().numpy().tolist(),
            'confidence': tracking_result.confidence
        }
        
        self.keyframes_data.append(keyframe_data)
        self.camera_poses.append(tracking_result.pose.cpu().numpy().tolist())
        
        # 如果有深度图，提取3D点
        if stereo_frame.depth_map is not None:
            points_3d = self._extract_3d_points(
                stereo_frame.left_image,
                stereo_frame.depth_map,
                stereo_frame.camera_matrices[0],
                tracking_result.pose
            )
            
            if len(points_3d) > 0:
                self.point_cloud_data.extend(points_3d)
    
    def _compute_depth_statistics(self, depth_map: np.ndarray) -> Dict[str, float]:
        """计算深度图统计信息"""
        valid_depths = depth_map[depth_map > 0]
        
        if len(valid_depths) == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'valid_ratio': 0}
        
        return {
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'mean': float(np.mean(valid_depths)),
            'std': float(np.std(valid_depths)),
            'valid_ratio': float(len(valid_depths) / depth_map.size)
        }
    
    def _extract_3d_points(self, image: np.ndarray, depth_map: np.ndarray, 
                          camera_matrix: np.ndarray, pose: torch.Tensor) -> List[Dict]:
        """从深度图提取3D点"""
        points_3d = []
        
        # 相机内参
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # 世界坐标系变换矩阵
        T_world_cam = pose.cpu().numpy()
        
        # 采样策略：每隔N个像素提取一个点
        step = 8
        height, width = depth_map.shape
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                
                if depth > 0.1 and depth < 10.0:  # 有效深度范围
                    # 相机坐标系3D点
                    X_cam = (x - cx) * depth / fx
                    Y_cam = (y - cy) * depth / fy
                    Z_cam = depth
                    
                    # 转换到世界坐标系
                    point_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                    point_world = T_world_cam @ point_cam
                    
                    # 获取颜色
                    if len(image.shape) == 3:
                        color = image[y, x].tolist()
                    else:
                        color = [image[y, x]] * 3
                    
                    points_3d.append({
                        'position': point_world[:3].tolist(),
                        'color': color,
                        'confidence': 1.0  # 简单设置
                    })
        
        return points_3d
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_frames': self.total_frames,
            'successful_tracks': self.successful_tracks,
            'success_rate': self.successful_tracks / max(self.total_frames, 1),
            'keyframe_count': self.keyframe_count,
            'total_3d_points': len(self.point_cloud_data),
            'trajectory_length': len(self.camera_poses)
        }
    
    def save_reconstruction_data(self, save_dir: Path):
        """保存重建数据"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存轨迹数据
        trajectory_file = save_dir / "camera_trajectory.json"
        with open(trajectory_file, 'w') as f:
            json.dump({
                'poses': self.camera_poses,
                'keyframes': self.keyframes_data,
                'metadata': self.get_statistics()
            }, f, indent=2)
        
        # 保存点云数据（如果有）
        if self.point_cloud_data:
            pointcloud_file = save_dir / "point_cloud.json"
            with open(pointcloud_file, 'w') as f:
                json.dump({
                    'points': self.point_cloud_data,
                    'count': len(self.point_cloud_data)
                }, f, indent=2)
        
        # 保存TUM格式轨迹（用于评估）
        tum_trajectory_file = save_dir / "trajectory_tum.txt"
        with open(tum_trajectory_file, 'w') as f:
            for frame_data in self.frames_data:
                if 'timestamp' in frame_data and 'pose' in frame_data:
                    pose_matrix = np.array(frame_data['pose'])
                    t = pose_matrix[:3, 3]
                    R = pose_matrix[:3, :3]
                    
                    # 转换为四元数
                    from scipy.spatial.transform import Rotation as R_scipy
                    quat = R_scipy.from_matrix(R).as_quat()  # (x, y, z, w)
                    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]  # 转换为(w, x, y, z)
                    
                    timestamp = frame_data['timestamp']
                    f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                           f"{quat_wxyz[1]:.6f} {quat_wxyz[2]:.6f} {quat_wxyz[3]:.6f} {quat_wxyz[0]:.6f}\n")
        
        print(f"Reconstruction data saved to {save_dir}")
        return str(save_dir)
    
    def export_for_visualization(self, save_dir: Path) -> Dict[str, str]:
        """导出可视化数据"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 导出PLY格式点云
        if self.point_cloud_data:
            ply_file = save_dir / "point_cloud.ply"
            self._export_ply(ply_file)
            exported_files['pointcloud'] = str(ply_file)
        
        # 导出相机轨迹
        if self.camera_poses:
            camera_file = save_dir / "camera_poses.txt"
            with open(camera_file, 'w') as f:
                for pose in self.camera_poses:
                    pose_matrix = np.array(pose)
                    # 写入12个数字（3x4矩阵按行展开）
                    pose_flat = pose_matrix[:3, :].flatten()
                    f.write(' '.join([f"{x:.6f}" for x in pose_flat]) + '\n')
            exported_files['trajectory'] = str(camera_file)
        
        return exported_files
    
    def _export_ply(self, ply_file: Path):
        """导出PLY格式点云"""
        with open(ply_file, 'w') as f:
            # PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.point_cloud_data)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 点数据
            for point in self.point_cloud_data:
                pos = point['position']
                color = point['color']
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{int(color[0])} {int(color[1])} {int(color[2])}\n")