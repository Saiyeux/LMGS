# Hybrid SLAM 使用指南

这是一个双摄像头实时重建系统，整合了EfficientLoFTR特征匹配、OpenCV PnP几何求解和MonoGS渲染优化技术。

## 系统架构

```
Hybrid SLAM System
├── 双摄像头输入 (StereoCameraDataset)
├── 立体视觉前端 (HybridFrontEnd)
│   ├── EfficientLoFTR 特征匹配
│   ├── 立体约束验证
│   └── PnP 位姿估计
├── 后端优化 (MonoGS集成)
├── 实时可视化 (RealtimeVisualizer)
└── 性能监控与内存管理
```

## 快速开始

### 1. 基本测试（使用模拟数据）

```bash
# 基础功能测试
python test_minimal.py

# 完整系统测试
python test_complete_system.py

# 使用模拟数据运行SLAM
python run_hybrid_slam.py --mock
```

### 2. 实际双摄像头运行

```bash
# 使用默认配置
python run_hybrid_slam.py

# 使用自定义配置
python run_hybrid_slam.py --config configs/stereo_camera_config.yaml

# 指定设备和保存目录
python run_hybrid_slam.py --device cuda --save-dir results/experiment1
```

### 3. 高性能运行（关闭可视化）

```bash
# 关闭可视化提高性能
python run_hybrid_slam.py --no-vis --device cuda
```

## 配置文件说明

主配置文件: `configs/stereo_camera_config.yaml`

```yaml
# 计算设备
device: cuda  # 或 'cpu'

# 输入配置
input:
  source: camera  # 'camera', 'mock', 'dataset'
  camera:
    left_device: 0      # 左摄像头设备ID
    right_device: 1     # 右摄像头设备ID
    resolution: [640, 480]  # 图像分辨率
    fps: 30             # 目标帧率
    calibration_file: null  # 标定文件路径（可选）

# 前端配置
frontend:
  matcher_type: loftr   # 特征匹配器类型
  tracking_method: pnp  # 跟踪方法

# 性能配置
performance_targets:
  target_fps: 20        # 目标处理帧率
  max_memory_gb: 8      # 最大内存使用
  max_gpu_memory_gb: 6  # 最大GPU内存使用

# 可视化配置
visualization: true
visualization_config:
  window_size: [1200, 800]
  show_trajectory: true
  show_pointcloud: true

# 输出配置
output:
  save_trajectory: true
  save_keyframes: false
  formats:
    trajectory: [tum, kitti]
```

## 摄像头设置

### 1. USB摄像头连接

确保两个USB摄像头正确连接到电脑：
- 左摄像头：设备ID通常为0
- 右摄像头：设备ID通常为1

### 2. 摄像头测试

```python
import cv2

# 测试左摄像头
cap_left = cv2.VideoCapture(0)
if cap_left.isOpened():
    ret, frame = cap_left.read()
    print(f"左摄像头: {frame.shape if ret else '失败'}")
cap_left.release()

# 测试右摄像头
cap_right = cv2.VideoCapture(1)
if cap_right.isOpened():
    ret, frame = cap_right.read()
    print(f"右摄像头: {frame.shape if ret else '失败'}")
cap_right.release()
```

### 3. 摄像头标定（可选）

如果需要更高精度，可以进行摄像头标定：

```bash
# 使用OpenCV标定工具
python scripts/stereo_calibration.py --left 0 --right 1 --output calibration.json
```

## 系统监控

系统运行时会显示以下信息：

```
Processing frame 1234...
  Left image: (480, 640, 3)
  Right image: (480, 640, 3)
  Matches: 256 -> 189 (after stereo filtering)
  Pose estimation: Success (confidence: 0.85)
  Processing time: 45.2ms
  FPS: 22.1
```

## 输出结果

系统运行后会在指定目录生成以下文件：

```
results/
├── trajectory_complete.txt     # TUM格式轨迹
├── trajectory_kitti.txt        # KITTI格式轨迹
├── performance_report.json     # 性能报告
├── keyframes/                  # 关键帧图像（如果启用）
│   ├── left_000001.jpg
│   └── right_000001.jpg
└── slam.log                   # 系统日志
```

## 性能优化建议

### 1. 硬件要求

- **GPU**: RTX 3060或更高（用于EfficientLoFTR和可视化）
- **内存**: 8GB或更多
- **摄像头**: 支持30fps的USB 3.0摄像头

### 2. 性能调优

```yaml
# 高性能配置
performance_targets:
  target_fps: 30
  max_memory_gb: 16
  max_gpu_memory_gb: 8

# 图像配置
input:
  camera:
    resolution: [1280, 720]  # 更高分辨率
    fps: 30
```

### 3. 内存优化

- 关闭可视化: `--no-vis`
- 使用较低分辨率
- 定期清理GPU内存

## 故障排除

### 1. 摄像头无法打开

```bash
# 检查设备
ls /dev/video*  # Linux
# 或使用设备管理器（Windows）

# 尝试不同设备ID
python run_hybrid_slam.py --config configs/test_camera.yaml
```

### 2. GPU内存不足

```yaml
# 降低性能要求
performance_targets:
  max_gpu_memory_gb: 4
  target_fps: 15
```

### 3. EfficientLoFTR加载失败

这是正常的，系统会自动使用OpenCV特征点作为后备方案。

### 4. 实时性能不足

- 关闭可视化
- 降低图像分辨率
- 使用CPU模式进行测试

## API使用示例

### 程序化使用

```python
from hybrid_slam.core.slam_system import HybridSLAMSystem
from hybrid_slam.utils.config_manager import ConfigManager

# 加载配置
config = ConfigManager.load_config('configs/stereo_camera_config.yaml')

# 创建系统
slam = HybridSLAMSystem(config, save_dir='results/my_experiment')

# 运行系统
slam.run()
```

### 自定义数据源

```python
from hybrid_slam.datasets.dataset_factory import create_mock_stereo_dataset

# 创建自定义数据集
dataset = create_mock_stereo_dataset(num_frames=1000, resolution=(640, 480))

# 遍历数据
for stereo_frame in dataset:
    print(f"Frame {stereo_frame.frame_id}: {stereo_frame.left_image.shape}")
```

## 扩展开发

### 1. 添加新的数据源

实现 `__iter__` 和 `__next__` 方法，返回 `StereoFrame` 对象。

### 2. 自定义特征匹配器

继承 `BaseMatcher` 类并实现相应接口。

### 3. 集成其他后端

修改 `HybridSLAMSystem` 类的后端初始化部分。

## 联系和支持

如有问题请查看：
1. 系统日志文件 `slam.log`
2. 性能报告 `performance_report.json`
3. 运行测试脚本进行诊断