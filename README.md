# LMGS 3D Reconstruction System

一个模块化的3D重建系统，集成EfficientLoFTR和MonoGS技术，支持实时相机输入和智能回退机制。

## 功能特点

- **模块化架构**: 清晰分离相机管理、重建算法和可视化组件
- **智能相机管理**: 自动检测真实相机，不可用时回退到模拟数据
- **多算法支持**: 集成EfficientLoFTR、传统立体视觉和单目SLAM
- **实时可视化**: 交互式3D点云显示和系统状态监控
- **灵活配置**: 命令行参数支持各种运行模式

## 快速开始

### 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 开发安装
pip install -e .
```

### 基础使用

```bash
# 启动3D重建系统
python run_reconstruction.py

# 使用CUDA加速
python run_reconstruction.py --device cuda

# 无头模式运行
python run_reconstruction.py --headless

# 自定义设置
python run_reconstruction.py --max-cameras 3 --fps-limit 15 --output-dir results
```

## 系统架构

### 模块结构

```
lmgs_reconstruction/
├── camera/           # 相机管理模块
├── reconstruction/   # 3D重建算法
├── visualization/    # 可视化系统
└── utils/           # 工具和依赖管理
```

### 核心组件

1. **SmartCameraManager**: 智能相机管理，支持真实和模拟相机
2. **HybridAdvanced3DReconstructor**: 混合重建算法协调器
3. **UltimateVisualization**: 综合可视化系统
4. **Dependency Management**: 自动检测和管理第三方库

## 命令行选项

```bash
python run_reconstruction.py [选项]

选项:
  --device {cpu,cuda}     计算设备 (默认: cpu)
  --max-cameras INT       最大搜索相机数量 (默认: 5)
  --output-dir PATH       输出目录 (默认: output)
  --fps-limit FLOAT       帧率限制 (默认: 30.0)
  --save-interval INT     保存间隔帧数 (默认: 100)
  --headless             无头模式运行
  --window-size W H       窗口尺寸 (默认: 1600 1000)
```

## 第三方集成

### EfficientLoFTR (可选)

用于高质量特征匹配:

```bash
cd thirdparty/EfficientLoFTR
conda env create -f environment.yaml
conda activate eloftr
```

### MonoGS (可选)

用于Gaussian Splatting SLAM:

```bash
cd thirdparty/MonoGS
conda env create -f environment.yml
conda activate MonoGS
```

## 输出文件

系统会自动保存以下文件到输出目录:

- `reconstruction_frame_XXXXXX.npz`: 定期保存的重建结果
- `final_reconstruction.npz`: 最终完整的3D点云数据

## 性能优化

- **帧率控制**: 可调整的FPS限制避免过载
- **点云管理**: 自动限制点云大小维持性能
- **智能采样**: 显示时智能采样减少渲染负载
- **并行处理**: 支持CUDA加速的特征匹配

## 故障排除

### 相机问题
- 系统会自动检测可用相机
- 无真实相机时自动使用模拟数据
- 支持多种相机后端 (V4L2, DirectShow)

### 依赖问题
- 第三方库不可用时自动回退到OpenCV
- 启动时检查所有依赖状态
- 提供详细的错误信息和建议

### 性能问题
- 降低FPS限制: `--fps-limit 15`
- 使用无头模式: `--headless`
- 减少相机数量: `--max-cameras 2`

## 开发指南

### 添加新的重建算法

1. 在 `lmgs_reconstruction/reconstruction/` 创建新模块
2. 实现标准接口 `process_frames()` 方法
3. 在 `HybridAdvanced3DReconstructor` 中集成

### 自定义可视化

1. 扩展 `DisplayManager` 添加新的显示区域
2. 创建自定义 `Viewer` 组件
3. 在 `UltimateVisualization` 中整合

### 测试

```bash
# 基础功能测试
python run_reconstruction.py --headless --max-cameras 0

# 性能测试
python run_reconstruction.py --fps-limit 60 --save-interval 30
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 相关项目

- [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR): 高效的局部特征匹配
- [MonoGS](https://github.com/muskie82/MonoGS): Gaussian Splatting SLAM