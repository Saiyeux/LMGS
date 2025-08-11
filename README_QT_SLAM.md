# Hybrid SLAM Qt架构 - 使用说明

## 概述

基于设计方案重构的Hybrid SLAM系统，采用Qt+OpenCV+AI架构，提供统一的视频流处理和AI集成界面。

## 新架构特点

### 🎯 **核心改进**
- **统一Qt界面**：替代混乱的OpenCV窗口显示
- **模块化设计**：清晰的组件分离和接口定义
- **线程安全**：Qt信号槽机制确保跨线程通信
- **高性能**：异步处理管道，GPU加速支持
- **易扩展**：插件式AI模型集成

### 🏗 **架构组件**
1. **VideoStreamManager** - 双摄像头同步采集
2. **AIProcessingPipeline** - AI算法集成管道
3. **QtDisplayWidget** - 统一显示界面
4. **MainWindow** - 主应用程序窗口

## 安装依赖

```bash
# 必需依赖
pip install PyQt5 opencv-python numpy

# 可选依赖（AI功能）
pip install torch torchvision
```

## 使用方法

### 1. 快速启动

```bash
# 启动Qt界面
python run_qt_slam.py

# 指定摄像头设备
python run_qt_slam.py --left-cam 0 --right-cam 1

# 禁用AI功能，仅显示视频流
python run_qt_slam.py --no-ai

# 设置目标FPS
python run_qt_slam.py --fps 30
```

### 2. 系统测试

```bash
# 运行完整测试
python run_qt_slam.py --test

# 检查系统依赖
python run_qt_slam.py --check-deps

# 或直接运行测试脚本
python test_qt_slam.py
```

## 界面说明

### 主界面布局

```
┌─────────────────────────────────────────────────────────────┐
│                  Hybrid SLAM Qt系统                        │
├────────────────────────┬────────────────────────────────────┤
│      左摄像头           │         右摄像头                    │
│    (640x480)          │       (640x480)                   │
├────────────────────────┴────────────────────────────────────┤
│              AI处理结果可视化              │    系统信息面板    │
│             (1200x400)                   │   (350x400)     │
│                                          │                 │
└──────────────────────────────────────────┴─────────────────┘
```

### 功能说明

**视频显示区域：**
- 左摄像头：实时显示左摄像头视频流
- 右摄像头：实时显示右摄像头视频流
- 自动缩放保持比例

**AI结果区域：**
- 特征匹配可视化
- 处理结果叠加显示
- 实时性能指标

**信息面板：**
- 系统运行状态
- FPS统计信息
- AI处理结果详情
- 错误信息显示

### 菜单功能

**文件菜单：**
- 保存/加载配置
- 退出程序

**控制菜单：**
- 开始/停止系统
- 重置显示

**设置菜单：**
- 摄像头配置
- AI模型配置
- 系统参数调整

## 配置说明

配置文件位置：`config/qt_slam_config.json`

```json
{
  "left_device": 0,           // 左摄像头设备ID
  "right_device": 1,          // 右摄像头设备ID  
  "target_fps": 30,           // 目标FPS
  "enable_loftr": true,       // 启用EfficientLoFTR
  "enable_pnp": true,         // 启用PnP位姿估计
  "enable_mono_gs": false,    // 启用MonoGS（实验性）
  "confidence_threshold": 0.8, // 匹配置信度阈值
  "buffer_size": 30           // 缓冲区大小
}
```

## API接口

### VideoStreamManager

```python
from hybrid_slam.core.video_stream_manager import VideoStreamManager

manager = VideoStreamManager(left_device=0, right_device=1)
manager.start_capture()
frame = manager.get_latest_frame()
stats = manager.get_stats()
manager.stop_capture()
```

### AIProcessingPipeline

```python
from hybrid_slam.core.ai_processing_pipeline import AIProcessingPipeline

pipeline = AIProcessingPipeline(config)
pipeline.initialize_models()
pipeline.start_processing()
pipeline.enqueue_frame(stereo_frame)
pipeline.stop_processing()
```

## 性能优化

### 1. 系统设置
- 使用USB 3.0摄像头获得最佳性能
- 确保GPU驱动程序最新（AI加速）
- 关闭不必要的后台程序

### 2. 配置调优
```json
{
  "target_fps": 20,          // 降低FPS减少CPU负载
  "buffer_size": 10,         // 减小缓冲区节省内存
  "confidence_threshold": 0.9 // 提高阈值减少误匹配
}
```

### 3. AI模型优化
- 仅启用需要的AI功能
- 使用GPU加速（如果可用）
- 调整处理分辨率

## 故障排除

### 1. 摄像头问题
```bash
# 检查摄像头设备
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(4)])"

# 测试摄像头
python run_qt_slam.py --left-cam 0 --right-cam 1 --no-ai
```

### 2. PyQt5问题
```bash
# 重新安装PyQt5
pip uninstall PyQt5
pip install PyQt5

# 检查Qt环境
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
```

### 3. AI模型问题
```bash
# 跳过AI功能测试基础流程
python run_qt_slam.py --no-ai

# 检查AI依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 4. 性能问题
- 降低目标FPS
- 减小缓冲区大小
- 关闭不需要的AI功能
- 检查CPU/GPU使用率

## 开发说明

### 文件结构
```
hybrid_slam/
├── core/
│   ├── video_stream_manager.py    # 视频流管理
│   └── ai_processing_pipeline.py  # AI处理管道
├── gui/
│   ├── qt_display_widget.py       # 显示组件
│   └── main_window.py             # 主窗口
└── utils/
    └── data_structures.py         # 数据结构
```

### 扩展AI模型
1. 继承相应的基类
2. 实现必需的接口方法
3. 在配置中启用新模型
4. 更新处理管道

### 自定义界面
1. 继承QtDisplayWidget
2. 重写显示方法
3. 添加新的布局元素
4. 连接新的信号槽

## 版本信息

- **版本**: 1.0.0
- **架构**: Qt5 + OpenCV + PyTorch
- **兼容性**: Windows/Linux/macOS
- **Python**: 3.7+

---

更多信息请参考：
- 设计文档：`design_document.md`
- 测试脚本：`test_qt_slam.py`
- 启动脚本：`run_qt_slam.py`