# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository (LMGS) implements a 3D reconstruction system that integrates multiple computer vision and SLAM technologies. The main component is a hybrid 3D reconstruction system that combines EfficientLoFTR feature matching with MonoGS (Gaussian Splatting SLAM) for real-time 3D scene reconstruction from camera inputs.

### Core Architecture

The system is built around three main components:

1. **SmartCameraManager** (`3d_reconstruction.py:40-177`) - Handles camera initialization and frame acquisition with automatic fallback to mock data when real cameras are unavailable
2. **HybridAdvanced3DReconstructor** (`3d_reconstruction.py:179-488`) - The main reconstruction engine that integrates multiple algorithms
3. **Interactive3DViewer** and **UltimateVisualization** (`3d_reconstruction.py:490+`) - Real-time 3D visualization and user interface

### Third-party Dependencies

The project integrates two major third-party libraries:

- **EfficientLoFTR** (`thirdparty/EfficientLoFTR/`) - Semi-dense local feature matching for stereo vision
- **MonoGS** (`thirdparty/MonoGS/`) - Gaussian Splatting SLAM system for monocular/RGB-D/stereo SLAM

## Development Commands

### Environment Setup

1. **Main LMGS Environment**:
   ```bash
   # Install the modular package
   pip install -e .
   # OR install dependencies manually
   pip install -r requirements.txt
   ```

2. **EfficientLoFTR Environment** (optional for advanced features):
   ```bash
   cd thirdparty/EfficientLoFTR
   conda env create -f environment.yaml
   conda activate eloftr
   pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

3. **MonoGS Environment** (optional for SLAM features):
   ```bash
   cd thirdparty/MonoGS
   conda env create -f environment.yml
   conda activate MonoGS
   ```

### Running the Modular System

- **Main 3D Reconstruction System** (new modular version):
  ```bash
  # Basic usage
  python run_reconstruction.py
  
  # With CUDA support
  python run_reconstruction.py --device cuda
  
  # Headless mode (no GUI)
  python run_reconstruction.py --headless
  
  # Custom settings
  python run_reconstruction.py --max-cameras 3 --fps-limit 15 --output-dir results
  
  # Different window size
  python run_reconstruction.py --window-size 1920 1080
  ```

- **Legacy Single Script** (deprecated):
  ```bash
  python 3d_reconstruction.py
  ```

- **MonoGS SLAM Examples**:
  ```bash
  # Monocular SLAM
  python thirdparty/MonoGS/slam.py --config thirdparty/MonoGS/configs/mono/tum/fr3_office.yaml
  
  # RGB-D SLAM
  python thirdparty/MonoGS/slam.py --config thirdparty/MonoGS/configs/rgbd/tum/fr3_office.yaml
  
  # Live demo with RealSense
  python thirdparty/MonoGS/slam.py --config thirdparty/MonoGS/configs/live/realsense.yaml
  ```

- **Evaluation Mode**:
  ```bash
  python thirdparty/MonoGS/slam.py --config [config_file] --eval
  ```

### Package Installation

- **Development Installation**:
  ```bash
  pip install -e .
  ```

- **Command Line Usage**:
  ```bash
  lmgs-reconstruction --help
  ```

### Testing and Benchmarks

- **EfficientLoFTR Accuracy Tests**:
  ```bash
  cd thirdparty/EfficientLoFTR
  bash scripts/reproduce_test/outdoor_full_auc.sh
  bash scripts/reproduce_test/indoor_full_auc.sh
  ```

- **EfficientLoFTR Performance Tests**:
  ```bash
  cd thirdparty/EfficientLoFTR
  bash scripts/reproduce_test/indoor_full_time.sh
  bash scripts/reproduce_test/indoor_opt_time.sh
  ```

## Key Configuration Files

- **MonoGS Base Config**: `thirdparty/MonoGS/configs/mono/tum/base_config.yaml` - Contains training parameters, optimization settings, and model configurations
- **EfficientLoFTR Configs**: Located in `thirdparty/EfficientLoFTR/configs/` with different dataset and model configurations

## System Integration Points

The modular system is organized as follows:

### Package Structure
```
lmgs_reconstruction/
├── __init__.py                    # Main package imports
├── camera/                        # Camera management
│   ├── smart_camera_manager.py    # Smart camera detection and fallback
│   └── mock_camera.py            # Simulated camera data generation
├── reconstruction/                # 3D reconstruction algorithms
│   ├── hybrid_reconstructor.py    # Main reconstruction coordinator
│   ├── loftr_processor.py        # EfficientLoFTR integration
│   ├── stereo_processor.py       # Traditional stereo vision
│   └── mono_processor.py         # Monocular processing
├── visualization/                 # Display and visualization
│   ├── viewer_3d.py              # 3D point cloud viewer
│   ├── display_manager.py        # Layout and canvas management
│   └── ultimate_viz.py           # Main visualization coordinator
└── utils/                        # Utilities
    └── dependencies.py           # Third-party library management
```

### Key Integration Points

- **Camera System**: `SmartCameraManager` automatically detects real cameras and falls back to `MockCameraGenerator`
- **Reconstruction Pipeline**: `HybridAdvanced3DReconstructor` coordinates between LoFTR, stereo, and mono processors
- **Visualization System**: `UltimateVisualization` manages the display pipeline with modular components
- **Dependency Management**: Automatic detection and graceful fallback when third-party libraries are unavailable

## Important Dependencies

- **PyTorch**: Required for both EfficientLoFTR and MonoGS (version compatibility important)
- **OpenCV**: Core computer vision operations
- **Matplotlib**: 3D visualization
- **NumPy**: Mathematical operations
- **CUDA**: GPU acceleration (optional but recommended)

## Development Notes

### Modular Design Benefits
- **Separation of Concerns**: Each module has a specific responsibility (camera, reconstruction, visualization)
- **Easy Testing**: Individual components can be tested in isolation
- **Flexible Configuration**: Components can be easily swapped or upgraded
- **Graceful Degradation**: System works even when advanced libraries are unavailable

### Migration from Legacy System
- **Legacy Script**: The original `3d_reconstruction.py` (934 lines) has been split into focused modules
- **Backward Compatibility**: Original functionality is preserved in the new modular system
- **Enhanced Features**: New command-line interface with more configuration options
- **Better Error Handling**: Improved error reporting and recovery mechanisms

### Development Practices
- **Smart Fallbacks**: System automatically uses OpenCV when EfficientLoFTR/MonoGS unavailable
- **Mock Data Generation**: Comprehensive simulation for development without physical cameras
- **Real-time Performance**: Frame rate limiting and intelligent point cloud management
- **Cross-platform Support**: Works on Windows and Linux with appropriate camera backends
- **Modular Installation**: Can install just the core system or with optional advanced features

### Code Organization
- Each module is self-contained with clear interfaces
- Chinese comments preserved where they provide important context
- Configuration through command-line arguments rather than hardcoded values
- Output management with automatic saving and performance monitoring