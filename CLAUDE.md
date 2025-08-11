# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision research project that implements a **Hybrid SLAM System** integrating:

- **EfficientLoFTR** - Semi-dense local feature matching with sparse-level speed
- **OpenCV PnP** - Perspective-n-Point pose estimation for geometric constraints
- **MonoGS** - 3D Gaussian Splatting SLAM supporting mono, stereo, and RGB-D cameras

The main contribution is a real-time dual-camera reconstruction system that combines feature matching, geometric solving, and neural radiance field rendering.

## Development Environment

### Environment Setup Commands
```bash
# Windows setup (recommended)
setup_environment.bat

# Manual setup (note: uses MonoGS environment.yml)
conda env create -f thirdparty/MonoGS/environment.yml
conda activate MonoGS
cd thirdparty/MonoGS
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
cd ../..

# Development installation
pip install -e .
```

### Third-party Dependencies
The project requires compiling CUDA extensions for MonoGS:
```bash
cd thirdparty/MonoGS
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
```

## Main Commands

### Hybrid SLAM System
```bash
# Basic dual-camera run
python run_hybrid_slam.py --config configs/stereo_camera_config.yaml

# Mock data testing
python run_hybrid_slam.py --mock

# High performance (no visualization)
python run_hybrid_slam.py --no-vis --device cuda

# Custom save directory
python run_hybrid_slam.py --save-dir results/experiment1
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration

# Run with coverage
python run_tests.py --coverage

# Check dependencies
python run_tests.py --check-deps

# Or use pytest directly
pytest                              # All tests
pytest tests/unit/                  # Unit tests only
pytest tests/integration/           # Integration tests only
pytest -v --cov=hybrid_slam         # With coverage

# Quick testing without visualization
python run_hybrid_slam.py --mock --no-vis
```

### MonoGS (standalone)
```bash
cd thirdparty/MonoGS

# Basic mono SLAM
python slam.py --config configs/mono/tum/fr3_office.yaml

# RGB-D mode
python slam.py --config configs/rgbd/tum/fr3_office.yaml
python slam.py --config configs/rgbd/replica/office0.yaml

# Evaluation mode
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```

### EfficientLoFTR (standalone)
```bash
cd thirdparty/EfficientLoFTR

# Training
python train.py [data_cfg_path] [main_cfg_path] --exp_name=[exp_name]

# Testing
python test.py [data_cfg_path] [main_cfg_path] --ckpt_path=[checkpoint_path]

# Evaluation scripts (requires bash)
bash scripts/reproduce_test/indoor_full_auc.sh
bash scripts/reproduce_test/outdoor_full_auc.sh
```

## Code Architecture

### Main Package Structure
```
hybrid_slam/                    # Main hybrid SLAM package
├── core/                      # Core system components
│   └── slam_system.py        # HybridSLAMSystem main class
├── frontend/                 # Frontend tracking modules
│   ├── hybrid_frontend.py    # Main frontend coordinator
│   ├── feature_tracker.py    # Feature-based tracking
│   ├── pnp_tracker.py        # PnP-based tracking
│   └── render_tracker.py     # Render-based tracking
├── matchers/                 # Feature matching implementations
│   ├── loftr_matcher.py      # EfficientLoFTR wrapper
│   └── matcher_base.py       # Base matcher interface
├── solvers/                  # Geometric solvers
│   ├── pnp_solver.py         # PnP pose estimation
│   └── pose_estimator.py     # Pose estimation utilities
└── utils/                    # Utility modules
    ├── config_manager.py     # Configuration management
    ├── data_structures.py    # Core data structures (StereoFrame, etc.)
    ├── performance_monitor.py # Performance monitoring
    └── visualization.py      # Real-time visualization
```

### Key Classes and Entry Points
- **HybridSLAMSystem** (`hybrid_slam/core/slam_system.py`): Main system orchestrator
- **HybridFrontEnd** (`hybrid_slam/frontend/hybrid_frontend.py`): Frontend tracking coordinator
- **EfficientLoFTRMatcher** (`hybrid_slam/matchers/loftr_matcher.py`): Feature matching interface
- **PnPSolver** (`hybrid_slam/solvers/pnp_solver.py`): Geometric pose solver
- **StereoFrame** (`hybrid_slam/utils/data_structures.py`): Core data structure for stereo input

### Configuration System
The system uses hierarchical YAML configurations:
- `configs/base/`: Base configuration templates
- `configs/datasets/`: Dataset-specific configurations (TUM, Replica, EuRoC)
- `configs/models/`: Model configurations (EfficientLoFTR variants)

## Development Workflow

### Code Quality Tools
```bash
# For MonoGS components (if ruff is installed)
cd thirdparty/MonoGS
ruff check .
ruff format .

# For EfficientLoFTR components (if pylint is installed)
cd thirdparty/EfficientLoFTR
pylint src/

# For main hybrid_slam package
black hybrid_slam/ --check
flake8 hybrid_slam/

# Basic Python syntax check
python -m py_compile hybrid_slam/**/*.py
```

### Adding New Components
1. **New Matchers**: Inherit from `BaseMatcher` in `hybrid_slam/matchers/matcher_base.py`
2. **New Trackers**: Inherit from `BaseTracker` in `hybrid_slam/core/base_tracker.py`
3. **New Datasets**: Implement iterator returning `StereoFrame` objects
4. **New Solvers**: Follow interface patterns in `hybrid_slam/solvers/`

### Dataset Preparation
```bash
# Download MonoGS datasets (Linux/WSL)
bash thirdparty/MonoGS/scripts/download_tum.sh
bash thirdparty/MonoGS/scripts/download_replica.sh
bash thirdparty/MonoGS/scripts/download_euroc.sh

# Download EfficientLoFTR model weights
python scripts/download_models.py --models outdoor indoor

# For Windows users: manually download datasets from official sources
# TUM: https://vision.in.tum.de/data/datasets/rgbd-dataset
# Replica: https://github.com/facebookresearch/Replica-Dataset
```

## System Requirements and Performance

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **Memory**: 8GB+ RAM, 6GB+ GPU memory for optimal performance
- **Camera**: USB 3.0 dual cameras for real-time operation
- **Performance Targets**: 20+ FPS real-time processing

### Common Configuration Adjustments
```yaml
# High performance settings
performance_targets:
  target_fps: 30
  max_memory_gb: 16
  max_gpu_memory_gb: 8

# Memory optimization
performance_targets:
  target_fps: 15
  max_gpu_memory_gb: 4
```

## Troubleshooting

### CUDA Extensions Build Issues
```bash
# Ensure CUDA toolkit is properly installed
nvcc --version

# Rebuild extensions
cd thirdparty/MonoGS
pip uninstall simple-knn diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
```

### EfficientLoFTR Model Loading
- The system gracefully falls back to OpenCV features if EfficientLoFTR fails to load
- Check model weights are present in `thirdparty/EfficientLoFTR/weights/`
- Use `python scripts/download_models.py` to download missing weights

### Camera Issues
```bash
# Test camera access
python diagnose_cameras.py

# Test camera calibration
python camera_calibration.py

# Try different device IDs in config
input:
  camera:
    left_device: 0
    right_device: 1

# For Windows camera backend issues
python fix_camera_backend.py
```

## Project Status

This is an active research project combining multiple SLAM approaches. The main hybrid system integrates:

1. **Feature Matching**: EfficientLoFTR for robust correspondences
2. **Geometric Constraints**: OpenCV PnP for pose estimation
3. **Backend Optimization**: MonoGS Gaussian splatting for scene reconstruction
4. **Real-time Processing**: Dual-camera stereo vision system

Key files to understand the system:
- `run_hybrid_slam.py`: Main entry point
- `hybrid_slam/core/slam_system.py`: System orchestrator  
- `hybrid_slam/frontend/hybrid_frontend.py`: Frontend processing
- `hybrid_slam/gui/`: Qt-based GUI components
- `run_qt_slam.py`: GUI version entry point
- Configuration files in `configs/` for different setups
- `diagnose_cameras.py`: Camera diagnostics utility
- `scripts/`: Various utility scripts

## Additional Utilities

### Camera Setup and Calibration
```bash
# Camera diagnostics
python diagnose_cameras.py

# Camera calibration
python camera_calibration.py

# Test dual camera setup
python cam.py
```

### GUI Version
```bash
# Run Qt-based GUI
python run_qt_slam.py

# Test Qt GUI components
python test_qt_slam.py
```

### Performance Analysis
```bash
# Profile system performance
python tools/profile_performance.py

# Evaluate performance metrics
python scripts/evaluate_performance.py
```