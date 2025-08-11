# 安装指南

本文档详细说明了Hybrid SLAM系统的安装步骤。

## 系统要求

### 硬件要求

**推荐配置**:
- GPU: RTX 4090 (24GB VRAM)
- CPU: Intel i7-12700K 或同等性能
- RAM: 32GB DDR4
- 存储: 1TB NVMe SSD

**最低配置**:
- GPU: GTX 1080 Ti (11GB VRAM)  
- CPU: Intel i5-10400 或同等性能
- RAM: 16GB DDR4
- 存储: 500GB SSD

### 软件要求

- Python 3.8+
- CUDA 11.8+
- Git

## 安装步骤

### 1. 克隆项目

```bash
git clone --recursive https://github.com/lmgs-team/hybrid-slam.git
cd hybrid-slam
```

### 2. 创建环境

```bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate LMGS

# 或使用pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. 安装CUDA扩展

```bash
# 安装MonoGS的CUDA扩展
cd thirdparty/MonoGS
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
cd ../..
```

### 4. 安装主包

```bash
# 开发模式安装
pip install -e .
```

### 5. 下载预训练模型

```bash
# 下载EfficientLoFTR模型
python scripts/download_models.py --models outdoor indoor
```

### 6. 验证安装

```bash
# 运行环境检查
python scripts/setup_environment.py
```

## 故障排除

### 常见问题

1. **CUDA扩展编译失败**
   ```bash
   # 设置CUDA环境变量
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   
   # 重新编译
   pip uninstall diff-gaussian-rasterization simple-knn -y
   pip install ./submodules/simple-knn ./submodules/diff-gaussian-rasterization --force-reinstall
   ```

2. **内存不足**
   - 降低batch size
   - 使用fp16精度
   - 减少特征点数量

3. **模型加载失败**
   ```bash
   # 重新下载模型
   python scripts/download_models.py --models all
   ```

## 验证安装

运行测试以确保安装正确：

```bash
# 单元测试
pytest tests/unit/

# 集成测试  
pytest tests/integration/

# 完整测试
pytest
```

如果所有测试通过，说明安装成功！