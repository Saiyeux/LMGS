#!/usr/bin/env python3
"""
模型下载脚本
下载预训练的EfficientLoFTR模型
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Download pretrained models')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['outdoor', 'indoor'],
                       choices=['outdoor', 'indoor', 'all'],
                       help='Models to download')
    parser.add_argument('--output-dir', type=str,
                       default='thirdparty/EfficientLoFTR/weights',
                       help='Output directory')
    
    return parser.parse_args()

class DownloadProgress:
    """下载进度条"""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()

def download_model(model_name: str, url: str, output_path: str):
    """下载模型文件"""
    print(f"Downloading {model_name} model...")
    
    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        return
    
    try:
        urllib.request.urlretrieve(url, output_path, DownloadProgress())
        print(f"Downloaded: {output_path}")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

def main():
    """主函数"""
    args = parse_args()
    
    print("="*50)
    print("EfficientLoFTR Model Downloader")
    print("="*50)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型URL配置 (这里使用示例URL，实际需要替换为真实地址)
    model_urls = {
        'outdoor': {
            'url': 'https://github.com/zju3dv/EfficientLoFTR/releases/download/v1.0/eloftr_outdoor.ckpt',
            'filename': 'eloftr_outdoor.ckpt'
        },
        'indoor': {
            'url': 'https://github.com/zju3dv/EfficientLoFTR/releases/download/v1.0/indoor_ds.ckpt', 
            'filename': 'indoor_ds.ckpt'
        }
    }
    
    # 处理all选项
    if 'all' in args.models:
        models_to_download = list(model_urls.keys())
    else:
        models_to_download = args.models
    
    # 下载模型
    for model_name in models_to_download:
        if model_name in model_urls:
            model_info = model_urls[model_name]
            output_path = output_dir / model_info['filename']
            
            download_model(model_name, model_info['url'], str(output_path))
        else:
            print(f"Unknown model: {model_name}")
    
    print("\nModel download completed!")
    
    # 显示已下载的模型
    print("\nAvailable models:")
    for file_path in output_dir.glob("*.ckpt"):
        file_size = file_path.stat().st_size / (1024*1024)  # MB
        print(f"  {file_path.name} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()