#!/usr/bin/env python3
"""
测试EfficientLoFTR匹配器
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_loftr_matcher():
    """测试EfficientLoFTR匹配器"""
    try:
        print("=== 测试EfficientLoFTR匹配器 ===")
        
        # 导入匹配器
        from hybrid_slam.matchers.loftr_matcher import EfficientLoFTRMatcher
        
        # 创建配置
        config = {
            'device': 'cpu',  # 使用CPU避免CUDA问题
            'resize_to': [640, 480],
            'match_threshold': 0.2,
            'max_keypoints': 1000,
            'model_path': 'thirdparty/EfficientLoFTR/weights/outdoor_ds.ckpt'
        }
        
        print(f"配置: {config}")
        
        # 创建匹配器
        print("创建匹配器...")
        matcher = EfficientLoFTRMatcher(config)
        
        # 检查模型是否加载成功
        if matcher.model is None:
            print("ERROR: 模型未加载成功")
            return False
        else:
            print("OK: 模型加载成功")
        
        # 创建测试图像
        print("创建测试图像...")
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()  # 相同图像应该有很多匹配
        
        # 添加一些噪声使图像稍有不同
        noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        print(f"图像1尺寸: {img1.shape}")
        print(f"图像2尺寸: {img2.shape}")
        
        # 执行匹配
        print("执行特征匹配...")
        matches, confidence = matcher.match_pair(img1, img2)
        
        if matches is None:
            print("WARNING: 没有找到匹配")
            return False
        
        print(f"找到 {len(matches)} 个匹配点")
        print(f"平均置信度: {confidence:.3f}")
        
        # 验证匹配结果
        if len(matches) > 0:
            print(f"前5个匹配点:")
            for i, (pt1, pt2) in enumerate(matches[:5]):
                print(f"  匹配{i+1}: ({pt1[0]:.1f}, {pt1[1]:.1f}) -> ({pt2[0]:.1f}, {pt2[1]:.1f})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 匹配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_fallback():
    """测试OpenCV后备匹配"""
    try:
        print("\n=== 测试OpenCV后备匹配 ===")
        
        # 创建测试图像
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        # 添加一些噪声
        noise = np.random.randint(-30, 30, img2.shape, dtype=np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 直接测试OpenCV匹配
        from hybrid_slam.core.ai_processing_pipeline import AIProcessingPipeline
        
        # 创建禁用LoFTR的配置
        config = {
            'enable_loftr': False,  # 强制使用OpenCV
            'enable_pnp': True,
            'confidence_threshold': 0.5
        }
        
        pipeline = AIProcessingPipeline(config)
        
        # 测试OpenCV匹配
        matches, confidence = pipeline._opencv_feature_matching(img1, img2)
        
        if matches is None:
            print("WARNING: OpenCV匹配也失败了")
            return False
        
        print(f"OpenCV找到 {len(matches)} 个匹配点")
        print(f"平均置信度: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: OpenCV测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始匹配器诊断测试...")
    
    # 测试LoFTR匹配器
    loftr_ok = test_loftr_matcher()
    
    # 测试OpenCV后备
    opencv_ok = test_opencv_fallback()
    
    print(f"\n=== 测试总结 ===")
    print(f"EfficientLoFTR匹配器: {'OK' if loftr_ok else 'FAILED'}")
    print(f"OpenCV后备匹配: {'OK' if opencv_ok else 'FAILED'}")
    
    if not loftr_ok and not opencv_ok:
        print("ERROR: 所有匹配器都失败了")
        sys.exit(1)
    elif not loftr_ok:
        print("WARNING: EfficientLoFTR失败，但OpenCV可用")
        sys.exit(0)
    else:
        print("OK: 至少一个匹配器工作正常")
        sys.exit(0)