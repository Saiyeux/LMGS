#!/usr/bin/env python3
"""
测试真实摄像头图像的匹配
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_realistic_test_images():
    """创建更真实的测试图像对"""
    # 创建带纹理的图像
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加棋盘格图案
    for i in range(0, 480, 60):
        for j in range(0, 640, 60):
            if (i//60 + j//60) % 2 == 0:
                img1[i:i+60, j:j+60] = [200, 200, 200]
            else:
                img1[i:i+60, j:j+60] = [50, 50, 50]
    
    # 添加一些圆形特征
    cv2.circle(img1, (160, 120), 30, (0, 255, 0), -1)
    cv2.circle(img1, (480, 120), 30, (255, 0, 0), -1)
    cv2.circle(img1, (320, 360), 40, (0, 0, 255), -1)
    
    # 添加一些矩形
    cv2.rectangle(img1, (100, 200), (200, 280), (255, 255, 0), -1)
    cv2.rectangle(img1, (400, 300), (540, 400), (255, 0, 255), -1)
    
    # 创建稍微不同的第二幅图像（模拟立体视觉）
    img2 = img1.copy()
    
    # 添加轻微位移（模拟视差）
    M = np.float32([[1, 0, -20], [0, 1, 0]])  # 向左位移20像素
    img2 = cv2.warpAffine(img2, M, (640, 480))
    
    # 添加轻微噪声
    noise = np.random.normal(0, 5, img2.shape).astype(np.uint8)
    img2 = cv2.add(img2, noise)
    
    return img1, img2

def test_matching_with_realistic_images():
    """使用真实纹理测试匹配"""
    try:
        print("=== 测试真实纹理图像匹配 ===")
        
        # 创建真实纹理图像
        img1, img2 = create_realistic_test_images()
        
        print(f"左图尺寸: {img1.shape}")
        print(f"右图尺寸: {img2.shape}")
        
        # 保存图像用于调试
        cv2.imwrite('debug_left.jpg', img1)
        cv2.imwrite('debug_right.jpg', img2)
        print("调试图像已保存: debug_left.jpg, debug_right.jpg")
        
        # 测试EfficientLoFTR匹配器
        from hybrid_slam.matchers.loftr_matcher import EfficientLoFTRMatcher
        
        config = {
            'device': 'cpu',
            'resize_to': [640, 480],
            'match_threshold': 0.2,
            'max_keypoints': 1000,
            'model_path': 'thirdparty/EfficientLoFTR/weights/outdoor_ds.ckpt',
            'enable_stereo_constraints': True,
            'max_disparity': 100,
            'stereo_y_tolerance': 5.0
        }
        
        print("创建LoFTR匹配器...")
        matcher = EfficientLoFTRMatcher(config)
        
        if matcher.model is None:
            print("ERROR: 模型未加载")
            return False
        
        print("执行立体匹配...")
        result = matcher.match_stereo_pair(img1, img2)
        
        if result is None:
            print("ERROR: 立体匹配失败")
            return False
        
        num_matches = result['num_matches']
        print(f"找到 {num_matches} 个立体匹配点")
        
        if num_matches == 0:
            print("WARNING: 没有找到匹配点")
            
            # 尝试不应用立体约束
            print("尝试不使用立体约束...")
            result2 = matcher.match_frames(img1, img2, stereo_matching=False)
            if result2:
                print(f"无立体约束匹配数: {result2['num_matches']}")
                
                # 可视化匹配结果
                if result2['num_matches'] > 0:
                    vis = matcher.visualize_matches(img1, img2, result2, max_matches=50)
                    cv2.imwrite('debug_matches.jpg', vis)
                    print("匹配可视化已保存: debug_matches.jpg")
        else:
            # 显示匹配质量
            quality = matcher.get_match_quality(result)
            print(f"匹配质量: {quality}")
            
            # 可视化匹配结果
            vis = matcher.visualize_matches(img1, img2, result, max_matches=50)
            cv2.imwrite('debug_stereo_matches.jpg', vis)
            print("立体匹配可视化已保存: debug_stereo_matches.jpg")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 真实纹理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_on_realistic_images():
    """在真实纹理上测试OpenCV匹配"""
    try:
        print("\n=== 测试OpenCV在真实纹理上的表现 ===")
        
        # 创建图像
        img1, img2 = create_realistic_test_images()
        
        # 测试OpenCV匹配
        from hybrid_slam.core.ai_processing_pipeline import AIProcessingPipeline
        
        pipeline = AIProcessingPipeline({'enable_loftr': False})
        matches, confidence = pipeline._opencv_feature_matching(img1, img2)
        
        if matches is None:
            print("ERROR: OpenCV匹配失败")
            return False
        
        print(f"OpenCV找到 {len(matches)} 个匹配点")
        print(f"置信度: {confidence:.3f}")
        
        # 可视化OpenCV匹配结果
        if len(matches) > 0:
            # 创建可视化
            combined = cv2.hconcat([img1, img2])
            
            # 绘制匹配点
            for i, (pt1, pt2) in enumerate(matches[:50]):  # 只显示前50个
                if len(pt1) >= 2 and len(pt2) >= 2:
                    pt1_int = (int(pt1[0]), int(pt1[1]))
                    pt2_int = (int(pt2[0] + 640), int(pt2[1]))  # 右图偏移
                    
                    cv2.circle(combined, pt1_int, 3, (0, 255, 0), -1)
                    cv2.circle(combined, pt2_int, 3, (0, 255, 0), -1)
                    cv2.line(combined, pt1_int, pt2_int, (0, 255, 255), 1)
            
            cv2.imwrite('debug_opencv_matches.jpg', combined)
            print("OpenCV匹配可视化已保存: debug_opencv_matches.jpg")
        
        return True
        
    except Exception as e:
        print(f"ERROR: OpenCV真实纹理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始真实图像匹配测试...")
    
    # 测试LoFTR在真实纹理上的表现
    loftr_ok = test_matching_with_realistic_images()
    
    # 测试OpenCV在真实纹理上的表现
    opencv_ok = test_opencv_on_realistic_images()
    
    print(f"\n=== 真实纹理测试总结 ===")
    print(f"EfficientLoFTR: {'OK' if loftr_ok else 'FAILED'}")
    print(f"OpenCV: {'OK' if opencv_ok else 'FAILED'}")
    
    if loftr_ok or opencv_ok:
        print("至少一个匹配器在真实纹理上工作正常")
        print("如果Qt界面仍然没有匹配，可能是摄像头图像质量或配置问题")
    else:
        print("ERROR: 所有匹配器在真实纹理上都失败了")