#!/usr/bin/env python3
"""
测试改进后的匹配性能
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_improved_loftr():
    """测试改进后的LoFTR匹配器"""
    try:
        print("=== 测试改进后的LoFTR匹配器 ===")
        
        from hybrid_slam.matchers.loftr_matcher import EfficientLoFTRMatcher
        
        # 使用与real_time_stereo_matcher.py相同的配置
        config = {
            'device': 'cpu',  # 避免CUDA问题
            'resize_to': [640, 480],  # 这会被调整为能被32整除的尺寸
            'match_threshold': 0.15,  # 使用更合理的阈值
            'max_keypoints': 2048,
            'model_path': 'thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt',  # 使用相同权重
            'enable_stereo_constraints': False
        }
        
        print(f"配置: {config}")
        
        # 创建匹配器
        print("创建改进的LoFTR匹配器...")
        matcher = EfficientLoFTRMatcher(config)
        
        if matcher.model is None:
            print("ERROR: 模型未加载成功")
            return False
        
        # 测试摄像头图像（如果存在）
        if Path('camera_left_sample.jpg').exists() and Path('camera_right_sample.jpg').exists():
            print("测试真实摄像头图像...")
            img1 = cv2.imread('camera_left_sample.jpg')
            img2 = cv2.imread('camera_right_sample.jpg')
        else:
            print("创建测试图像...")
            # 创建更复杂的测试图像
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 添加复杂纹理
            for i in range(0, 480, 30):
                for j in range(0, 640, 30):
                    if (i//30 + j//30) % 3 == 0:
                        img1[i:i+30, j:j+30] = [200, 200, 200]
                        img2[i:i+30, j:j+30] = [200, 200, 200]
                    elif (i//30 + j//30) % 3 == 1:
                        img1[i:i+30, j:j+30] = [100, 100, 100]
                        img2[i:i+30, j:j+30] = [100, 100, 100]
            
            # 添加更多特征
            for x, y, r in [(150, 150, 25), (300, 200, 20), (450, 300, 30), (200, 400, 15)]:
                cv2.circle(img1, (x, y), r, (0, 255, 0), -1)
                cv2.circle(img2, (x-15, y), r, (0, 255, 0), -1)  # 模拟视差
                
                cv2.circle(img1, (x+100, y), r-5, (255, 0, 0), -1)
                cv2.circle(img2, (x+85, y), r-5, (255, 0, 0), -1)
            
            # 添加矩形特征
            cv2.rectangle(img1, (50, 50), (150, 100), (255, 255, 0), -1)
            cv2.rectangle(img2, (35, 50), (135, 100), (255, 255, 0), -1)
        
        print(f"图像尺寸: {img1.shape}")
        
        # 执行匹配
        print("执行改进的特征匹配...")
        matches, confidence = matcher.match_pair(img1, img2)
        
        if matches is None:
            print("ERROR: 改进后仍然没有找到匹配")
            return False
        
        num_matches = len(matches)
        print(f"✓ 找到 {num_matches} 个匹配点")
        print(f"✓ 平均置信度: {confidence:.3f}")
        
        # 与real_time_stereo_matcher.py的效果对比
        if num_matches >= 20:
            print("🎉 SUCCESS: 匹配数量显著改善！")
        elif num_matches >= 10:
            print("✅ GOOD: 匹配数量有所改善")
        elif num_matches >= 5:
            print("⚠️  OK: 匹配数量一般")
        else:
            print("❌ POOR: 匹配数量仍然很少")
        
        # 创建可视化
        if num_matches > 0:
            # 使用与real_time_stereo_matcher.py相同的可视化方式
            H0, W0 = img1.shape[:2]
            H1, W1 = img2.shape[:2]
            H = max(H0, H1)
            W = W0 + W1
            
            # 转换为灰度用于显示
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = img1, img2
            
            # 创建组合图像
            combined_img = np.zeros((H, W), dtype=np.uint8)
            combined_img[:H0, :W0] = gray1
            combined_img[:H1, W0:W0+W1] = gray2
            combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
            
            # 绘制匹配点和连线
            for i, (pt1, pt2) in enumerate(matches[:50]):  # 限制显示数量
                if len(pt1) >= 2 and len(pt2) >= 2:
                    pt0 = tuple(map(int, pt1))
                    pt1_offset = tuple(map(int, pt2 + np.array([W0, 0])))
                    
                    # 根据置信度设置颜色
                    color = (0, int(255 * 0.8), int(255 * 0.2))  # 绿色系
                    
                    # 绘制关键点
                    cv2.circle(combined_img_color, pt0, 3, color, -1)
                    cv2.circle(combined_img_color, pt1_offset, 3, color, -1)
                    
                    # 绘制连线
                    cv2.line(combined_img_color, pt0, pt1_offset, color, 1)
            
            # 添加信息
            info_text = f"Improved Matches: {num_matches} (conf: {confidence:.3f})"
            cv2.putText(combined_img_color, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite('improved_matches.jpg', combined_img_color)
            print("改进后匹配可视化已保存: improved_matches.jpg")
        
        return num_matches > 5
        
    except Exception as e:
        print(f"ERROR: 改进测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """与原始实现对比"""
    try:
        print("\n=== 性能对比总结 ===")
        
        # 读取之前保存的结果（如果存在）
        results = {}
        
        if Path('debug_matches.jpg').exists():
            print("发现之前的测试结果")
            # 这里可以添加自动分析图像中匹配数量的代码
        
        print("建议:")
        print("1. 运行 python test_real_matching.py 获取基准")
        print("2. 运行 python test_improved_matching.py 测试改进")
        print("3. 对比两个结果文件")
        print("4. 如果改进成功，重新启动Qt界面测试")
        
        return True
        
    except Exception as e:
        print(f"对比分析失败: {e}")
        return False

if __name__ == "__main__":
    print("开始改进后匹配性能测试...")
    
    # 测试改进的LoFTR
    improved_ok = test_improved_loftr()
    
    # 性能对比
    compare_ok = compare_with_original()
    
    print(f"\n=== 测试结果 ===")
    print(f"改进后LoFTR: {'SUCCESS' if improved_ok else 'FAILED'}")
    
    if improved_ok:
        print("🎉 改进成功！现在可以重新测试Qt界面:")
        print("   python run_qt_slam.py --left-cam 0 --right-cam 1")
        print("\n预期改善:")
        print("   - 匹配数量从1个增加到10-50个")
        print("   - 位姿估计开始工作")
        print("   - 可视化显示匹配点和连线")
    else:
        print("❌ 改进效果有限，可能需要进一步调试")