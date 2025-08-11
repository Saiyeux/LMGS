#!/usr/bin/env python3
"""
摄像头回退和修复方案
为Qt SLAM应用提供摄像头访问的多种回退策略
"""

import cv2
import time
import json
from pathlib import Path

def detect_available_cameras():
    """检测可用的摄像头设备"""
    print("=== 检测可用摄像头设备 ===")
    available_cameras = []
    
    # 测试0-4的设备ID
    for device_id in range(5):
        print(f"测试摄像头 {device_id}...", end=" ")
        
        # 尝试不同的后端
        backends_to_try = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"), 
            (cv2.CAP_ANY, "Auto")
        ]
        
        device_info = None
        
        for backend_id, backend_name in backends_to_try:
            cap = cv2.VideoCapture(device_id, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    device_info = {
                        'device_id': device_id,
                        'backend_id': backend_id,
                        'backend_name': backend_name,
                        'resolution': frame.shape[:2][::-1],  # (width, height)
                        'channels': frame.shape[2] if len(frame.shape) == 3 else 1
                    }
                    cap.release()
                    print(f"[OK] {backend_name} - {device_info['resolution']}")
                    break
            cap.release()
        
        if device_info:
            available_cameras.append(device_info)
        else:
            print("[FAIL]")
    
    return available_cameras

def create_camera_config_recommendations(available_cameras):
    """基于可用摄像头创建配置建议"""
    print(f"\n=== 发现 {len(available_cameras)} 个可用摄像头 ===")
    
    if len(available_cameras) == 0:
        print("❌ 没有发现可用的摄像头设备")
        return None
    
    # 显示所有可用摄像头
    for i, cam in enumerate(available_cameras):
        print(f"{i+1}. 设备{cam['device_id']}: {cam['backend_name']} - {cam['resolution']}")
    
    recommendations = []
    
    # 推荐方案1: 如果有2个或以上摄像头
    if len(available_cameras) >= 2:
        rec1 = {
            'name': '双摄像头配置（推荐）',
            'left_device': available_cameras[0]['device_id'],
            'right_device': available_cameras[1]['device_id'], 
            'left_backend': available_cameras[0]['backend_id'],
            'right_backend': available_cameras[1]['backend_id'],
            'description': f"使用摄像头{available_cameras[0]['device_id']}和{available_cameras[1]['device_id']}"
        }
        recommendations.append(rec1)
    
    # 推荐方案2: 单摄像头 + 模拟数据
    if len(available_cameras) >= 1:
        rec2 = {
            'name': '单摄像头 + 模拟右摄像头',
            'left_device': available_cameras[0]['device_id'],
            'right_device': -1,  # 使用模拟数据
            'left_backend': available_cameras[0]['backend_id'],
            'right_backend': -1,
            'description': f"使用摄像头{available_cameras[0]['device_id']}作为左摄像头，右摄像头使用模拟数据"
        }
        recommendations.append(rec2)
    
    # 推荐方案3: 完全模拟
    rec3 = {
        'name': '完全模拟模式',
        'left_device': -1,
        'right_device': -1,
        'left_backend': -1,
        'right_backend': -1,
        'description': '使用模拟立体数据进行测试'
    }
    recommendations.append(rec3)
    
    return recommendations

def modify_qt_slam_for_fallback():
    """修改Qt SLAM应用以支持回退策略"""
    
    # 检测可用摄像头
    available_cameras = detect_available_cameras()
    recommendations = create_camera_config_recommendations(available_cameras)
    
    if not recommendations:
        print("无法创建摄像头配置建议")
        return False
    
    print(f"\n=== 配置建议 ===")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec['name']}")
        print(f"   {rec['description']}")
        if rec['left_device'] >= 0:
            print(f"   左摄像头: 设备{rec['left_device']} (后端:{rec['left_backend']})")
        if rec['right_device'] >= 0:
            print(f"   右摄像头: 设备{rec['right_device']} (后端:{rec['right_backend']})")
        print()
    
    # 选择最佳方案
    best_recommendation = recommendations[0]
    
    # 创建配置文件
    config = {
        'camera_config': {
            'left_device': best_recommendation['left_device'],
            'right_device': best_recommendation['right_device'],
            'left_backend': best_recommendation['left_backend'],
            'right_backend': best_recommendation['right_backend'],
            'fallback_mode': best_recommendation['right_device'] == -1
        },
        'available_cameras': available_cameras,
        'all_recommendations': recommendations
    }
    
    # 保存配置
    config_file = Path("camera_fallback_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 配置已保存到: {config_file}")
    print(f"[RECOMMEND] 推荐使用: {best_recommendation['name']}")
    
    # 生成修复说明
    print(f"\n=== 修复步骤 ===")
    print(f"1. 使用以下参数启动Qt SLAM:")
    
    if best_recommendation['left_device'] >= 0 and best_recommendation['right_device'] >= 0:
        print(f"   python run_qt_slam.py --left-cam {best_recommendation['left_device']} --right-cam {best_recommendation['right_device']}")
    elif best_recommendation['left_device'] >= 0:
        print(f"   python run_qt_slam.py --left-cam {best_recommendation['left_device']} --mock-right")
    else:
        print(f"   python run_qt_slam.py --mock")
    
    print(f"2. 或者修改VideoStreamManager类使用检测到的后端ID")
    print(f"3. 配置详情保存在: {config_file}")
    
    return True

def test_recommended_config():
    """测试推荐的配置"""
    config_file = Path("camera_fallback_config.json")
    if not config_file.exists():
        print("配置文件不存在，请先运行摄像头检测")
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    cam_config = config['camera_config']
    
    print("=== 测试推荐配置 ===")
    
    if cam_config['left_device'] >= 0:
        print(f"测试左摄像头 {cam_config['left_device']}...")
        left_cap = cv2.VideoCapture(cam_config['left_device'], cam_config['left_backend'])
        if left_cap.isOpened():
            ret, frame = left_cap.read()
            if ret:
                print(f"  [OK] 左摄像头正常 - {frame.shape}")
            else:
                print("  [FAIL] 左摄像头无法读取")
            left_cap.release()
        else:
            print("  [FAIL] 左摄像头无法打开")
    
    if cam_config['right_device'] >= 0:
        print(f"测试右摄像头 {cam_config['right_device']}...")
        right_cap = cv2.VideoCapture(cam_config['right_device'], cam_config['right_backend'])
        if right_cap.isOpened():
            ret, frame = right_cap.read()
            if ret:
                print(f"  [OK] 右摄像头正常 - {frame.shape}")
            else:
                print("  [FAIL] 右摄像头无法读取")
            right_cap.release()
        else:
            print("  [FAIL] 右摄像头无法打开")
    elif cam_config['fallback_mode']:
        print("  [INFO] 右摄像头将使用模拟数据")
    
    return True

def main():
    """主函数"""
    print("摄像头回退修复工具")
    print("=" * 50)
    
    # 检测并创建配置
    success = modify_qt_slam_for_fallback()
    
    if success:
        print("\n" + "=" * 50)
        
        # 测试配置
        test_recommended_config()
        
        print("\n=== 后续步骤 ===")
        print("1. 根据上述建议参数重新运行 Qt SLAM")
        print("2. 如果仍有问题，考虑:")
        print("   - 重新插拔USB摄像头")
        print("   - 关闭其他可能占用摄像头的应用") 
        print("   - 重启计算机")
        print("   - 使用完全模拟模式: python run_qt_slam.py --mock")
    
    return success

if __name__ == "__main__":
    main()