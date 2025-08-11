#!/usr/bin/env python3
"""
测试运行脚本
支持不同类型的测试运行
"""

import sys
import subprocess
from pathlib import Path
import argparse

def run_tests(test_type='all', verbose=True, coverage=False):
    """运行测试"""
    
    # 基本pytest命令
    cmd = ['python', '-m', 'pytest']
    
    # 根据测试类型添加参数
    if test_type == 'unit':
        cmd.append('tests/unit/')
        print("Running unit tests...")
    elif test_type == 'integration':
        cmd.append('tests/integration/')
        print("Running integration tests...")
    elif test_type == 'all':
        cmd.append('tests/')
        print("Running all tests...")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # 添加额外选项
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=hybrid_slam', '--cov-report=html', '--cov-report=term-missing'])
    
    # 其他有用的选项
    cmd.extend([
        '--tb=short',
        '--color=yes',
        '--disable-warnings'
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def check_dependencies():
    """检查测试依赖"""
    try:
        import pytest
        import numpy
        import torch
        print("[OK] All test dependencies are available")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing test dependency: {e}")
        print("Please install test dependencies with:")
        print("pip install pytest numpy torch")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run Hybrid SLAM tests')
    parser.add_argument('--type', choices=['unit', 'integration', 'all'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check test dependencies only')
    
    args = parser.parse_args()
    
    print("Hybrid SLAM Test Runner")
    print("=" * 50)
    
    if args.check_deps:
        return check_dependencies()
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 运行测试
    success = run_tests(args.type, args.verbose, args.coverage)
    
    if success:
        print("\n[OK] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)