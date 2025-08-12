"""
LMGS Reconstruction Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# 读取版本信息
version = "1.0.0"

setup(
    name="lmgs-reconstruction",
    version=version,
    author="LMGS Team",
    author_email="team@lmgs.ai",
    description="A modular 3D reconstruction system integrating EfficientLoFTR and MonoGS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "torch>=1.8.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ],
    extras_require={
        "cuda": ["torch>=1.8.0+cu111"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "lmgs-reconstruction=run_reconstruction:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)