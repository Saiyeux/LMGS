"""
Hybrid SLAM Package Setup
EfficientLoFTR + OpenCV PnP + MonoGS Integration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
exec(open('hybrid_slam/version.py').read())

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    if Path(filename).exists():
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="hybrid-slam",
    version=__version__,
    author="LMGS Team",
    author_email="team@lmgs.dev",
    description="Hybrid SLAM system integrating EfficientLoFTR, OpenCV PnP, and MonoGS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(exclude=["tests*", "examples*"]),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements("requirements.txt"),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "visualization": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "hybrid-slam=scripts.run_hybrid_slam:main",
            "hybrid-slam-eval=scripts.evaluate_performance:main",
        ],
    },
    
    package_data={
        "hybrid_slam": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)