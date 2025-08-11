@echo off
echo Creating LMGS unified environment...

REM 创建conda环境
conda env create -f environment.yml

echo.
echo Environment created! Now activating and installing additional dependencies...

REM 激活环境
call conda activate LMGS

REM 安装MonoGS的自定义子模块
echo Installing MonoGS submodules...
cd thirdparty\MonoGS
pip install .\submodules\simple-knn
pip install .\submodules\diff-gaussian-rasterization

REM 返回根目录
cd ..\..

echo.
echo Setup complete! To activate the environment, run:
echo conda activate LMGS
echo.
echo Then you can run:
echo - MonoGS: python thirdparty/MonoGS/slam.py --config thirdparty/MonoGS/configs/mono/tum/fr3_office.yaml
echo - EfficientLoFTR: python thirdparty/EfficientLoFTR/train.py [args]
pause