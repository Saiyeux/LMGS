@echo off
chcp 65001 >nul
echo ========================================
echo    Dual Camera 3D Reconstruction System
echo ========================================
echo.

echo Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Select startup mode:
echo [1] Basic 3D Reconstruction (Recommended)
echo [2] High Performance Mode (No Visualization)  
echo [3] Qt GUI Interface
echo [4] Test Demo
echo.

set /p choice="Please choose (1-4): "

if "%choice%"=="1" (
    echo Starting basic 3D reconstruction mode...
    python run_dual_camera_3d.py --left-cam 0 --right-cam 1
) else if "%choice%"=="2" (
    echo Starting high performance mode...
    python run_dual_camera_3d.py --no-vis --save-dir performance_3d
) else if "%choice%"=="3" (
    echo Starting Qt GUI...
    python run_qt_slam.py --left-cam 0 --right-cam 1
) else if "%choice%"=="4" (
    echo Starting test demo...
    python test_3d_reconstruction_demo.py
) else (
    echo Invalid choice, starting default mode...
    python run_dual_camera_3d.py
)

echo.
echo ========================================
echo       3D Reconstruction Complete
echo ========================================
pause