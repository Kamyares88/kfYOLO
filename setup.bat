@echo off
echo 🏥 YOLO Biomedical Object Detection - Windows Setup
echo ===================================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo Next steps:
echo 1. Prepare your biomedical dataset
echo 2. Run: python data_preparation.py --input ^<your_data^> --output ./dataset
echo 3. Run: python train.py --dataset ./dataset/dataset.yaml
echo 4. Run: python example_usage.py to see examples
echo.
echo For help, see README.md
echo.
pause
