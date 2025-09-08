#!/usr/bin/env python3
"""
Setup script for YOLO Biomedical Object Detection
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   PyTorch version: {torch.__version__}")
        else:
            print("⚠️  CUDA not available - training will use CPU")
            print(f"   PyTorch version: {torch.__version__}")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "dataset",
        "runs",
        "results",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "ultralytics",
        "opencv-python",
        "PIL",
        "numpy",
        "matplotlib",
        "seaborn",
        "yaml",
        "tqdm"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
                print(f"✅ {package}")
            elif package == "opencv-python":
                import cv2
                print(f"✅ {package}")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All packages imported successfully")
    return True

def main():
    """Main setup function"""
    print("🏥 YOLO Biomedical Object Detection - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check the error messages above.")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("Some packages failed to import. Please check the installation.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your biomedical dataset")
    print("2. Run: python data_preparation.py --input <your_data> --output ./dataset")
    print("3. Run: python train.py --dataset ./dataset/dataset.yaml")
    print("4. Run: python example_usage.py to see examples")
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()
