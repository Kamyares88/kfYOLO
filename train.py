#!/usr/bin/env python3
"""
YOLO Training Script for Biomedical Object Detection
Supports both object detection and instance segmentation
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔍 Setting up environment...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Check ultralytics version
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed. Please install with: pip install ultralytics")
        sys.exit(1)

def validate_dataset_structure(dataset_path):
    """Validate that the dataset follows YOLO format"""
    print(f"🔍 Validating dataset structure at {dataset_path}...")
    
    required_dirs = ['images/train', 'images/val', 'images/test']
    required_files = ['labels/train', 'labels/val', 'labels/test']
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return False
        else:
            # Count images
            image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            print(f"✅ {dir_path}: {len(image_files)} images")
    
    for file_path in required_files:
        full_path = os.path.join(dataset_path, file_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing labels directory: {full_path}")
            return False
    
    print("✅ Dataset structure validation passed")
    return True

def download_pretrained_model(model_name="yolov9-c.pt"):
    """Download pre-trained YOLO model if not exists"""
    print(f"🔍 Checking for pre-trained model: {model_name}")
    
    if not os.path.exists(model_name):
        print(f"📥 Downloading {model_name}...")
        model = YOLO(model_name)
        print(f"✅ Model downloaded successfully")
    else:
        print(f"✅ Model {model_name} already exists")
    
    return model_name

def train_model(config_path, dataset_path, epochs=None, batch_size=None, img_size=None):
    """Train YOLO model with custom configuration"""
    print("🚀 Starting YOLO training...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if epochs:
        config['epochs'] = epochs
    if batch_size:
        config['batch_size'] = batch_size
    if img_size:
        config['imgsz'] = img_size
    
    print(f"📊 Training Configuration:")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Batch Size: {config['batch_size']}")
    print(f"   - Image Size: {config['imgsz']}")
    print(f"   - Device: {config['device']}")
    
    # Initialize model
    model = YOLO(config['model'])
    
    # Start training
    try:
        results = model.train(
            data=dataset_path,
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch_size'],
            device=config['device'],
            project=config['project'],
            name=config['name'],
            exist_ok=True,
            pretrained=config['pretrained'],
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            cache=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save_period=config['save_period']
        )
        
        print("✅ Training completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for biomedical object detection')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--img-size', type=int, help='Input image size')
    parser.add_argument('--validate-only', action='store_true', help='Only validate dataset structure')
    
    args = parser.parse_args()
    
    print("🏥 YOLO Biomedical Object Detection Training")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Validate dataset structure
    if not validate_dataset_structure(args.dataset):
        print("❌ Dataset validation failed. Please check your dataset structure.")
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Dataset validation completed. Exiting.")
        return
    
    # Download pre-trained model
    model_path = download_pretrained_model()
    
    # Start training
    results = train_model(
        config_path=args.config,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    if results:
        print("\n🎉 Training Summary:")
        print(f"   - Best model saved at: {results.save_dir}")
        print(f"   - Training completed in: {results.epochs} epochs")
        print(f"   - Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   - Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        if 'seg' in results.results_dict:
            print(f"   - Segmentation mAP@0.5: {results.results_dict.get('metrics/mAP50(S)', 'N/A')}")
            print(f"   - Segmentation mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(S)', 'N/A')}")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
