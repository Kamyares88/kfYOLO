#!/usr/bin/env python3
"""
Example Usage Script for YOLO Biomedical Object Detection
Demonstrates how to use the pipeline programmatically
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import BiomedicalDataPreparation
from train import train_model
from inference import BiomedicalYOLOInference
from evaluate import BiomedicalYOLOEvaluation

def example_data_preparation():
    """Example of data preparation workflow"""
    print("🔧 Example: Data Preparation")
    print("=" * 40)
    
    # Initialize data preparation
    prep = BiomedicalDataPreparation(
        input_dir="./example_data",
        output_dir="./dataset",
        class_mapping={
            "cell": 0,
            "nucleus": 1,
            "mitochondria": 2
        }
    )
    
    # Detect annotation format
    format_type = prep.detect_annotation_format()
    if format_type:
        print(f"Detected format: {format_type}")
        
        # Convert annotations
        prep.convert_annotations(
            format_type=format_type,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1
        )
        
        # Create dataset.yaml
        prep.create_dataset_yaml()
    else:
        print("No annotation format detected")

def example_training():
    """Example of training workflow"""
    print("\n🚀 Example: Model Training")
    print("=" * 40)
    
    # Example training configuration
    config = {
        'epochs': 50,
        'batch_size': 16,
        'imgsz': 640,
        'device': 0,
        'project': 'biomedical_yolo',
        'name': 'example_exp'
    }
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Note: This would start actual training
    # Uncomment to run:
    # results = train_model(
    #     config_path='config.yaml',
    #     dataset_path='./dataset/dataset.yaml',
    #     epochs=config['epochs'],
    #     batch_size=config['batch_size'],
    #     img_size=config['imgsz']
    # )

def example_inference():
    """Example of inference workflow"""
    print("\n🔍 Example: Model Inference")
    print("=" * 40)
    
    # Initialize inference (assuming model exists)
    model_path = "./runs/train/biomedical_yolo/example_exp/weights/best.pt"
    
    if os.path.exists(model_path):
        inference = BiomedicalYOLOInference(
            model_path=model_path,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        # Get model info
        model_info = inference.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            if key != 'class_names':
                print(f"  {key}: {value}")
        
        # Example single image prediction
        test_image = "./test_image.jpg"
        if os.path.exists(test_image):
            print(f"\nRunning inference on: {test_image}")
            # result = inference.predict_single_image(test_image)
            # print(f"Detections: {result['summary']['total_detections']}")
        else:
            print(f"Test image not found: {test_image}")
    else:
        print(f"Model not found: {model_path}")
        print("Please train a model first or update the path")

def example_evaluation():
    """Example of evaluation workflow"""
    print("\n📊 Example: Model Evaluation")
    print("=" * 40)
    
    # Initialize evaluation (assuming model exists)
    model_path = "./runs/train/biomedical_yolo/example_exp/weights/best.pt"
    dataset_path = "./dataset/dataset.yaml"
    
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        evaluator = BiomedicalYOLOEvaluation(
            model_path=model_path,
            dataset_path=dataset_path,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        print("Running evaluation...")
        # evaluator.evaluate_model("./evaluation_results")
        print("Evaluation completed")
    else:
        print("Model or dataset not found")
        print("Please ensure both exist before evaluation")

def create_example_dataset():
    """Create a minimal example dataset structure"""
    print("\n📁 Creating Example Dataset Structure")
    print("=" * 40)
    
    # Create example data directory
    example_dir = Path("./example_data")
    example_dir.mkdir(exist_ok=True)
    
    # Create example annotation files
    example_annotations = {
        "image1.xml": """<?xml version="1.0"?>
<annotation>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>cell</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
  <object>
    <name>nucleus</name>
    <bndbox>
      <xmin>120</xmin>
      <ymin>170</ymin>
      <xmax>180</xmax>
      <ymax>230</ymax>
    </bndbox>
  </object>
</annotation>""",
        
        "image2.xml": """<?xml version="1.0"?>
<annotation>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>mitochondria</name>
    <bndbox>
      <xmin>300</xmin>
      <ymin>200</ymin>
      <xmax>350</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>"""
    }
    
    # Create example annotation files
    for filename, content in example_annotations.items():
        file_path = example_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created: {filename}")
    
    # Create dummy image files (empty files for demonstration)
    for i in range(1, 3):
        img_path = example_dir / f"image{i}.jpg"
        img_path.touch()  # Create empty file
        print(f"Created: image{i}.jpg")
    
    print(f"\nExample dataset created in: {example_dir.absolute()}")
    print("Note: Image files are empty - replace with actual images for real training")

def main():
    """Main example workflow"""
    print("🏥 YOLO Biomedical Object Detection - Example Usage")
    print("=" * 60)
    
    # Check if example data exists, create if not
    if not os.path.exists("./example_data"):
        create_example_dataset()
    
    # Run examples
    example_data_preparation()
    example_training()
    example_inference()
    example_evaluation()
    
    print("\n🎉 Example workflow completed!")
    print("\nNext steps:")
    print("1. Replace example images with real biomedical data")
    print("2. Adjust class names in data_preparation.py")
    print("3. Run: python data_preparation.py --input ./example_data --output ./dataset")
    print("4. Run: python train.py --dataset ./dataset/dataset.yaml")
    print("5. Run: python inference.py --model <trained_model> --input <test_image>")

if __name__ == "__main__":
    main()
