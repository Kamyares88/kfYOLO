#!/usr/bin/env python3
"""
Data Preparation Script for YOLO Biomedical Object Detection
Converts various annotation formats to YOLO format and creates proper directory structure
"""

import os
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
from PIL import Image
import yaml

class BiomedicalDataPreparation:
    def __init__(self, input_dir, output_dir, class_mapping=None):
        """
        Initialize data preparation
        
        Args:
            input_dir (str): Input directory containing images and annotations
            output_dir (str): Output directory for YOLO format
            class_mapping (dict): Mapping from original class names to YOLO class IDs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping or {}
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Supported annotation formats
        self.supported_formats = ['.xml', '.json', '.txt', '.csv']
        
    def _create_directory_structure(self):
        """Create YOLO directory structure"""
        directories = [
            'images/train',
            'images/val', 
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for dir_path in directories:
            full_path = self.output_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {full_path}")
    
    def detect_annotation_format(self):
        """Detect the annotation format used in the input directory"""
        print("🔍 Detecting annotation format...")
        
        # Look for annotation files
        annotation_files = []
        for ext in self.supported_formats:
            annotation_files.extend(self.input_dir.glob(f"*{ext}"))
        
        if not annotation_files:
            print("❌ No annotation files found")
            return None
        
        # Determine format based on file extension
        format_counts = {}
        for file in annotation_files:
            ext = file.suffix.lower()
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Find most common format
        most_common = max(format_counts.items(), key=lambda x: x[1])
        print(f"✅ Detected format: {most_common[0]} ({most_common[1]} files)")
        
        return most_common[0]
    
    def convert_annotations(self, format_type, train_split=0.7, val_split=0.2, test_split=0.1):
        """
        Convert annotations to YOLO format
        
        Args:
            format_type (str): Annotation format (xml, json, txt, csv)
            train_split (float): Training data split ratio
            val_split (float): Validation data split ratio
            test_split (float): Test data split ratio
        """
        print(f"🔄 Converting {format_type} annotations to YOLO format...")
        
        # Get all image files
        image_files = self._get_image_files()
        annotation_files = self._get_annotation_files(format_type)
        
        if not image_files:
            print("❌ No image files found")
            return
        
        if not annotation_files:
            print("❌ No annotation files found")
            return
        
        # Create train/val/test splits
        splits = self._create_data_splits(image_files, train_split, val_split, test_split)
        
        # Process each split
        for split_name, files in splits.items():
            print(f"\n📁 Processing {split_name} split ({len(files)} files)...")
            
            for img_file in files:
                # Find corresponding annotation file
                ann_file = self._find_annotation_file(img_file, format_type)
                
                if ann_file:
                    # Convert annotation
                    yolo_annotations = self._convert_single_annotation(
                        ann_file, format_type, img_file
                    )
                    
                    if yolo_annotations:
                        # Save image
                        img_dest = self.output_dir / f"images/{split_name}" / img_file.name
                        shutil.copy2(img_file, img_dest)
                        
                        # Save labels
                        label_dest = self.output_dir / f"labels/{split_name}" / f"{img_file.stem}.txt"
                        self._save_yolo_labels(yolo_annotations, label_dest)
        
        print(f"\n✅ Data conversion completed!")
        print(f"   - Training: {len(splits['train'])} images")
        print(f"   - Validation: {len(splits['val'])} images") 
        print(f"   - Test: {len(splits['test'])} images")
    
    def _get_image_files(self):
        """Get all image files from input directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _get_annotation_files(self, format_type):
        """Get all annotation files of specified format"""
        return sorted(self.input_dir.glob(f"*{format_type}"))
    
    def _create_data_splits(self, files, train_split, val_split, test_split):
        """Create train/val/test splits"""
        np.random.seed(42)  # For reproducible splits
        shuffled_files = np.random.permutation(files)
        
        n_files = len(shuffled_files)
        n_train = int(n_files * train_split)
        n_val = int(n_files * val_split)
        
        splits = {
            'train': shuffled_files[:n_train],
            'val': shuffled_files[n_train:n_train + n_val],
            'test': shuffled_files[n_train + n_val:]
        }
        
        return splits
    
    def _find_annotation_file(self, image_file, format_type):
        """Find corresponding annotation file for an image"""
        # Try different naming patterns
        possible_names = [
            image_file.stem + format_type,
            image_file.stem + format_type.upper(),
            image_file.stem + '_annotations' + format_type,
            image_file.stem + '_labels' + format_type
        ]
        
        for name in possible_names:
            ann_file = self.input_dir / name
            if ann_file.exists():
                return ann_file
        
        return None
    
    def _convert_single_annotation(self, ann_file, format_type, img_file):
        """Convert single annotation file to YOLO format"""
        try:
            if format_type == '.xml':
                return self._convert_xml_to_yolo(ann_file, img_file)
            elif format_type == '.json':
                return self._convert_json_to_yolo(ann_file, img_file)
            elif format_type == '.txt':
                return self._convert_txt_to_yolo(ann_file, img_file)
            elif format_type == '.csv':
                return self._convert_csv_to_yolo(ann_file, img_file)
            else:
                print(f"❌ Unsupported format: {format_type}")
                return None
        except Exception as e:
            print(f"❌ Error converting {ann_file.name}: {str(e)}")
            return None
    
    def _convert_xml_to_yolo(self, xml_file, img_file):
        """Convert XML (Pascal VOC) format to YOLO"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Get class ID
            if class_name in self.class_mapping:
                class_id = self.class_mapping[class_name]
            else:
                # Auto-assign class ID if not in mapping
                if class_name not in [v for v in self.class_mapping.values()]:
                    class_id = len(self.class_mapping)
                    self.class_mapping[class_name] = class_id
                else:
                    class_id = self.class_mapping[class_name]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center_x, center_y, width, height) - normalized
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_annotations.append([class_id, center_x, center_y, width, height])
        
        return yolo_annotations
    
    def _convert_json_to_yolo(self, json_file, img_file):
        """Convert JSON format to YOLO"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Try to get image dimensions
        img = Image.open(img_file)
        img_width, img_height = img.size
        
        yolo_annotations = []
        
        # Handle different JSON formats
        if 'annotations' in data:
            # COCO format
            for ann in data['annotations']:
                if 'bbox' in ann and 'category_id' in ann:
                    class_id = ann['category_id']
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    yolo_annotations.append([class_id, center_x, center_y, width, height])
        
        elif 'shapes' in data:
            # LabelMe format
            for shape in data['shapes']:
                class_name = shape['label']
                
                # Get class ID
                if class_name in self.class_mapping:
                    class_id = self.class_mapping[class_name]
                else:
                    class_id = len(self.class_mapping)
                    self.class_mapping[class_name] = class_id
                
                # Get bounding box from points
                points = shape['points']
                if len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    
                    # Convert to YOLO format
                    center_x = (xmin + xmax) / (2 * img_width)
                    center_y = (ymin + ymax) / (2 * img_height)
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    yolo_annotations.append([class_id, center_x, center_y, width, height])
        
        return yolo_annotations
    
    def _convert_txt_to_yolo(self, txt_file, img_file):
        """Convert TXT format to YOLO (if already in YOLO format, just copy)"""
        # Check if already in YOLO format
        with open(txt_file, 'r') as f:
            first_line = f.readline().strip()
        
        # If already in YOLO format, just return the content
        if first_line and len(first_line.split()) == 5:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            yolo_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    yolo_annotations.append([float(x) for x in parts])
            
            return yolo_annotations
        
        return []
    
    def _convert_csv_to_yolo(self, csv_file, img_file):
        """Convert CSV format to YOLO"""
        import pandas as pd
        
        try:
            df = pd.read_csv(csv_file)
            img = Image.open(img_file)
            img_width, img_height = img.size
            
            yolo_annotations = []
            
            # Common CSV column names
            x_col = None
            y_col = None
            w_col = None
            h_col = None
            class_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'x' in col_lower and 'min' in col_lower:
                    x_col = col
                elif 'y' in col_lower and 'min' in col_lower:
                    y_col = col
                elif 'width' in col_lower or 'w' in col_lower:
                    w_col = col
                elif 'height' in col_lower or 'h' in col_lower:
                    h_col = col
                elif 'class' in col_lower or 'label' in col_lower:
                    class_col = col
            
            if all([x_col, y_col, w_col, h_col, class_col]):
                for _, row in df.iterrows():
                    class_name = str(row[class_col])
                    
                    # Get class ID
                    if class_name in self.class_mapping:
                        class_id = self.class_mapping[class_name]
                    else:
                        class_id = len(self.class_mapping)
                        self.class_mapping[class_name] = class_id
                    
                    x, y, w, h = row[x_col], row[y_col], row[w_col], row[h_col]
                    
                    # Convert to YOLO format
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    yolo_annotations.append([class_id, center_x, center_y, width, height])
            
            return yolo_annotations
            
        except Exception as e:
            print(f"❌ Error reading CSV: {str(e)}")
            return []
    
    def _save_yolo_labels(self, annotations, output_file):
        """Save YOLO format labels to file"""
        with open(output_file, 'w') as f:
            for ann in annotations:
                line = ' '.join([str(x) for x in ann])
                f.write(line + '\n')
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_mapping),
            'names': [k for k, v in sorted(self.class_mapping.items(), key=lambda x: x[1])]
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created dataset.yaml: {yaml_path}")
        print(f"📋 Classes: {yaml_content['names']}")
        
        return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Prepare biomedical data for YOLO training')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images and annotations')
    parser.add_argument('--output', type=str, default='./dataset', help='Output directory for YOLO format')
    parser.add_argument('--format', type=str, choices=['auto', 'xml', 'json', 'txt', 'csv'], 
                       default='auto', help='Annotation format (auto-detect if not specified)')
    parser.add_argument('--classes', type=str, help='Comma-separated list of class names')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training data split ratio')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation data split ratio')
    parser.add_argument('--test-split', type=float, default=0.1, help='Test data split ratio')
    
    args = parser.parse_args()
    
    print("🏥 Biomedical Data Preparation for YOLO")
    print("=" * 50)
    
    # Parse class mapping if provided
    class_mapping = {}
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(',')]
        class_mapping = {name: i for i, name in enumerate(class_names)}
        print(f"📋 Using provided classes: {class_names}")
    
    # Initialize data preparation
    prep = BiomedicalDataPreparation(args.input, args.output, class_mapping)
    
    # Detect format if auto
    if args.format == 'auto':
        detected_format = prep.detect_annotation_format()
        if not detected_format:
            print("❌ Could not detect annotation format. Please specify manually.")
            return
        format_type = detected_format
    else:
        format_type = f".{args.format}"
    
    # Convert annotations
    prep.convert_annotations(
        format_type, 
        args.train_split, 
        args.val_split, 
        args.test_split
    )
    
    # Create dataset.yaml
    prep.create_dataset_yaml()
    
    print(f"\n🎉 Data preparation completed!")
    print(f"📁 Output directory: {args.output}")
    print(f"📊 Total classes: {len(prep.class_mapping)}")
    print(f"🔧 Next step: Run training with 'python train.py'")

if __name__ == "__main__":
    main()
