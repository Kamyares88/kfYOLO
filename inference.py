#!/usr/bin/env python3
"""
YOLO Inference Script for Biomedical Object Detection
Supports both object detection and instance segmentation
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json

class BiomedicalYOLOInference:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLO inference
        
        Args:
            model_path (str): Path to trained YOLO model
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"🔍 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Get class names
        self.class_names = self.model.names
        print(f"✅ Model loaded successfully with {len(self.class_names)} classes")
        print(f"📋 Classes: {list(self.class_names.values())}")
        
        # Check if model supports segmentation
        self.supports_segmentation = hasattr(self.model, 'task') and self.model.task == 'segment'
        print(f"🎯 Segmentation support: {'Yes' if self.supports_segmentation else 'No'}")

    def predict_single_image(self, image_path, save_results=True, output_dir="./results"):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to input image
            save_results (bool): Whether to save results
            output_dir (str): Directory to save results
            
        Returns:
            dict: Prediction results
        """
        print(f"🔍 Processing image: {image_path}")
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Process results
        processed_results = self._process_results(results[0], image_path)
        
        # Save results if requested
        if save_results:
            self._save_results(processed_results, image_path, output_dir)
        
        return processed_results

    def predict_batch(self, image_dir, save_results=True, output_dir="./results"):
        """
        Run inference on a batch of images
        
        Args:
            image_dir (str): Directory containing images
            save_results (bool): Whether to save results
            output_dir (str): Directory to save results
            
        Returns:
            list: List of prediction results
        """
        print(f"🔍 Processing batch from: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return []
        
        print(f"📸 Found {len(image_files)} images")
        
        # Process each image
        all_results = []
        for i, image_path in enumerate(image_files):
            print(f"🔄 Processing {i+1}/{len(image_files)}: {image_path.name}")
            try:
                result = self.predict_single_image(str(image_path), save_results, output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {str(e)}")
                continue
        
        return all_results

    def _process_results(self, result, image_path):
        """Process YOLO results into a structured format"""
        processed = {
            'image_path': image_path,
            'image_shape': result.orig_shape,
            'detections': [],
            'summary': {}
        }
        
        # Process detections
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])]
                }
                
                # Add segmentation if available
                if self.supports_segmentation and result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()
                    detection['segmentation'] = mask.tolist()
                
                processed['detections'].append(detection)
        
        # Add summary statistics
        processed['summary'] = {
            'total_detections': len(processed['detections']),
            'classes_detected': list(set([d['class_name'] for d in processed['detections']])),
            'confidence_range': {
                'min': min([d['confidence'] for d in processed['detections']]) if processed['detections'] else 0,
                'max': max([d['confidence'] for d in processed['detections']]) if processed['detections'] else 0
            }
        }
        
        return processed

    def _save_results(self, results, image_path, output_dir):
        """Save inference results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        base_name = Path(image_path).stem
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save annotated image
        annotated_image = self._create_annotated_image(image_path, results)
        image_output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        annotated_image.save(image_output_path)
        
        print(f"💾 Results saved to: {output_dir}")

    def _create_annotated_image(self, image_path, results):
        """Create annotated image with bounding boxes and labels"""
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw detections
        for detection in results['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            
            # Draw label background
            draw.rectangle([x1, y1-label_height-5, x1+label_width+10, y1], fill='red')
            
            # Draw label text
            draw.text((x1+5, y1-label_height-2), label, fill='white', font=font)
        
        return image

    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            'model_path': self.model_path,
            'class_names': self.class_names,
            'supports_segmentation': self.supports_segmentation,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }
        return info

def main():
    parser = argparse.ArgumentParser(description='YOLO Inference for Biomedical Object Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLO model')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--batch', action='store_true', help='Process input as directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    print("🏥 YOLO Biomedical Object Detection Inference")
    print("=" * 50)
    
    # Initialize inference
    try:
        inference = BiomedicalYOLOInference(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return
    
    # Display model info
    model_info = inference.get_model_info()
    print(f"📊 Model Information:")
    print(f"   - Path: {model_info['model_path']}")
    print(f"   - Classes: {len(model_info['class_names'])}")
    print(f"   - Segmentation: {'Yes' if model_info['supports_segmentation'] else 'No'}")
    print(f"   - Confidence threshold: {model_info['conf_threshold']}")
    print(f"   - IoU threshold: {model_info['iou_threshold']}")
    print()
    
    # Run inference
    if args.batch or os.path.isdir(args.input):
        print(f"🔄 Running batch inference on directory: {args.input}")
        results = inference.predict_batch(
            image_dir=args.input,
            save_results=not args.no_save,
            output_dir=args.output
        )
        print(f"✅ Batch processing completed. Processed {len(results)} images.")
    else:
        print(f"🔄 Running single image inference: {args.input}")
        result = inference.predict_single_image(
            image_path=args.input,
            save_results=not args.no_save,
            output_dir=args.output
        )
        
        # Display results
        print(f"\n📊 Detection Results:")
        print(f"   - Total detections: {result['summary']['total_detections']}")
        print(f"   - Classes detected: {', '.join(result['summary']['classes_detected'])}")
        if result['summary']['total_detections'] > 0:
            print(f"   - Confidence range: {result['summary']['confidence_range']['min']:.3f} - {result['summary']['confidence_range']['max']:.3f}")
        
        # Show individual detections
        for i, detection in enumerate(result['detections']):
            print(f"   {i+1}. {detection['class_name']}: {detection['confidence']:.3f}")

if __name__ == "__main__":
    main()
