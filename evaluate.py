#!/usr/bin/env python3
"""
YOLO Model Evaluation Script for Biomedical Object Detection
Evaluates model performance on test data and generates comprehensive reports
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

class BiomedicalYOLOEvaluation:
    def __init__(self, model_path, dataset_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize evaluation
        
        Args:
            model_path (str): Path to trained YOLO model
            dataset_path (str): Path to dataset.yaml file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"🔍 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Load dataset info
        self._load_dataset_info()
        
        # Results storage
        self.evaluation_results = {}
        
    def _load_dataset_info(self):
        """Load dataset information from dataset.yaml"""
        import yaml
        
        with open(self.dataset_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
        
        self.class_names = dataset_info['names']
        self.num_classes = dataset_info['nc']
        self.dataset_root = Path(dataset_info['path'])
        
        print(f"✅ Dataset loaded: {self.num_classes} classes")
        print(f"📋 Classes: {self.class_names}")
    
    def evaluate_model(self, output_dir="./evaluation_results"):
        """Run comprehensive model evaluation"""
        print("🚀 Starting model evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get test images
        test_images_dir = self.dataset_root / "images/test"
        test_labels_dir = self.dataset_root / "labels/test"
        
        if not test_images_dir.exists():
            print("❌ Test images directory not found")
            return
        
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if not test_images:
            print("❌ No test images found")
            return
        
        print(f"📸 Found {len(test_images)} test images")
        
        # Run predictions
        all_predictions = []
        all_ground_truth = []
        
        for i, img_path in enumerate(test_images):
            print(f"🔄 Processing {i+1}/{len(test_images)}: {img_path.name}")
            
            # Get ground truth
            gt_path = test_labels_dir / f"{img_path.stem}.txt"
            ground_truth = self._load_ground_truth(gt_path, img_path)
            
            # Run prediction
            predictions = self._run_prediction(img_path)
            
            # Store results
            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)
        
        # Calculate metrics
        self._calculate_metrics(all_predictions, all_ground_truth)
        
        # Generate reports
        self._generate_reports(output_dir)
        
        # Create visualizations
        self._create_visualizations(output_dir)
        
        print(f"✅ Evaluation completed! Results saved to: {output_dir}")
    
    def _load_ground_truth(self, label_path, image_path):
        """Load ground truth annotations"""
        ground_truth = []
        
        if not label_path.exists():
            return ground_truth
        
        # Load image dimensions
        img = cv2.imread(str(image_path))
        img_height, img_width = img.shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to absolute coordinates
                    x1 = (center_x - width/2) * img_width
                    y1 = (center_y - height/2) * img_height
                    x2 = (center_x + width/2) * img_width
                    y2 = (center_y + height/2) * img_height
                    
                    ground_truth.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'class_name': self.class_names[class_id]
                    })
        
        return ground_truth
    
    def _run_prediction(self, image_path):
        """Run model prediction on single image"""
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        predictions = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                predictions.append({
                    'class_id': int(class_ids[i]),
                    'bbox': boxes[i].tolist(),
                    'confidence': float(confidences[i]),
                    'class_name': self.class_names[int(class_ids[i])]
                })
        
        return predictions
    
    def _calculate_metrics(self, predictions, ground_truth):
        """Calculate evaluation metrics"""
        print("📊 Calculating evaluation metrics...")
        
        # Initialize metrics storage
        class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(self.num_classes)}
        
        # Calculate IoU and assign predictions to ground truth
        for pred_list, gt_list in zip(predictions, ground_truth):
            # For each ground truth, find best matching prediction
            for gt in gt_list:
                best_iou = 0
                best_pred_idx = -1
                
                for i, pred in enumerate(pred_list):
                    if pred['class_id'] == gt['class_id']:
                        iou = self._calculate_iou(gt['bbox'], pred['bbox'])
                        if iou > best_iou and iou >= self.iou_threshold:
                            best_iou = iou
                            best_pred_idx = i
                
                if best_pred_idx >= 0:
                    class_metrics[gt['class_id']]['tp'] += 1
                    # Remove matched prediction
                    pred_list.pop(best_pred_idx)
                else:
                    class_metrics[gt['class_id']]['fn'] += 1
            
            # Remaining predictions are false positives
            for pred in pred_list:
                class_metrics[pred['class_id']]['fp'] += 1
        
        # Calculate per-class metrics
        self.evaluation_results = {}
        for class_id in range(self.num_classes):
            tp = class_metrics[class_id]['tp']
            fp = class_metrics[class_id]['fp']
            fn = class_metrics[class_id]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.evaluation_results[self.class_names[class_id]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        # Calculate overall metrics
        total_tp = sum(class_metrics[c]['tp'] for c in range(self.num_classes))
        total_fp = sum(class_metrics[c]['fp'] for c in range(self.num_classes))
        total_fn = sum(class_metrics[c]['fn'] for c in range(self.num_classes))
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        self.evaluation_results['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_reports(self, output_dir):
        """Generate evaluation reports"""
        print("📝 Generating evaluation reports...")
        
        # Save detailed results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Create summary report
        summary_file = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("YOLO Biomedical Object Detection - Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Overall Performance:\n")
            overall = self.evaluation_results['overall']
            f.write(f"  Precision: {overall['precision']:.4f}\n")
            f.write(f"  Recall: {overall['recall']:.4f}\n")
            f.write(f"  F1-Score: {overall['f1_score']:.4f}\n")
            f.write(f"  True Positives: {overall['true_positives']}\n")
            f.write(f"  False Positives: {overall['false_positives']}\n")
            f.write(f"  False Negatives: {overall['false_negatives']}\n\n")
            
            f.write("Per-Class Performance:\n")
            for class_name, metrics in self.evaluation_results.items():
                if class_name != 'overall':
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F1-Score: {metrics['f1_score']:.4f}\n")
                    f.write(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}\n\n")
        
        print(f"📄 Reports saved to: {output_dir}")
    
    def _create_visualizations(self, output_dir):
        """Create evaluation visualizations"""
        print("📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Per-class performance comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        class_names = [name for name in self.evaluation_results.keys() if name != 'overall']
        precisions = [self.evaluation_results[name]['precision'] for name in class_names]
        recalls = [self.evaluation_results[name]['recall'] for name in class_names]
        f1_scores = [self.evaluation_results[name]['f1_score'] for name in class_names]
        
        # Precision
        axes[0].bar(class_names, precisions, color='skyblue')
        axes[0].set_title('Per-Class Precision')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # Recall
        axes[1].bar(class_names, recalls, color='lightcoral')
        axes[1].set_title('Per-Class Recall')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
        
        # F1-Score
        axes[2].bar(class_names, f1_scores, color='lightgreen')
        axes[2].set_title('Per-Class F1-Score')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_class_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix style visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix data
        matrix_data = []
        for class_name in class_names:
            metrics = self.evaluation_results[class_name]
            matrix_data.append([metrics['true_positives'], metrics['false_positives'], 
                              metrics['false_negatives']])
        
        matrix_data = np.array(matrix_data)
        
        # Create heatmap
        sns.heatmap(matrix_data, 
                   annot=True, 
                   fmt='d',
                   xticklabels=['TP', 'FP', 'FN'],
                   yticklabels=class_names,
                   cmap='Blues',
                   ax=ax)
        
        ax.set_title('Detection Results Matrix')
        ax.set_xlabel('Prediction Type')
        ax.set_ylabel('Class')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "detection_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Overall metrics pie chart
        overall = self.evaluation_results['overall']
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = ['True Positives', 'False Positives', 'False Negatives']
        sizes = [overall['true_positives'], overall['false_positives'], overall['false_negatives']]
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Detection Results Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for biomedical object detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLO model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.yaml file')
    parser.add_argument('--output', type=str, default='./evaluation_results', help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    
    args = parser.parse_args()
    
    print("🏥 YOLO Biomedical Object Detection - Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluation
    try:
        evaluator = BiomedicalYOLOEvaluation(
            model_path=args.model,
            dataset_path=args.dataset,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
    except Exception as e:
        print(f"❌ Failed to initialize evaluation: {str(e)}")
        return
    
    # Run evaluation
    evaluator.evaluate_model(args.output)
    
    # Display summary
    print("\n📊 Evaluation Summary:")
    overall = evaluator.evaluation_results['overall']
    print(f"   Overall Precision: {overall['precision']:.4f}")
    print(f"   Overall Recall: {overall['recall']:.4f}")
    print(f"   Overall F1-Score: {overall['f1_score']:.4f}")
    print(f"   Total Detections: {overall['true_positives'] + overall['false_positives']}")
    print(f"   Total Ground Truth: {overall['true_positives'] + overall['false_negatives']}")

if __name__ == "__main__":
    main()
