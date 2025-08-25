#!/usr/bin/env python3
"""
Model Evaluation Script for Triples Detection
Capstone Project - 2-Wheeler Triples Detection
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class TriplesEvaluator:
    def __init__(self, model_path, data_yaml_path):
        """
        Initialize the Triples Evaluator
        
        Args:
            model_path (str): Path to trained model weights
            data_yaml_path (str): Path to data.yaml file
        """
        self.model = YOLO(model_path)
        self.data_yaml_path = data_yaml_path
        self.class_names = ['legal_2_or_less', 'illegal_3_or_more']
        
        # Load dataset paths
        self._load_dataset_paths()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📊 Dataset config: {data_yaml_path}")
    
    def _load_dataset_paths(self):
        """Load dataset paths from YAML file"""
        import yaml
        
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.test_images_dir = data_config.get('test', 'test/images')
        self.test_labels_dir = data_config.get('test', 'test/labels').replace('images', 'labels')
        
        print(f"📁 Test images: {self.test_images_dir}")
        print(f"📁 Test labels: {self.test_labels_dir}")
    
    def evaluate_model(self, conf_threshold=0.5, iou_threshold=0.45):
        """
        Evaluate model performance on test set
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            
        Returns:
            dict: Evaluation results
        """
        print("🔍 Starting model evaluation...")
        
        # Get test image files
        test_images = [f for f in os.listdir(self.test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📸 Found {len(test_images)} test images")
        
        # Initialize metrics
        all_predictions = []
        all_ground_truth = []
        class_metrics = {class_name: {'tp': 0, 'fp': 0, 'fn': 0} for class_name in self.class_names}
        
        # Process each test image
        for i, image_file in enumerate(test_images):
            print(f"Processing {i+1}/{len(test_images)}: {image_file}")
            
            # Load image and ground truth
            image_path = os.path.join(self.test_images_dir, image_file)
            label_path = os.path.join(self.test_labels_dir, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            # Get predictions
            predictions = self._get_predictions(image_path, conf_threshold, iou_threshold)
            
            # Get ground truth
            ground_truth = self._load_ground_truth(label_path)
            
            # Store for overall metrics
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            
            # Calculate per-class metrics
            self._calculate_class_metrics(predictions, ground_truth, class_metrics)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(class_metrics)
        
        # Generate evaluation report
        evaluation_report = {
            'class_metrics': class_metrics,
            'overall_metrics': overall_metrics,
            'predictions': all_predictions,
            'ground_truth': all_ground_truth
        }
        
        print("✅ Evaluation completed!")
        return evaluation_report
    
    def _get_predictions(self, image_path, conf_threshold, iou_threshold):
        """
        Get model predictions for an image
        
        Args:
            image_path (str): Path to image
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold
            
        Returns:
            list: List of prediction dictionaries
        """
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        
        predictions = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                predictions.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        
        return predictions
    
    def _load_ground_truth(self, label_path):
        """
        Load ground truth annotations from YOLO format
        
        Args:
            label_path (str): Path to label file
            
        Returns:
            list: List of ground truth dictionaries
        """
        ground_truth = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        # Note: This is a simplified conversion, you might need to load image dimensions
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2
                        
                        ground_truth.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
        
        return ground_truth
    
    def _calculate_class_metrics(self, predictions, ground_truth, class_metrics):
        """
        Calculate per-class metrics (TP, FP, FN)
        
        Args:
            predictions (list): Model predictions
            ground_truth (list): Ground truth annotations
            class_metrics (dict): Dictionary to store metrics
        """
        # Simple matching based on class and IoU
        # In a real implementation, you'd want more sophisticated matching
        
        for gt in ground_truth:
            gt_class = gt['class_name']
            matched = False
            
            for pred in predictions:
                if pred['class_name'] == gt_class:
                    # Calculate IoU (simplified)
                    iou = self._calculate_iou(gt['bbox'], pred['bbox'])
                    if iou > 0.5:  # IoU threshold
                        class_metrics[gt_class]['tp'] += 1
                        matched = True
                        break
            
            if not matched:
                class_metrics[gt_class]['fn'] += 1
        
        # Count false positives
        for pred in predictions:
            pred_class = pred['class_name']
            matched = False
            
            for gt in ground_truth:
                if gt['class_name'] == pred_class:
                    iou = self._calculate_iou(gt['bbox'], pred['bbox'])
                    if iou > 0.5:
                        matched = True
                        break
            
            if not matched:
                class_metrics[pred_class]['fp'] += 1
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
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
    
    def _calculate_overall_metrics(self, class_metrics):
        """
        Calculate overall metrics from per-class metrics
        
        Args:
            class_metrics (dict): Per-class metrics
            
        Returns:
            dict: Overall metrics
        """
        overall_metrics = {}
        
        for class_name, metrics in class_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            overall_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Calculate macro averages
        avg_precision = np.mean([metrics['precision'] for metrics in overall_metrics.values()])
        avg_recall = np.mean([metrics['recall'] for metrics in overall_metrics.values()])
        avg_f1 = np.mean([metrics['f1_score'] for metrics in overall_metrics.values()])
        
        overall_metrics['macro_avg'] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        }
        
        return overall_metrics
    
    def generate_visualizations(self, evaluation_report, output_dir='evaluation_results'):
        """
        Generate evaluation visualizations
        
        Args:
            evaluation_report (dict): Evaluation results
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Generating evaluation visualizations...")
        
        # 1. Per-class metrics bar chart
        self._plot_class_metrics(evaluation_report['overall_metrics'], output_dir)
        
        # 2. Confusion matrix heatmap
        self._plot_confusion_matrix(evaluation_report, output_dir)
        
        # 3. Precision-Recall curves
        self._plot_precision_recall_curves(evaluation_report, output_dir)
        
        # 4. Save detailed metrics to JSON
        self._save_metrics_json(evaluation_report, output_dir)
        
        print(f"📁 Visualizations saved to: {output_dir}")
    
    def _plot_class_metrics(self, overall_metrics, output_dir):
        """Plot per-class precision, recall, and F1-score"""
        classes = [name for name in self.class_names if name in overall_metrics]
        precision = [overall_metrics[name]['precision'] for name in classes]
        recall = [overall_metrics[name]['recall'] for name in classes]
        f1_scores = [overall_metrics[name]['f1_score'] for name in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        ax.bar(x, recall, width, label='Recall', color='lightcoral')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, evaluation_report, output_dir):
        """Plot confusion matrix heatmap"""
        # Create confusion matrix data
        cm_data = np.zeros((len(self.class_names), len(self.class_names)))
        
        for class_name in self.class_names:
            if class_name in evaluation_report['overall_metrics']:
                metrics = evaluation_report['overall_metrics'][class_name]
                class_idx = self.class_names.index(class_name)
                
                # TP on diagonal
                cm_data[class_idx, class_idx] = metrics['tp']
                
                # FP in row (excluding diagonal)
                for other_idx in range(len(self.class_names)):
                    if other_idx != class_idx:
                        cm_data[class_idx, other_idx] = metrics['fp']
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='g', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, evaluation_report, output_dir):
        """Plot precision-recall curves for each class"""
        plt.figure(figsize=(10, 6))
        
        for class_name in self.class_names:
            if class_name in evaluation_report['overall_metrics']:
                metrics = evaluation_report['overall_metrics'][class_name]
                precision = metrics['precision']
                recall = metrics['recall']
                
                plt.scatter(recall, precision, label=f'{class_name} (P={precision:.3f}, R={recall:.3f})', s=100)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Scatter Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics_json(self, evaluation_report, output_dir):
        """Save detailed metrics to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_data = convert_numpy_types(evaluation_report)
        
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def print_evaluation_summary(self, evaluation_report):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        overall_metrics = evaluation_report['overall_metrics']
        
        for class_name in self.class_names:
            if class_name in overall_metrics:
                metrics = overall_metrics[class_name]
                print(f"\n🎯 {class_name.upper()}:")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall:    {metrics['recall']:.3f}")
                print(f"   F1-Score:  {metrics['f1_score']:.3f}")
                print(f"   TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        
        if 'macro_avg' in overall_metrics:
            macro_avg = overall_metrics['macro_avg']
            print(f"\n📈 MACRO AVERAGES:")
            print(f"   Precision: {macro_avg['precision']:.3f}")
            print(f"   Recall:    {macro_avg['recall']:.3f}")
            print(f"   F1-Score:  {macro_avg['f1_score']:.3f}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Triples Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='triples.yaml', help='Path to data.yaml file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        return
    
    # Check if data.yaml exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data config not found at {args.data}")
        return
    
    try:
        # Initialize evaluator
        evaluator = TriplesEvaluator(args.model, args.data)
        
        # Run evaluation
        evaluation_report = evaluator.evaluate_model(args.conf, args.iou)
        
        # Print summary
        evaluator.print_evaluation_summary(evaluation_report)
        
        # Generate visualizations
        evaluator.generate_visualizations(evaluation_report, args.output)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

