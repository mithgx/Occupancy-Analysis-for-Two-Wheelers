#!/usr/bin/env python3
"""
Triples Detection Training Script
Capstone Project - 2-Wheeler Triples Detection
"""

import os
import yaml
from ultralytics import YOLO
import torch
import argparse
from pathlib import Path

def train_model(data_yaml_path, epochs=100, batch_size=16, imgsz=640, device='auto'):
    """
    Train YOLOv8 model for triples detection
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        imgsz (int): Input image size
        device (str): Device to use for training ('cpu', '0', 'auto')
    """
    
    print("🚀 Starting Triples Detection Model Training...")
    print(f"📊 Dataset config: {data_yaml_path}")
    print(f"⚙️  Training parameters: {epochs} epochs, batch_size={batch_size}, imgsz={imgsz}")
    
    # Check CUDA availability and adjust device accordingly
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'auto'  # Let YOLO auto-detect GPU
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("⚠️  CUDA not available, using CPU")
    
    # Adjust batch size for CPU training if needed
    adjusted_batch_size = batch_size
    if device == 'cpu' and batch_size > 8:
        adjusted_batch_size = 8
        print(f"⚠️  Reduced batch size to {adjusted_batch_size} for CPU training")
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano model, can be changed to s/m/l/x
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=adjusted_batch_size,
        imgsz=imgsz,
        device=device,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,
        project='triples_detection',
        name='yolov8_triples',
        exist_ok=True,
        verbose=True,
        plots=True,
        save_txt=True,
        save_conf=True
    )
    
    print("✅ Training completed successfully!")
    return results

def validate_model(model_path, data_yaml_path, device='auto'):
    """
    Validate the trained model
    
    Args:
        model_path (str): Path to trained model weights
        data_yaml_path (str): Path to data.yaml file
        device (str): Device to use for validation
    """
    
    print("🔍 Validating trained model...")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate the model
    results = model.val(
        data=data_yaml_path,
        device=device,
        plots=True,
        save_txt=True,
        save_conf=True
    )
    
    print("✅ Validation completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train Triples Detection Model')
    parser.add_argument('--data', type=str, default='triples.yaml', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, 0, auto)')
    parser.add_argument('--validate', action='store_true', help='Validate model after training')
    
    args = parser.parse_args()
    
    # Check if data.yaml exists
    if not os.path.exists(args.data):
        print(f"❌ Error: {args.data} not found!")
        return
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        if args.device == 'auto':
            print("🎯 Using auto device selection for GPU")
    else:
        print("⚠️  CUDA not available, using CPU")
        if args.device == 'auto':
            args.device = 'cpu'
            print("🔄 Automatically switched to CPU mode")
    
    try:
        # Train the model
        results = train_model(
            data_yaml_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device
        )
        
        # Validate if requested
        if args.validate:
            model_path = 'triples_detection/yolov8_triples/weights/best.pt'
            if os.path.exists(model_path):
                validate_model(model_path, args.data, args.device)
            else:
                print(f"⚠️  Model weights not found at {model_path}")
        
        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Check the 'triples_detection' folder for results")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()

