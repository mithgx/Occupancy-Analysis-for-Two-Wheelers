#!/usr/bin/env python3
"""
Test Setup Script for Triples Detection
Capstone Project - 2-Wheeler Triples Detection
"""

import os
import sys
import torch
import yaml

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    # Test PyTorch
    try:
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Test Ultralytics
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics: Available")
    except ImportError:
        print("❌ Ultralytics not found")
        return False
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    # Test other dependencies
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    return True

def test_dataset():
    """Test if dataset is properly configured"""
    print("\n📊 Testing dataset configuration...")
    
    # Check if triples.yaml exists
    if not os.path.exists('triples.yaml'):
        print("❌ triples.yaml not found")
        return False
    
    # Load and validate YAML
    try:
        with open('triples.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ triples.yaml loaded successfully")
        
        # Check required keys
        required_keys = ['train', 'val', 'test', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing key: {key}")
                return False
        
        print("✅ All required keys present")
        
        # Check class names
        if len(config['names']) != 2:
            print(f"❌ Expected 2 classes, found {len(config['names'])}")
            return False
        
        print(f"✅ Classes: {list(config['names'].values())}")
        
        # Check if directories exist
        for split in ['train', 'val', 'test']:
            images_dir = config[split]
            labels_dir = images_dir.replace('images', 'labels')
            
            if not os.path.exists(images_dir):
                print(f"❌ Images directory not found: {images_dir}")
                return False
            
            if not os.path.exists(labels_dir):
                print(f"❌ Labels directory not found: {labels_dir}")
                return False
            
            # Count files
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            print(f"   {split}: {len(image_files)} images, {len(label_files)} labels")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset config: {str(e)}")
        return False

def test_model_loading():
    """Test if YOLO model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small model
        print("   Loading YOLOv8n...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n loaded successfully")
        
        # Test basic inference
        print("   Testing basic inference...")
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✅ Basic inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def test_training_environment():
    """Test if training environment is ready"""
    print("\n🏋️  Testing training environment...")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ System memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
        
        if memory.available < 4 * (1024**3):  # Less than 4GB available
            print("⚠️  Low memory available - consider reducing batch size")
        
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        print(f"✅ Disk space: {free // (1024**3)} GB available")
        
        if free < 5 * (1024**3):  # Less than 5GB free
            print("⚠️  Low disk space - training may fail")
            
    except Exception as e:
        print(f"⚠️  Could not check disk space: {str(e)}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Triples Detection - Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    if not test_dependencies():
        all_tests_passed = False
    
    if not test_dataset():
        all_tests_passed = False
    
    if not test_model_loading():
        all_tests_passed = False
    
    test_training_environment()  # This is informational, doesn't fail the test
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Run: python train_triples.py --epochs 100 --batch-size 8")
        print("2. Or use the batch file: run_training.bat")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check dataset structure and triples.yaml file")
        print("3. Ensure sufficient disk space and memory")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
