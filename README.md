# 🚗 Triples Detection on 2-Wheelers - Capstone Project

A computer vision system for detecting the number of passengers on 2-wheelers (motorcycles/bikes) using YOLOv8 object detection. This project addresses road safety concerns by identifying instances of triple riding, which is illegal in many jurisdictions.

## 🎯 Project Overview

This system detects three classes:
- **Single**: One person on the vehicle (✅ Legal)
- **Double**: Two people on the vehicle (✅ Legal)  
- **Triple**: Three or more people on the vehicle (❌ Illegal - Triple Riding)

## 📊 Dataset Information

- **Source**: Roboflow Universe - Tripling Detection Dataset
- **Total Images**: 329 annotated images
- **Format**: YOLO v5 PyTorch format
- **Classes**: 3 (single, double, triple)
- **Split**: Train (223), Validation (60), Test (46)

## 🏗️ Project Structure

```
DATASET1_HELMET/
├── train/                 # Training images and labels
├── valid/                 # Validation images and labels  
├── test/                  # Test images and labels
├── triples.yaml          # Dataset configuration
├── train_triples.py      # Training script
├── detect_triples.py     # Detection/inference script
├── evaluate_model.py     # Model evaluation script
├── demo_app.py          # GUI demo application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_triples.py --epochs 100 --batch-size 16
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device to use (cpu, 0, auto)
- `--validate`: Validate model after training

### 3. Run Detection

**On Images:**
```bash
python detect_triples.py --model triples_detection/yolov8_triples/weights/best.pt --source path/to/image.jpg --mode image
```

**On Videos:**
```bash
python detect_triples.py --model triples_detection/yolov8_triples/weights/best.pt --source path/to/video.mp4 --mode video
```

**Real-time (Webcam):**
```bash
python detect_triples.py --model triples_detection/yolov8_triples/weights/best.pt --source 0 --mode realtime
```

### 4. Evaluate Model Performance

```bash
python evaluate_model.py --model triples_detection/yolov8_triples/weights/best.pt --data triples.yaml
```

### 5. Launch Demo Application

```bash
python demo_app.py
```

## 🔧 Configuration

### Dataset Configuration (`triples.yaml`)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

names:
  0: single
  1: double
  2: triple
```

### Model Parameters

- **Confidence Threshold**: Minimum confidence for detections (default: 0.5)
- **IoU Threshold**: Intersection over Union threshold for NMS (default: 0.45)
- **Input Size**: Model input resolution (default: 640x640)

## 📈 Training Details

### Model Architecture
- **Base Model**: YOLOv8 (nano, small, medium, large, or xlarge)
- **Input Size**: 640x640 pixels
- **Optimizer**: AdamW
- **Learning Rate**: Auto-scaled based on batch size
- **Data Augmentation**: Mosaic, random affine, color jittering

### Training Process
1. **Pre-training**: Uses pre-trained YOLOv8 weights
2. **Fine-tuning**: Adapts to triples detection dataset
3. **Validation**: Regular validation on validation set
4. **Early Stopping**: Stops training if no improvement for 20 epochs
5. **Model Saving**: Saves best and last checkpoints

## 🎮 Demo Application Features

The GUI demo application (`demo_app.py`) provides:

- **Model Loading**: Load trained model weights
- **Image Upload**: Load images for detection
- **Parameter Tuning**: Adjust confidence and IoU thresholds
- **Real-time Detection**: Perform detection with visual feedback
- **Result Visualization**: Display bounding boxes and labels
- **Result Saving**: Save annotated images

## 📊 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Per-class Performance**: Precision, Recall, F1-Score for each class
- **Confusion Matrix**: Visual representation of predictions vs. ground truth
- **Precision-Recall Curves**: Performance across different thresholds
- **Overall Metrics**: Macro-averaged performance scores

## 🚨 Safety and Legal Considerations

### Triple Riding Dangers
- **Reduced Stability**: Higher center of gravity affects balance
- **Limited Control**: Driver has restricted movement and visibility
- **Increased Risk**: Higher probability of accidents and injuries
- **Legal Violation**: Illegal in most jurisdictions

### Use Cases
- **Traffic Monitoring**: Automated detection in surveillance systems
- **Law Enforcement**: Support for traffic rule enforcement
- **Safety Research**: Data collection for road safety analysis
- **Public Awareness**: Educational content and campaigns

## 🔍 Technical Implementation

### Object Detection Pipeline
1. **Image Preprocessing**: Resize, normalize, and format input
2. **Model Inference**: Forward pass through YOLOv8 network
3. **Post-processing**: Non-maximum suppression and threshold filtering
4. **Visualization**: Draw bounding boxes and labels
5. **Output Generation**: Return detection results and annotated images

### Performance Optimization
- **GPU Acceleration**: CUDA support for faster inference
- **Batch Processing**: Efficient handling of multiple images
- **Memory Management**: Optimized tensor operations
- **Real-time Processing**: Optimized for live video streams

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_triples.py --batch-size 8

# Use smaller model
# Change from yolov8l.pt to yolov8n.pt in training script
```

**2. Model Not Loading**
```bash
# Check model path
ls -la triples_detection/yolov8_triples/weights/

# Verify file permissions
chmod 644 triples_detection/yolov8_triples/weights/*.pt
```

**3. Poor Detection Quality**
```bash
# Increase training epochs
python train_triples.py --epochs 200

# Adjust confidence threshold
python detect_triples.py --conf 0.3
```

## 📚 Additional Resources

### Documentation
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Research Papers
- [YOLOv8: You Only Look Once v8](https://arxiv.org/abs/2309.16526)
- [Object Detection in Computer Vision](https://arxiv.org/abs/1809.02165)

### Datasets
- [Roboflow Universe](https://universe.roboflow.com/)
- [COCO Dataset](https://cocodataset.org/)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

## 🤝 Contributing

This is a capstone project, but contributions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🙏 Acknowledgments

- **Roboflow**: For providing the annotated dataset
- **Ultralytics**: For the YOLOv8 implementation
- **OpenCV**: For computer vision utilities
- **PyTorch**: For deep learning framework


