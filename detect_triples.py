#!/usr/bin/env python3
"""
Triples Detection Inference Script
Capstone Project - 2-Wheeler Triples Detection
"""

import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO
from pathlib import Path
import time

class TriplesDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the Triples Detector
        
        Args:
            model_path (str): Path to trained model weights
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names
        self.class_names = ['legal_2_or_less', 'illegal_3_or_more']
        
        # Colors for visualization (BGR format)
        self.colors = {
            'legal_2_or_less': (0, 255, 0),    # Green - Legal
            'illegal_3_or_more': (0, 0, 255)   # Red - Illegal
        }
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"🎯 Classes: {self.class_names}")
    
    def detect_image(self, image_path, save_result=True, output_dir='results'):
        """
        Detect triples in a single image
        
        Args:
            image_path (str): Path to input image
            save_result (bool): Whether to save the result
            output_dir (str): Directory to save results
            
        Returns:
            dict: Detection results
        """
        print(f"🔍 Processing image: {image_path}")
        
        # Perform detection
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Process results
        detections = self._process_results(results[0])
        
        # Visualize results
        annotated_image = self._visualize_detections(image_path, detections)
        
        # Save result if requested
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{Path(image_path).name}")
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Result saved to: {output_path}")
        
        return detections, annotated_image
    
    def detect_video(self, video_path, output_path=None, show_video=True):
        """
        Detect triples in a video file
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            show_video (bool): Whether to display video during processing
        """
        print(f"🎬 Processing video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection on frame
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            detections = self._process_results(results[0])
            
            # Visualize detections
            annotated_frame = self._visualize_detections_frame(frame, detections)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame if output is specified
            if writer:
                writer.write(annotated_frame)
            
            # Display frame if requested
            if show_video:
                cv2.imshow('Triples Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"✅ Video processing completed!")
        print(f"📊 Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"⚡ Average FPS: {frame_count/elapsed_time:.2f}")
    
    def detect_realtime(self, camera_id=0, output_path=None):
        """
        Real-time triples detection using webcam
        
        Args:
            camera_id (int): Camera device ID
            output_path (str): Path to save output video
        """
        print(f"📹 Starting real-time detection with camera {camera_id}")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("🎮 Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            detections = self._process_results(results[0])
            
            # Visualize detections
            annotated_frame = self._visualize_detections_frame(frame, detections)
            
            # Add performance info
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame if output is specified
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Real-time Triples Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print("✅ Real-time detection stopped")
    
    def _process_results(self, result):
        """
        Process YOLO detection results
        
        Args:
            result: YOLO result object
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def _visualize_detections(self, image_path, detections):
        """
        Visualize detections on image
        
        Args:
            image_path (str): Path to input image
            detections (list): List of detection dictionaries
            
        Returns:
            numpy.ndarray: Annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image {image_path}")
            return None
        
        return self._visualize_detections_frame(image, detections)
    
    def _visualize_detections_frame(self, frame, detections):
        """
        Visualize detections on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection dictionaries
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for class
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

def main():
    parser = argparse.ArgumentParser(description='Triples Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--source', type=str, help='Path to image/video file or camera ID (0, 1, etc.)')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--mode', choices=['image', 'video', 'realtime'], default='image', 
                       help='Detection mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        return
    
    # Initialize detector
    detector = TriplesDetector(args.model, args.conf, args.iou)
    
    try:
        if args.mode == 'image':
            if not args.source:
                print("❌ Error: Source image path required for image mode")
                return
            detector.detect_image(args.source, save_result=True, output_dir='results')
            
        elif args.mode == 'video':
            if not args.source:
                print("❌ Error: Source video path required for video mode")
                return
            detector.detect_video(args.source, args.output, show_video=True)
            
        elif args.mode == 'realtime':
            camera_id = int(args.source) if args.source else 0
            detector.detect_realtime(camera_id, args.output)
            
    except KeyboardInterrupt:
        print("\n⏹️  Detection stopped by user")
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        raise

if __name__ == "__main__":
    main()
