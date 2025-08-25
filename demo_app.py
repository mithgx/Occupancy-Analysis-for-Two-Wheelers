#!/usr/bin/env python3
"""
Demo Application for Triples Detection
Capstone Project - 2-Wheeler Triples Detection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from ultralytics import YOLO
import threading
import time

class TriplesDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Triples Detection Demo - Capstone Project")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.is_processing = False
        
        # Setup UI
        self.setup_ui()
        
        # Load default model if available
        self.load_default_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚗 Triples Detection on 2-Wheelers", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model selection frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model path
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, padx=(0, 5))
        
        browse_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        browse_btn.grid(row=0, column=2)
        
        load_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=0, column=3, padx=(5, 0))
        
        # Model status
        self.model_status_var = tk.StringVar(value="No model loaded")
        status_label = ttk.Label(model_frame, textvariable=self.model_status_var, 
                               foreground="red")
        status_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Detection parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(params_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.conf_var, 
                              orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, padx=(10, 0))
        conf_label = ttk.Label(params_frame, textvariable=self.conf_var)
        conf_label.grid(row=0, column=2, padx=(5, 0))
        
        # IoU threshold
        ttk.Label(params_frame, text="IoU Threshold:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.iou_var, 
                             orient=tk.HORIZONTAL, length=200)
        iou_scale.grid(row=1, column=1, padx=(10, 0), pady=(10, 0))
        iou_label = ttk.Label(params_frame, textvariable=self.iou_var)
        iou_label.grid(row=1, column=2, padx=(5, 0), pady=(10, 0))
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="10")
        image_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=400)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Configure image frame grid weights
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        # Load image button
        load_img_btn = ttk.Button(control_frame, text="📁 Load Image", command=self.load_image)
        load_img_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Detect button
        self.detect_btn = ttk.Button(control_frame, text="🔍 Detect Triples", 
                                    command=self.detect_triples, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save result button
        self.save_btn = ttk.Button(control_frame, text="💾 Save Result", 
                                  command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(control_frame, text="🗑️ Clear", command=self.clear_display)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Results text
        self.results_text = tk.Text(results_frame, height=6, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                        command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Configure results frame grid weights
        results_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                             pady=(10, 0))
        
        # Bind events
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Update labels when scales change
        self.conf_var.trace("w", lambda *args: conf_label.configure(text=f"{self.conf_var.get():.2f}"))
        self.iou_var.trace("w", lambda *args: iou_label.configure(text=f"{self.iou_var.get():.2f}"))
    
    def on_canvas_configure(self, event):
        """Handle canvas resize events"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            self.model_status_var.set("Loading model...")
            self.root.update()
            
            # Load model in a separate thread to avoid blocking UI
            def load_model_thread():
                try:
                    self.model = YOLO(model_path)
                    self.root.after(0, self.on_model_loaded, True)
                except Exception as e:
                    self.root.after(0, self.on_model_loaded, False, str(e))
            
            threading.Thread(target=load_model_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_status_var.set("Failed to load model")
    
    def on_model_loaded(self, success, error_msg=None):
        """Callback when model loading is complete"""
        if success:
            self.model_status_var.set("Model loaded successfully")
            self.model_status_var.set("Model loaded successfully")
            # Update label color
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and hasattr(child, 'cget'):
                            if child.cget('textvariable') == self.model_status_var:
                                child.configure(foreground="green")
                                break
        else:
            self.model_status_var.set(f"Failed to load model: {error_msg}")
            messagebox.showerror("Error", f"Failed to load model: {error_msg}")
    
    def load_default_model(self):
        """Try to load a default model if available"""
        # Check for common model locations
        default_paths = [
            "triples_detection/yolov8_triples/weights/best.pt",
            "triples_detection/yolov8_triples/weights/last.pt",
            "best.pt",
            "last.pt"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                self.model_path_var.set(path)
                self.load_model()
                break
    
    def load_image(self):
        """Load an image file"""
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                # Load and display image
                self.current_image_path = filename
                self.current_image = cv2.imread(filename)
                
                if self.current_image is None:
                    messagebox.showerror("Error", "Failed to load image")
                    return
                
                # Convert BGR to RGB for display
                rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                # Resize image to fit canvas
                self.display_image(rgb_image)
                
                # Enable detect button
                self.detect_btn.configure(state=tk.NORMAL)
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Image loaded: {os.path.basename(filename)}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet configured, use default size
            canvas_width, canvas_height = 600, 400
        
        # Calculate scaling factor
        img_height, img_width = image.shape[:2]
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up
        
        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized_image)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Update scroll region
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
    
    def detect_triples(self):
        """Perform triples detection on the loaded image"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if self.current_image is None:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        # Disable buttons during processing
        self.detect_btn.configure(state=tk.DISABLED)
        self.is_processing = True
        self.progress_var.set(0)
        
        # Run detection in separate thread
        def detection_thread():
            try:
                # Perform detection
                results = self.model(
                    self.current_image,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get()
                )
                
                # Process results
                detections = self.process_results(results[0])
                
                # Update UI with results
                self.root.after(0, self.on_detection_complete, detections, results[0])
                
            except Exception as e:
                self.root.after(0, self.on_detection_error, str(e))
        
        threading.Thread(target=detection_thread, daemon=True).start()
        
        # Start progress animation
        self.animate_progress()
    
    def animate_progress(self):
        """Animate progress bar during detection"""
        if self.is_processing:
            current = self.progress_var.get()
            if current < 90:
                self.progress_var.set(current + 10)
                self.root.after(100, self.animate_progress)
    
    def process_results(self, result):
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = ['legal_2_or_less', 'illegal_3_or_more'][class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def on_detection_complete(self, detections, result):
        """Handle detection completion"""
        self.is_processing = False
        self.progress_var.set(100)
        self.detect_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
        
        # Display results
        self.display_results(detections)
        
        # Visualize detections on image
        self.visualize_detections(detections)
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Detection completed!\n\n")
        
        if detections:
            for i, detection in enumerate(detections):
                self.results_text.insert(tk.END, 
                    f"Detection {i+1}:\n"
                    f"  Class: {detection['class_name']}\n"
                    f"  Confidence: {detection['confidence']:.3f}\n"
                    f"  BBox: {detection['bbox']}\n\n")
        else:
            self.results_text.insert(tk.END, "No detections found.\n")
        
        # Reset progress bar
        self.root.after(1000, lambda: self.progress_var.set(0))
    
    def on_detection_error(self, error_msg):
        """Handle detection error"""
        self.is_processing = False
        self.progress_var.set(0)
        self.detect_btn.configure(state=tk.NORMAL)
        messagebox.showerror("Detection Error", f"Detection failed: {error_msg}")
    
    def display_results(self, detections):
        """Display detection results"""
        print(f"Found {len(detections)} detections:")
        for detection in detections:
            print(f"  {detection['class_name']}: {detection['confidence']:.3f}")
    
    def visualize_detections(self, detections):
        """Visualize detections on the image"""
        if self.current_image is None:
            return
        
        # Create a copy for visualization
        vis_image = self.current_image.copy()
        
        # Colors for different classes (BGR format)
        colors = {
            'single': (0, 255, 0),    # Green
            'double': (0, 165, 255),  # Orange
            'triple': (0, 0, 255)     # Red
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color for class
            color = colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), 
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert to RGB and display
        rgb_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        self.display_image(rgb_image)
    
    def save_result(self):
        """Save the detection result"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                # Get the visualized image with detections
                # For now, save the original image (you can modify this to save with detections)
                cv2.imwrite(filename, self.current_image)
                messagebox.showinfo("Success", f"Result saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save result: {str(e)}")
    
    def clear_display(self):
        """Clear the current display"""
        self.current_image = None
        self.current_image_path = None
        self.canvas.delete("all")
        self.results_text.delete(1.0, tk.END)
        self.detect_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)

def main():
    root = tk.Tk()
    app = TriplesDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

