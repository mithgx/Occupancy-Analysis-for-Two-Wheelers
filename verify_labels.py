#!/usr/bin/env python3
"""
Verify Converted Labels Script
Checks if labels were converted correctly to 2-class system
"""

import os
import glob

def verify_labels():
    """Verify converted labels"""
    print("🔍 Verifying converted labels...")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = f"{split}/labels"
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            print(f"\n📁 {split}: {len(label_files)} label files")
            
            # Check class distribution
            class_counts = {0: 0, 1: 0}
            total_annotations = 0
            
            for label_file in label_files[:5]:  # Check first 5 files
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id in [0, 1]:
                                    class_counts[class_id] += 1
                                    total_annotations += 1
                                else:
                                    print(f"⚠️  Invalid class ID {class_id} in {label_file}")
            
            print(f"   Class 0 (legal): {class_counts[0]} annotations")
            print(f"   Class 1 (illegal): {class_counts[1]} annotations")
            print(f"   Total: {total_annotations} annotations")
        else:
            print(f"⚠️  {split} labels directory not found")

if __name__ == "__main__":
    verify_labels()

