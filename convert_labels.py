#!/usr/bin/env python3
"""
Label Conversion Script for Triples Detection
Converts 3-class labels (single, double, triple) to 2-class labels (legal, illegal)
Capstone Project - 2-Wheeler Triples Detection
"""

import os
import glob
from pathlib import Path

def convert_labels(input_dir, output_dir=None):
    """
    Convert 3-class YOLO labels to 2-class labels
    
    Args:
        input_dir (str): Directory containing label files
        output_dir (str): Directory to save converted labels (if None, overwrites original)
    
    Conversion mapping:
    - Class 0 (single) + Class 1 (double) → Class 0 (legal_2_or_less)
    - Class 2 (triple) → Class 1 (illegal_3_or_more)
    """
    
    if output_dir is None:
        output_dir = input_dir
        print(f"⚠️  Converting labels in place: {input_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Converting labels from {input_dir} to {output_dir}")
    
    # Find all label files
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"🔍 Found {len(label_files)} label files")
    
    converted_count = 0
    error_count = 0
    
    for label_file in label_files:
        try:
            # Read original labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    # Convert class IDs
                    if class_id == 0:  # Single → Legal
                        new_class_id = 0
                    elif class_id == 1:  # Double → Legal
                        new_class_id = 0
                    elif class_id == 2:  # Triple → Illegal
                        new_class_id = 1
                    else:
                        print(f"⚠️  Unknown class ID {class_id} in {label_file}")
                        continue
                    
                    # Create new line with converted class ID
                    parts[0] = str(new_class_id)
                    converted_lines.append(" ".join(parts))
            
            # Write converted labels
            output_file = os.path.join(output_dir, os.path.basename(label_file))
            with open(output_file, 'w') as f:
                f.write("\n".join(converted_lines))
            
            converted_count += 1
            
        except Exception as e:
            print(f"❌ Error converting {label_file}: {str(e)}")
            error_count += 1
    
    print(f"\n✅ Conversion completed!")
    print(f"   Converted: {converted_count} files")
    print(f"   Errors: {error_count} files")
    
    if output_dir != input_dir:
        print(f"   Original labels preserved in: {input_dir}")
        print(f"   Converted labels saved to: {output_dir}")

def main():
    """Main function to convert all dataset splits"""
    
    print("🔄 Triples Detection - Label Conversion Tool")
    print("=" * 50)
    print("Converting 3-class labels to 2-class labels:")
    print("   Class 0 (single) + Class 1 (double) → Class 0 (legal_2_or_less)")
    print("   Class 2 (triple) → Class 1 (illegal_3_or_more)")
    print("=" * 50)
    
    # Convert each dataset split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = f"{split}/labels"
        if os.path.exists(labels_dir):
            print(f"\n🔄 Converting {split} labels...")
            convert_labels(labels_dir)
        else:
            print(f"⚠️  {split} labels directory not found: {labels_dir}")
    
    print("\n🎉 All conversions completed!")
    print("\nNext steps:")
    print("1. Verify the converted labels")
    print("2. Retrain your model: python train_triples.py --epochs 100 --batch-size 8")
    print("3. Test detection: python detect_triples.py --model best.pt --source test_image.jpg")

if __name__ == "__main__":
    main()

