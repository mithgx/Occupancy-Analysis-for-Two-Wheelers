#!/usr/bin/env python3
"""
Backup Script for YOLO Labels
Creates a backup of original labels before conversion
Capstone Project - 2-Wheeler Triples Detection
"""

import os
import shutil
import glob
from datetime import datetime

def backup_labels():
    """Create backup of all label files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"labels_backup_{timestamp}"
    
    print(f"💾 Creating backup of labels: {backup_dir}")
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup each dataset split
    splits = ['train', 'valid', 'test']
    total_files = 0
    
    for split in splits:
        labels_dir = f"{split}/labels"
        if os.path.exists(labels_dir):
            # Create split directory in backup
            split_backup_dir = os.path.join(backup_dir, split, "labels")
            os.makedirs(split_backup_dir, exist_ok=True)
            
            # Copy all label files
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            for label_file in label_files:
                filename = os.path.basename(label_file)
                backup_path = os.path.join(split_backup_dir, filename)
                shutil.copy2(label_file, backup_path)
            
            print(f"   ✅ {split}: {len(label_files)} files backed up")
            total_files += len(label_files)
        else:
            print(f"   ⚠️  {split}: directory not found")
    
    print(f"\n🎉 Backup completed!")
    print(f"   Total files: {total_files}")
    print(f"   Backup location: {backup_dir}")
    print(f"\n💡 You can now safely run: python convert_labels.py")
    
    return backup_dir

if __name__ == "__main__":
    backup_labels()

