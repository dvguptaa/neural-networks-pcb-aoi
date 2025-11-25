# ==========================================
# Title:  convert_to_yolo.py
# Author: ECE539 Group 22
# Date:   Nov 2025
# ==========================================

"""
Convert DeepPCB annotations to YOLO format.

DeepPCB annotations are in format:
    x1, y1, x2, y2, class_id
    
YOLO format (normalized):
    class_id x_center y_center width height
    (all values normalized to 0-1)
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
RAW_DATA_PATH = "DeepPCB_Raw/PCBData"
YOLO_DATA_PATH = "data/yolo"
IMAGE_WIDTH = 640  # DeepPCB image width
IMAGE_HEIGHT = 640  # DeepPCB image height

# DeepPCB defect classes
DEFECT_CLASSES = {
    0: 'open',
    1: 'short', 
    2: 'mousebite',
    3: 'spur',
    4: 'copper',
    5: 'pinhole'
}

SEED = 42


def parse_annotation_file(anno_path):
    """
    Parse DeepPCB annotation file.
    
    Returns:
        list of (x1, y1, x2, y2, class_id) tuples
    """
    annotations = []
    
    if not os.path.exists(anno_path):
        return annotations
    
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(int, parts[:4])
                    class_id = int(parts[4])
                    annotations.append((x1, y1, x2, y2, class_id))
    
    return annotations


def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box to YOLO format (normalized center coordinates).
    
    Args:
        x1, y1, x2, y2: Pixel coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        x_center, y_center, width, height (all normalized 0-1)
    """
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Clamp to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def setup_yolo_directories():
    """Create YOLO directory structure."""
    dirs = [
        f"{YOLO_DATA_PATH}/images/train",
        f"{YOLO_DATA_PATH}/images/val",
        f"{YOLO_DATA_PATH}/images/test",
        f"{YOLO_DATA_PATH}/labels/train",
        f"{YOLO_DATA_PATH}/labels/val",
        f"{YOLO_DATA_PATH}/labels/test",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")


def find_all_samples():
    """
    Find all test images and their corresponding annotation files.
    
    DeepPCB structure:
        - Images in: group*/xxxxx/xxxxx_test.jpg
        - Annotations in: group*/xxxxx_not/xxxxx.txt
    
    Returns:
        list of (image_path, annotation_path) tuples
    """
    samples = []
    
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith('_test.jpg'):
                image_path = os.path.join(root, file)
                
                # Get the base name (e.g., "44000011" from "44000011_test.jpg")
                base_name = file.replace('_test.jpg', '')
                
                # Annotation is in the *_not folder
                # e.g., group44000/44000/44000011_test.jpg -> group44000/44000_not/44000011.txt
                parent_dir = os.path.dirname(root)
                folder_name = os.path.basename(root)
                anno_folder = f"{folder_name}_not"
                anno_path = os.path.join(parent_dir, anno_folder, f"{base_name}.txt")
                
                if os.path.exists(anno_path):
                    samples.append((image_path, anno_path))
    
    return samples


def convert_dataset():
    """Main conversion function."""
    print("=" * 60)
    print("DeepPCB to YOLO Format Converter")
    print("=" * 60)
    
    # Setup directories
    setup_yolo_directories()
    
    # Find all samples
    samples = find_all_samples()
    print(f"\nFound {len(samples)} annotated images")
    
    if len(samples) == 0:
        print("\n⚠️  No annotation files found!")
        print("Expected format: <name>.txt with bounding box annotations")
        print("Make sure DeepPCB annotations are in the data folder")
        return
    
    # Shuffle and split
    random.seed(SEED)
    random.shuffle(samples)
    
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
    # Process each split
    splits = [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]
    
    total_annotations = 0
    
    for split_name, split_samples in splits:
        print(f"\nProcessing {split_name} split...")
        
        for idx, (img_path, anno_path) in enumerate(split_samples):
            # Copy image
            img_name = f"{split_name}_{idx:04d}.jpg"
            dst_img = f"{YOLO_DATA_PATH}/images/{split_name}/{img_name}"
            shutil.copy(img_path, dst_img)
            
            # Convert annotations
            annotations = parse_annotation_file(anno_path)
            
            # Write YOLO format labels
            label_name = img_name.replace('.jpg', '.txt')
            dst_label = f"{YOLO_DATA_PATH}/labels/{split_name}/{label_name}"
            
            with open(dst_label, 'w') as f:
                for x1, y1, x2, y2, class_id in annotations:
                    x_c, y_c, w, h = convert_to_yolo_format(
                        x1, y1, x2, y2, IMAGE_WIDTH, IMAGE_HEIGHT
                    )
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    total_annotations += 1
        
        print(f"  Processed {len(split_samples)} images")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Output directory: {YOLO_DATA_PATH}")
    
    # Create dataset.yaml
    create_dataset_yaml()


def create_dataset_yaml():
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# PCB Defect Detection Dataset
# Auto-generated by convert_to_yolo.py

path: {os.path.abspath(YOLO_DATA_PATH)}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 6

# Class names
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pinhole
"""
    
    yaml_path = f"{YOLO_DATA_PATH}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n📄 Created: {yaml_path}")


if __name__ == "__main__":
    convert_dataset()
