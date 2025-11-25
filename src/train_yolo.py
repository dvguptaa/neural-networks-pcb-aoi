# ==========================================
# Title:  train_yolo.py
# Author: ECE539 Group 22
# Date:   Nov 2025
# ==========================================

"""
Train YOLOv8 on PCB defect detection dataset.
Uses ultralytics library (already in requirements.txt).
"""

import os
import argparse
from pathlib import Path


def check_dataset():
    """Check if YOLO dataset exists."""
    yaml_path = "data/yolo/dataset.yaml"
    
    if not os.path.exists(yaml_path):
        print("❌ Dataset not found!")
        print("\nPlease run the conversion script first:")
        print("  python -m src.convert_to_yolo")
        return False
    
    print(f"✅ Found dataset config: {yaml_path}")
    return True


def train_yolo(model_size='n', epochs=50, batch_size=16, imgsz=640):
    """
    Train YOLOv8 model on PCB dataset.
    
    Args:
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed!")
        print("Run: pip install ultralytics")
        return
    
    print("=" * 60)
    print("YOLOv8 Training for PCB Defect Detection")
    print("=" * 60)
    
    if not check_dataset():
        return
    
    # Load pretrained YOLOv8 model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n📦 Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    print(f"\n⚙️  Training config:")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {imgsz}")
    
    # Train
    print("\n🚀 Starting training...")
    results = model.train(
        data="data/yolo/dataset.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=10,  # Early stopping patience
        save=True,
        project="outputs",
        name="yolo_pcb",
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Print results location
    print(f"\n📁 Results saved to: outputs/yolo_pcb/")
    print(f"   Best weights: outputs/yolo_pcb/weights/best.pt")
    print(f"   Training plots: outputs/yolo_pcb/")
    
    return results


def evaluate_yolo():
    """Evaluate trained YOLO model on test set."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed!")
        return
    
    weights_path = "outputs/yolo_pcb/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ Trained weights not found: {weights_path}")
        print("Please train the model first: python -m src.train_yolo --train")
        return
    
    print("=" * 60)
    print("YOLOv8 Evaluation on Test Set")
    print("=" * 60)
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Evaluate on test set
    results = model.val(
        data="data/yolo/dataset.yaml",
        split='test',
        verbose=True,
    )
    
    # Print metrics
    print("\n📈 Test Results:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for PCB Defect Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate trained model')
    parser.add_argument('--model', type=str, default='s', choices=['n', 's', 'm', 'l'],
                        help='Model size: n(ano), s(mall), m(edium), l(arge)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    args = parser.parse_args()
    
    if args.train:
        train_yolo(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz
        )
    
    if args.eval:
        evaluate_yolo()
    
    if not args.train and not args.eval:
        print("Usage:")
        print("  Train: python -m src.train_yolo --train")
        print("  Evaluate: python -m src.train_yolo --eval")
        print("  Both: python -m src.train_yolo --train --eval")
        print("\nOptions:")
        print("  --model {n,s,m,l}  Model size (default: s)")
        print("  --epochs N         Number of epochs (default: 50)")
        print("  --batch N          Batch size (default: 16)")


if __name__ == "__main__":
    main()
