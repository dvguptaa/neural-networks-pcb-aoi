# ==========================================
# Title:  evaluate_all.py
# Author: Anurag Janaswamy
# Date:   Nov 2025
# ==========================================

"""
Comprehensive evaluation script for all PCB Defect Detection models.
Collects accuracy, precision, recall, F1, mAP, FPS, and model size metrics.
"""

import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dataset import PCBDataset
from src.models.mlp import SimpleMLP
from src.models.cnn import CustomCNN
from src.config import MLP_INPUT_SIZE, IMAGE_SIZE, SEED
from src.train_classifier import get_device, get_transforms
from src.benchmark_inference import benchmark_classifier, benchmark_yolo
from src.measure_model_size import (
    measure_mlp_size, measure_cnn_size, measure_yolo_size,
    get_model_file_size_mb
)


def evaluate_classifier(model, test_loader, device):
    """
    Evaluate classification model (MLP/CNN) on test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader with test data
        device: torch.device
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def evaluate_yolo_model():
    """
    Evaluate YOLO model on test set.
    
    Returns:
        dict: Evaluation metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    
    weights_path = "outputs/yolo_pcb/weights/best.pt"
    
    if not os.path.exists(weights_path):
        return None
    
    model = YOLO(weights_path)
    
    # Evaluate on test set
    results = model.val(
        data="data/yolo/dataset.yaml",
        split='test',
        verbose=False,
    )
    
    return {
        'map50': float(results.box.map50),
        'map50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr)
    }


def evaluate_all_models(device=None, benchmark_runs=50):
    """
    Evaluate all models and collect comprehensive metrics.
    
    Args:
        device: torch.device (if None, auto-detect)
        benchmark_runs: Number of runs for FPS benchmarking
        
    Returns:
        dict: Complete evaluation results
    """
    if device is None:
        device = get_device()
    
    print("="*60)
    print("Comprehensive Model Evaluation")
    print("="*60)
    print(f"Device: {device}\n")
    
    results = {}
    
    # Setup test dataset
    print("📂 Loading test dataset...")
    
    # Use 80/10/10 split (same as training)
    # We'll create test datasets with proper transforms for each model
    total_size = len(PCBDataset(transform=None))
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Evaluate MLP
    print("\n" + "="*60)
    print("📊 Evaluating MLP")
    print("="*60)
    try:
        mlp_model = SimpleMLP()
        mlp_checkpoint = torch.load("outputs/mlp_best.pth", map_location=device)
        mlp_model.load_state_dict(mlp_checkpoint['model_state_dict'])
        
        mlp_transform = get_transforms('mlp', train=False)
        mlp_full_dataset = PCBDataset(transform=mlp_transform)
        _, _, mlp_test_dataset = random_split(
            mlp_full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        mlp_loader = DataLoader(mlp_test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Classification metrics
        mlp_metrics = evaluate_classifier(mlp_model, mlp_loader, device)
        results['mlp'] = mlp_metrics
        
        # FPS benchmarking
        print("   Benchmarking inference speed...")
        mlp_fps = benchmark_classifier(mlp_model, mlp_loader, device, num_warmup=10, num_runs=benchmark_runs)
        results['mlp']['fps_mean'] = float(mlp_fps['mean_fps'])
        results['mlp']['fps_std'] = float(mlp_fps['std_fps'])
        
        # Model size
        print("   Measuring model size...")
        mlp_size = measure_mlp_size()
        if mlp_size:
            results['mlp']['parameters'] = mlp_size['parameters']
            results['mlp']['file_size_mb'] = mlp_size['file_size_mb']
            results['mlp']['memory_footprint_mb'] = mlp_size['memory_footprint_mb']
        
        print(f"   ✅ MLP: Accuracy={mlp_metrics['accuracy']:.4f}, FPS={mlp_fps['mean_fps']:.2f}")
    except Exception as e:
        print(f"   ❌ MLP evaluation failed: {e}")
        results['mlp'] = None
    
    # Evaluate CNN
    print("\n" + "="*60)
    print("📊 Evaluating CNN")
    print("="*60)
    try:
        cnn_model = CustomCNN()
        cnn_checkpoint = torch.load("outputs/cnn_best.pth", map_location=device)
        cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
        
        cnn_transform = get_transforms('cnn', train=False)
        cnn_full_dataset = PCBDataset(transform=cnn_transform)
        _, _, cnn_test_dataset = random_split(
            cnn_full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        cnn_loader = DataLoader(cnn_test_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Classification metrics
        cnn_metrics = evaluate_classifier(cnn_model, cnn_loader, device)
        results['cnn'] = cnn_metrics
        
        # FPS benchmarking
        print("   Benchmarking inference speed...")
        cnn_fps = benchmark_classifier(cnn_model, cnn_loader, device, num_warmup=10, num_runs=benchmark_runs)
        results['cnn']['fps_mean'] = float(cnn_fps['mean_fps'])
        results['cnn']['fps_std'] = float(cnn_fps['std_fps'])
        
        # Model size
        print("   Measuring model size...")
        cnn_size = measure_cnn_size()
        if cnn_size:
            results['cnn']['parameters'] = cnn_size['parameters']
            results['cnn']['file_size_mb'] = cnn_size['file_size_mb']
            results['cnn']['memory_footprint_mb'] = cnn_size['memory_footprint_mb']
        
        print(f"   ✅ CNN: Accuracy={cnn_metrics['accuracy']:.4f}, FPS={cnn_fps['mean_fps']:.2f}")
    except Exception as e:
        print(f"   ❌ CNN evaluation failed: {e}")
        results['cnn'] = None
    
    # Evaluate YOLO
    print("\n" + "="*60)
    print("📊 Evaluating YOLO")
    print("="*60)
    try:
        yolo_metrics = evaluate_yolo_model()
        if yolo_metrics:
            results['yolo'] = yolo_metrics
            
            # FPS benchmarking
            print("   Benchmarking inference speed...")
            yolo_model_path = "outputs/yolo_pcb/weights/best.pt"
            yolo_test_dir = "data/yolo/images/test"
            
            if device.type == 'cuda':
                yolo_device = 'cuda'
            elif device.type == 'mps':
                yolo_device = 'mps'
            else:
                yolo_device = 'cpu'
            
            yolo_fps = benchmark_yolo(yolo_model_path, yolo_test_dir, yolo_device, num_warmup=10, num_runs=benchmark_runs)
            results['yolo']['fps_mean'] = float(yolo_fps['mean_fps'])
            results['yolo']['fps_std'] = float(yolo_fps['std_fps'])
            
            # Model size
            print("   Measuring model size...")
            yolo_size = measure_yolo_size()
            if yolo_size:
                results['yolo']['parameters'] = yolo_size['parameters']
                results['yolo']['file_size_mb'] = yolo_size['file_size_mb']
                if yolo_size.get('memory_footprint_mb'):
                    results['yolo']['memory_footprint_mb'] = yolo_size['memory_footprint_mb']
            
            print(f"   ✅ YOLO: mAP@0.5={yolo_metrics['map50']:.4f}, FPS={yolo_fps['mean_fps']:.2f}")
        else:
            results['yolo'] = None
    except Exception as e:
        print(f"   ❌ YOLO evaluation failed: {e}")
        results['yolo'] = None
    
    return results


def generate_comparison_table(results):
    """
    Generate a comparison table from evaluation results.
    
    Args:
        results: dict with evaluation results
        
    Returns:
        pandas.DataFrame: Comparison table
    """
    rows = []
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        row = {'Model': model_name.upper()}
        
        # Classification metrics (MLP/CNN)
        if 'accuracy' in r:
            row['Accuracy'] = f"{r['accuracy']:.4f}"
            row['Precision'] = f"{r['precision']:.4f}"
            row['Recall'] = f"{r['recall']:.4f}"
            row['F1-Score'] = f"{r['f1_score']:.4f}"
            row['mAP@0.5'] = "N/A"
            row['mAP@0.5:0.95'] = "N/A"
        # Detection metrics (YOLO)
        elif 'map50' in r:
            row['Accuracy'] = "N/A"
            row['Precision'] = f"{r['precision']:.4f}"
            row['Recall'] = f"{r['recall']:.4f}"
            row['F1-Score'] = "N/A"
            row['mAP@0.5'] = f"{r['map50']:.4f}"
            row['mAP@0.5:0.95'] = f"{r['map50_95']:.4f}"
        
        # FPS
        if 'fps_mean' in r:
            row['FPS (mean)'] = f"{r['fps_mean']:.2f}"
            row['FPS (std)'] = f"{r['fps_std']:.2f}"
        else:
            row['FPS (mean)'] = "N/A"
            row['FPS (std)'] = "N/A"
        
        # Model size
        if 'parameters' in r:
            row['Parameters'] = f"{r['parameters']:,}"
        else:
            row['Parameters'] = "N/A"
        
        if 'file_size_mb' in r:
            row['File Size (MB)'] = f"{r['file_size_mb']:.2f}"
        else:
            row['File Size (MB)'] = "N/A"
        
        if 'memory_footprint_mb' in r:
            row['Memory (MB)'] = f"{r['memory_footprint_mb']:.2f}"
        else:
            row['Memory (MB)'] = "N/A"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_comparison_table(results):
    """Print formatted comparison table."""
    df = generate_comparison_table(results)
    
    print("\n" + "="*100)
    print("📊 Comprehensive Model Comparison")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)


def save_results(results, output_dir='outputs'):
    """Save evaluation results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        if model_results:
            json_results[model_name] = {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in model_results.items()
            }
        else:
            json_results[model_name] = None
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 JSON results saved to: {json_path}")
    
    # Save CSV
    df = generate_comparison_table(results)
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 CSV results saved to: {csv_path}")
    
    return json_path, csv_path


def main():
    """Main function to run comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of all models')
    parser.add_argument('--benchmark-runs', type=int, default=50, help='Number of FPS benchmark runs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU evaluation')
    args = parser.parse_args()
    
    device = get_device(force_cpu=args.cpu)
    
    results = evaluate_all_models(device=device, benchmark_runs=args.benchmark_runs)
    
    print_comparison_table(results)
    save_results(results)
    
    return results


if __name__ == "__main__":
    main()

