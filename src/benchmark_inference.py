# ==========================================
# Title:  benchmark_inference.py
# Author: Anurag Janaswamy
# Date:   Nov 2025
# ==========================================

"""
Inference speed benchmarking script for PCB Defect Detection models.
Measures FPS (frames per second) for MLP, CNN, and YOLO models.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import statistics

from src.dataset import PCBDataset
from src.models.mlp import SimpleMLP
from src.models.cnn import CustomCNN
from src.config import MLP_INPUT_SIZE, IMAGE_SIZE, SEED
from src.train_classifier import get_device, get_transforms


def benchmark_classifier(model, dataloader, device, num_warmup=10, num_runs=100):
    """
    Benchmark inference speed for classification models (MLP/CNN).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test images
        device: torch.device
        num_warmup: Number of warm-up iterations
        num_runs: Number of runs to measure
        
    Returns:
        dict: Statistics including mean, std, min, max FPS
    """
    model.eval()
    model.to(device)
    
    # Warm-up runs
    print(f"   Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            images = images.to(device)
            _ = model(images)
    
    # Synchronize if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Actual benchmarking
    print(f"   Benchmarking ({num_runs} iterations)...")
    times = []
    total_images = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if len(times) >= num_runs:
                break
            
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Synchronize before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            # Time inference
            start_time = time.time()
            _ = model(images)
            
            # Synchronize after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate FPS for this batch
            fps = batch_size / elapsed
            times.append(fps)
            total_images += batch_size
    
    # Calculate statistics
    mean_fps = np.mean(times)
    std_fps = np.std(times)
    min_fps = np.min(times)
    max_fps = np.max(times)
    median_fps = np.median(times)
    
    return {
        'mean_fps': mean_fps,
        'std_fps': std_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'median_fps': median_fps,
        'total_images': total_images,
        'num_runs': len(times)
    }


def benchmark_yolo(model_path, test_images_dir, device, num_warmup=10, num_runs=100):
    """
    Benchmark inference speed for YOLO model.
    
    Args:
        model_path: Path to YOLO model weights
        test_images_dir: Directory containing test images
        device: torch.device
        num_warmup: Number of warm-up iterations
        num_runs: Number of runs to measure
        
    Returns:
        dict: Statistics including mean, std, min, max FPS
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Get list of test images
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    if len(test_images) == 0:
        raise ValueError(f"No images found in {test_images_dir}")
    
    # Limit to num_runs images
    test_images = test_images[:num_runs + num_warmup]
    
    # Warm-up runs
    print(f"   Warming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        img_path = str(test_images[i])
        _ = model.predict(img_path, verbose=False, device=device)
    
    # Synchronize if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Actual benchmarking
    print(f"   Benchmarking ({num_runs} iterations)...")
    times = []
    
    for i in range(num_warmup, num_warmup + num_runs):
        img_path = str(test_images[i])
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        # Time inference
        start_time = time.time()
        _ = model.predict(img_path, verbose=False, device=device)
        
        # Synchronize after inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate FPS (1 image per inference)
        fps = 1.0 / elapsed
        times.append(fps)
    
    # Calculate statistics
    mean_fps = np.mean(times)
    std_fps = np.std(times)
    min_fps = np.min(times)
    max_fps = np.max(times)
    median_fps = np.median(times)
    
    return {
        'mean_fps': mean_fps,
        'std_fps': std_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'median_fps': median_fps,
        'total_images': num_runs,
        'num_runs': len(times)
    }


def benchmark_all_models(device=None, num_warmup=10, num_runs=100):
    """
    Benchmark all three models (MLP, CNN, YOLO).
    
    Args:
        device: torch.device (if None, auto-detect)
        num_warmup: Number of warm-up iterations
        num_runs: Number of runs to measure
        
    Returns:
        dict: Results for all models
    """
    if device is None:
        device = get_device()
    
    results = {}
    
    # Setup test dataset - will create per-model datasets
    print("\n📂 Loading test dataset...")
    
    # Use 80/10/10 split (same as training)
    from torch.utils.data import random_split
    total_size = len(PCBDataset(transform=None))
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Benchmark MLP
    print("\n" + "="*60)
    print("🚀 Benchmarking MLP")
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
        
        results['mlp'] = benchmark_classifier(mlp_model, mlp_loader, device, num_warmup, num_runs)
        print(f"   ✅ MLP: {results['mlp']['mean_fps']:.2f} ± {results['mlp']['std_fps']:.2f} FPS")
    except Exception as e:
        print(f"   ❌ MLP benchmark failed: {e}")
        results['mlp'] = None
    
    # Benchmark CNN
    print("\n" + "="*60)
    print("🚀 Benchmarking CNN")
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
        
        results['cnn'] = benchmark_classifier(cnn_model, cnn_loader, device, num_warmup, num_runs)
        print(f"   ✅ CNN: {results['cnn']['mean_fps']:.2f} ± {results['cnn']['std_fps']:.2f} FPS")
    except Exception as e:
        print(f"   ❌ CNN benchmark failed: {e}")
        results['cnn'] = None
    
    # Benchmark YOLO
    print("\n" + "="*60)
    print("🚀 Benchmarking YOLO")
    print("="*60)
    try:
        yolo_model_path = "outputs/yolo_pcb/weights/best.pt"
        yolo_test_dir = "data/yolo/images/test"
        
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        if not os.path.exists(yolo_test_dir):
            raise FileNotFoundError(f"YOLO test images not found: {yolo_test_dir}")
        
        # Determine device string for YOLO
        if device.type == 'cuda':
            yolo_device = 'cuda'
        elif device.type == 'mps':
            yolo_device = 'mps'
        else:
            yolo_device = 'cpu'
        
        results['yolo'] = benchmark_yolo(yolo_model_path, yolo_test_dir, yolo_device, num_warmup, num_runs)
        print(f"   ✅ YOLO: {results['yolo']['mean_fps']:.2f} ± {results['yolo']['std_fps']:.2f} FPS")
    except Exception as e:
        print(f"   ❌ YOLO benchmark failed: {e}")
        results['yolo'] = None
    
    return results


def print_benchmark_results(results):
    """Print formatted benchmark results."""
    print("\n" + "="*60)
    print("📊 Benchmark Results Summary")
    print("="*60)
    
    print(f"\n{'Model':<10} {'Mean FPS':<12} {'Std FPS':<12} {'Min FPS':<12} {'Max FPS':<12} {'>30 FPS':<10}")
    print("-" * 70)
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is not None:
            r = results[model_name]
            meets_target = "✅" if r['mean_fps'] > 30 else "❌"
            print(f"{model_name.upper():<10} {r['mean_fps']:>10.2f}  {r['std_fps']:>10.2f}  "
                  f"{r['min_fps']:>10.2f}  {r['max_fps']:>10.2f}  {meets_target:>8}")
        else:
            print(f"{model_name.upper():<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("\n" + "="*60)


def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark inference speed for all models')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warm-up iterations')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU benchmarking')
    args = parser.parse_args()
    
    device = get_device(force_cpu=args.cpu)
    
    print("="*60)
    print("Inference Speed Benchmarking")
    print("="*60)
    print(f"Device: {device}")
    print(f"Warm-up iterations: {args.warmup}")
    print(f"Benchmark runs: {args.runs}")
    
    results = benchmark_all_models(device=device, num_warmup=args.warmup, num_runs=args.runs)
    
    print_benchmark_results(results)
    
    # Save results
    import json
    os.makedirs('outputs', exist_ok=True)
    results_file = 'outputs/benchmark_results.json'
    
    # Convert numpy types to native Python types for JSON
    json_results = {}
    for model_name, model_results in results.items():
        if model_results is not None:
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in model_results.items()
            }
        else:
            json_results[model_name] = None
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()

