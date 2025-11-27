# ==========================================
# Title:  measure_model_size.py
# Author: Anurag Janaswamy
# Date:   Nov 2025
# ==========================================

"""
Model size measurement script for PCB Defect Detection models.
Calculates parameter counts, file sizes, and memory footprints.
"""

import os
import torch
import numpy as np
from pathlib import Path

from src.models.mlp import SimpleMLP
from src.models.cnn import CustomCNN
from src.config import MLP_INPUT_SIZE, IMAGE_SIZE


def get_model_file_size_mb(model_path):
    """
    Get model file size on disk in MB.
    
    Args:
        model_path: Path to model file
        
    Returns:
        float: File size in MB, or None if file doesn't exist
    """
    if not os.path.exists(model_path):
        return None
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def get_parameter_count(model):
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_footprint(model, input_size, batch_size=1):
    """
    Estimate memory footprint during inference.
    
    Args:
        model: PyTorch model
        input_size: Tuple of (height, width) or (channels, height, width)
        batch_size: Batch size for inference
        
    Returns:
        dict: Memory estimates in MB
    """
    # Handle input size format
    if len(input_size) == 2:
        h, w = input_size
        c = 3  # RGB
    else:
        c, h, w = input_size
    
    # Estimate input tensor size (float32 = 4 bytes)
    input_size_bytes = batch_size * c * h * w * 4
    input_size_mb = input_size_bytes / (1024 * 1024)
    
    # Estimate model weights size (float32 = 4 bytes per parameter)
    num_params = get_parameter_count(model)
    weights_size_bytes = num_params * 4
    weights_size_mb = weights_size_bytes / (1024 * 1024)
    
    # Rough estimate: activations + gradients (for training)
    # For inference, we only need activations (roughly 2-3x input size for CNNs)
    # This is a rough estimate
    if 'CNN' in model.__class__.__name__ or 'ResNet' in model.__class__.__name__:
        activation_multiplier = 3.0  # CNNs have intermediate activations
    else:
        activation_multiplier = 1.5  # MLPs are simpler
    
    activation_size_mb = input_size_mb * activation_multiplier
    
    # Total inference memory (rough estimate)
    total_inference_mb = weights_size_mb + input_size_mb + activation_size_mb
    
    return {
        'input_mb': input_size_mb,
        'weights_mb': weights_size_mb,
        'activations_mb': activation_size_mb,
        'total_inference_mb': total_inference_mb,
        'parameters': num_params
    }


def measure_mlp_size():
    """Measure MLP model size."""
    results = {}
    
    # Load model
    try:
        model = SimpleMLP()
        checkpoint_path = "outputs/mlp_best.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Parameter count
            results['parameters'] = get_parameter_count(model)
            
            # File size
            results['file_size_mb'] = get_model_file_size_mb(checkpoint_path)
            
            # Memory footprint
            memory = estimate_memory_footprint(model, MLP_INPUT_SIZE, batch_size=1)
            results['memory_footprint_mb'] = memory['total_inference_mb']
            results['weights_mb'] = memory['weights_mb']
        else:
            results = None
    except Exception as e:
        print(f"Error measuring MLP size: {e}")
        results = None
    
    return results


def measure_cnn_size():
    """Measure CNN model size."""
    results = {}
    
    # Load model
    try:
        model = CustomCNN()
        checkpoint_path = "outputs/cnn_best.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Parameter count
            results['parameters'] = get_parameter_count(model)
            
            # File size
            results['file_size_mb'] = get_model_file_size_mb(checkpoint_path)
            
            # Memory footprint
            memory = estimate_memory_footprint(model, IMAGE_SIZE, batch_size=1)
            results['memory_footprint_mb'] = memory['total_inference_mb']
            results['weights_mb'] = memory['weights_mb']
        else:
            results = None
    except Exception as e:
        print(f"Error measuring CNN size: {e}")
        results = None
    
    return results


def measure_yolo_size():
    """Measure YOLO model size."""
    results = {}
    
    try:
        model_path = "outputs/yolo_pcb/weights/best.pt"
        
        if os.path.exists(model_path):
            # Load YOLO model to get parameter count
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                
                # Get parameter count from model info
                # YOLO models have a .info() method or we can count parameters
                # For YOLOv8, we can access the model attribute
                if hasattr(model, 'model'):
                    pytorch_model = model.model
                    results['parameters'] = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
                else:
                    # Fallback: estimate from file size (rough)
                    file_size_mb = get_model_file_size_mb(model_path)
                    # Rough estimate: 1MB ≈ 250K parameters (float32)
                    results['parameters'] = int(file_size_mb * 250000) if file_size_mb else None
            except:
                # If we can't load the model, estimate from file size
                file_size_mb = get_model_file_size_mb(model_path)
                results['parameters'] = int(file_size_mb * 250000) if file_size_mb else None
            
            # File size
            results['file_size_mb'] = get_model_file_size_mb(model_path)
            
            # Memory footprint (YOLO uses 640x640 input)
            # Rough estimate for YOLO
            if results.get('parameters'):
                weights_mb = (results['parameters'] * 4) / (1024 * 1024)
                # YOLO has more complex activations
                input_mb = (1 * 3 * 640 * 640 * 4) / (1024 * 1024)  # ~4.7 MB
                activation_mb = input_mb * 5.0  # YOLO has many intermediate layers
                results['memory_footprint_mb'] = weights_mb + input_mb + activation_mb
                results['weights_mb'] = weights_mb
            else:
                results['memory_footprint_mb'] = None
                results['weights_mb'] = None
        else:
            results = None
    except Exception as e:
        print(f"Error measuring YOLO size: {e}")
        results = None
    
    return results


def measure_all_models():
    """Measure sizes for all models."""
    print("="*60)
    print("Model Size Measurement")
    print("="*60)
    
    results = {}
    
    # Measure MLP
    print("\n📏 Measuring MLP...")
    results['mlp'] = measure_mlp_size()
    if results['mlp']:
        print(f"   ✅ Parameters: {results['mlp']['parameters']:,}")
        print(f"   ✅ File size: {results['mlp']['file_size_mb']:.2f} MB")
        print(f"   ✅ Memory footprint: {results['mlp']['memory_footprint_mb']:.2f} MB")
    else:
        print("   ❌ MLP model not found")
    
    # Measure CNN
    print("\n📏 Measuring CNN...")
    results['cnn'] = measure_cnn_size()
    if results['cnn']:
        print(f"   ✅ Parameters: {results['cnn']['parameters']:,}")
        print(f"   ✅ File size: {results['cnn']['file_size_mb']:.2f} MB")
        print(f"   ✅ Memory footprint: {results['cnn']['memory_footprint_mb']:.2f} MB")
    else:
        print("   ❌ CNN model not found")
    
    # Measure YOLO
    print("\n📏 Measuring YOLO...")
    results['yolo'] = measure_yolo_size()
    if results['yolo']:
        print(f"   ✅ Parameters: {results['yolo']['parameters']:,}")
        print(f"   ✅ File size: {results['yolo']['file_size_mb']:.2f} MB")
        if results['yolo']['memory_footprint_mb']:
            print(f"   ✅ Memory footprint: {results['yolo']['memory_footprint_mb']:.2f} MB")
    else:
        print("   ❌ YOLO model not found")
    
    return results


def print_size_results(results):
    """Print formatted size results."""
    print("\n" + "="*60)
    print("📊 Model Size Summary")
    print("="*60)
    
    print(f"\n{'Model':<10} {'Parameters':<15} {'File Size (MB)':<18} {'Memory (MB)':<15}")
    print("-" * 60)
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name):
            r = results[model_name]
            params = f"{r['parameters']:,}" if r.get('parameters') else "N/A"
            file_size = f"{r['file_size_mb']:.2f}" if r.get('file_size_mb') else "N/A"
            memory = f"{r['memory_footprint_mb']:.2f}" if r.get('memory_footprint_mb') else "N/A"
            print(f"{model_name.upper():<10} {params:<15} {file_size:<18} {memory:<15}")
        else:
            print(f"{model_name.upper():<10} {'N/A':<15} {'N/A':<18} {'N/A':<15}")
    
    print("\n" + "="*60)


def main():
    """Main function to measure all model sizes."""
    results = measure_all_models()
    print_size_results(results)
    
    # Save results
    import json
    os.makedirs('outputs', exist_ok=True)
    results_file = 'outputs/model_sizes.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        if model_results:
            json_results[model_name] = {
                k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
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

