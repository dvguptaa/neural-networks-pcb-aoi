# ==========================================
# Title:  visualize_comparison.py
# Author: Anurag Janaswamy
# Date:   Nov 2025
# ==========================================

"""
Comparative analysis visualization script for PCB Defect Detection models.
Generates bar charts, scatter plots, and radar charts showing trade-offs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_evaluation_results(results_file='outputs/evaluation_results.json'):
    """Load evaluation results from JSON file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_accuracy_comparison(results, output_dir='outputs/comparison'):
    """Create bar chart comparing accuracy metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = []
    accuracies = []
    map50_scores = []  # For YOLO
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        models.append(model_name.upper())
        
        if 'accuracy' in r:
            accuracies.append(r['accuracy'] * 100)  # Convert to percentage
            map50_scores.append(None)
        elif 'map50' in r:
            accuracies.append(None)
            map50_scores.append(r['map50'] * 100)  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Plot accuracy bars
    acc_bars = []
    map_bars = []
    
    for i, (acc, map50) in enumerate(zip(accuracies, map50_scores)):
        if acc is not None:
            bar = ax.bar(i - width/2, acc, width, label='Accuracy' if i == 0 else '', 
                        color='#3498db', alpha=0.8)
            acc_bars.append(bar)
        if map50 is not None:
            bar = ax.bar(i + width/2, map50, width, label='mAP@0.5' if i == 0 else '', 
                        color='#e74c3c', alpha=0.8)
            map_bars.append(bar)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [acc_bars, map_bars]:
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_fps_comparison(results, output_dir='outputs/comparison'):
    """Create bar chart comparing FPS."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = []
    fps_means = []
    fps_stds = []
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        if 'fps_mean' in r:
            models.append(model_name.upper())
            fps_means.append(r['fps_mean'])
            fps_stds.append(r.get('fps_std', 0))
    
    if not models:
        print("⚠️  No FPS data available for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    bars = ax.bar(x, fps_means, yerr=fps_stds, capsize=5, 
                  color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    # Add 30 FPS threshold line
    ax.axhline(y=30, color='orange', linestyle='--', linewidth=2, 
              label='Real-time threshold (30 FPS)')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, fps_means, fps_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
               f'{mean:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fps_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_model_size_comparison(results, output_dir='outputs/comparison'):
    """Create bar chart comparing model sizes."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = []
    parameters = []
    file_sizes = []
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        if 'parameters' in r:
            models.append(model_name.upper())
            parameters.append(r['parameters'] / 1e6)  # Convert to millions
            file_sizes.append(r.get('file_size_mb', 0))
    
    if not models:
        print("⚠️  No model size data available for comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Parameters plot
    bars1 = ax1.bar(x, parameters, width, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Parameter Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, param in zip(bars1, parameters):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.2f}M',
                ha='center', va='bottom', fontsize=9)
    
    # File size plot
    bars2 = ax2.bar(x, file_sizes, width, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('File Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model File Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, size in zip(bars2, file_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_size_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_accuracy_vs_fps(results, output_dir='outputs/comparison'):
    """Create scatter plot: Accuracy vs FPS trade-off."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = []
    accuracies = []
    fps_values = []
    colors = []
    
    color_map = {'MLP': '#3498db', 'CNN': '#2ecc71', 'YOLO': '#e74c3c'}
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        
        # Get accuracy/mAP
        if 'accuracy' in r:
            acc = r['accuracy'] * 100
        elif 'map50' in r:
            acc = r['map50'] * 100
        else:
            continue
        
        # Get FPS
        if 'fps_mean' in r:
            fps = r['fps_mean']
        else:
            continue
        
        models.append(model_name.upper())
        accuracies.append(acc)
        fps_values.append(fps)
        colors.append(color_map[model_name.upper()])
    
    if not models:
        print("⚠️  No data available for accuracy vs FPS plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    for i, (model, acc, fps, color) in enumerate(zip(models, accuracies, fps_values, colors)):
        ax.scatter(fps, acc, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(model, (fps, acc), xytext=(5, 5), textcoords='offset points', 
                   fontsize=11, fontweight='bold')
    
    # Add 30 FPS threshold line
    ax.axvline(x=30, color='orange', linestyle='--', linewidth=2, 
              label='Real-time threshold (30 FPS)')
    
    ax.set_xlabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy / mAP@0.5 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Inference Speed Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_vs_fps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_metrics_radar(results, output_dir='outputs/comparison'):
    """Create radar/spider chart for multi-metric comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    model_data = {}
    
    for model_name in ['mlp', 'cnn', 'yolo']:
        if results.get(model_name) is None:
            continue
        
        r = results[model_name]
        data = {}
        
        # Normalize metrics to 0-100 scale
        if 'accuracy' in r:
            data['Accuracy'] = r['accuracy'] * 100
        elif 'map50' in r:
            data['mAP@0.5'] = r['map50'] * 100
        else:
            data['Accuracy'] = 0
        
        if 'fps_mean' in r:
            # Normalize FPS (assume max 200 FPS for scaling)
            data['FPS'] = min(r['fps_mean'] / 200 * 100, 100)
        else:
            data['FPS'] = 0
        
        if 'parameters' in r:
            # Invert parameter count (fewer is better) - normalize
            # Assume max 50M parameters
            data['Efficiency'] = max(0, 100 - (r['parameters'] / 50e6 * 100))
        else:
            data['Efficiency'] = 0
        
        if 'file_size_mb' in r:
            # Invert file size (smaller is better) - normalize
            # Assume max 100 MB
            data['Size'] = max(0, 100 - (r['file_size_mb'] / 100 * 100))
        else:
            data['Size'] = 0
        
        model_data[model_name.upper()] = data
    
    if not model_data:
        print("⚠️  No data available for radar chart")
        return
    
    # Create radar chart
    categories = list(list(model_data.values())[0].keys())
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    color_map = {'MLP': '#3498db', 'CNN': '#2ecc71', 'YOLO': '#e74c3c'}
    
    for model_name, data in model_data.items():
        values = [data[cat] for cat in categories]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
               color=color_map.get(model_name, '#000000'))
        ax.fill(angles, values, alpha=0.15, color=color_map.get(model_name, '#000000'))
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True)
    
    ax.set_title('Multi-Metric Model Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_radar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def generate_all_visualizations(results_file='outputs/evaluation_results.json', 
                                output_dir='outputs/comparison'):
    """Generate all comparison visualizations."""
    print("="*60)
    print("Generating Comparative Visualizations")
    print("="*60)
    
    # Load results
    results = load_evaluation_results(results_file)
    
    # Generate all plots
    print("\n📊 Creating visualizations...")
    plot_accuracy_comparison(results, output_dir)
    plot_fps_comparison(results, output_dir)
    plot_model_size_comparison(results, output_dir)
    plot_accuracy_vs_fps(results, output_dir)
    plot_metrics_radar(results, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")


def main():
    """Main function to generate visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comparative analysis visualizations')
    parser.add_argument('--results', type=str, default='outputs/evaluation_results.json',
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output', type=str, default='outputs/comparison',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    generate_all_visualizations(args.results, args.output)


if __name__ == "__main__":
    main()

