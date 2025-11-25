# ==========================================
# Title:  train_classifier.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Fixed for stable CNN training (CPU option for BatchNorm stability)
# ==========================================

__author__ = "Divyansh Gupta"

"""
Training script for PCB Defect Detection models.
Supports MLP and custom CNN (as per project proposal).
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.dataset import PCBDataset
from src.models.mlp import SimpleMLP
from src.models.cnn import CustomCNN
from src.config import (
    MLP_INPUT_SIZE, IMAGE_SIZE, BATCH_SIZE, CNN_BATCH_SIZE,
    LEARNING_RATE, CNN_LEARNING_RATE, EPOCHS, SEED, PATIENCE
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to: {seed}")


def get_device(force_cpu=False):
    """Get the best available device."""
    if force_cpu:
        print("💻 Using CPU (forced)")
        return torch.device('cpu')
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device


def get_transforms(model_type, train=True):
    """Get appropriate transforms based on model type."""
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if model_type == 'mlp':
        size = MLP_INPUT_SIZE
    else:
        size = IMAGE_SIZE
    
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize
        ])


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, warmup_epochs=5):
    """Train for one epoch with optional warmup."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Handle single sample case
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Print a simple confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}:")
    print(f"              Predicted")
    print(f"              Normal  Defect")
    if len(cm) == 2:
        print(f"Actual Normal   {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"       Defect   {cm[1][0]:5d}   {cm[1][1]:5d}")
    else:
        print(f"Warning: Unexpected confusion matrix shape: {cm.shape}")
        print(cm)


def main():
    parser = argparse.ArgumentParser(description='Train PCB Defect Detection Model')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn'], required=True,
                        help='Model type: mlp or cnn')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training (more stable for CNN with BatchNorm)')
    args = parser.parse_args()
    
    set_seed(SEED)
    
    # For CNN, recommend CPU for stable BatchNorm training
    if args.model == 'cnn' and not args.cpu:
        print("\n⚠️  Note: CNN with BatchNorm may train better with --cpu flag")
        print("   If training gets stuck, try: python -m src.train_classifier --model cnn --cpu\n")
    
    device = get_device(force_cpu=args.cpu)
    
    # Model-specific settings
    if args.model == 'mlp':
        batch_size = BATCH_SIZE
        learning_rate = LEARNING_RATE
    else:  # cnn
        batch_size = CNN_BATCH_SIZE
        learning_rate = CNN_LEARNING_RATE
    
    # Transforms
    train_transform = get_transforms(args.model, train=not args.no_augment)
    val_transform = get_transforms(args.model, train=False)
    
    print("\n📂 Loading dataset...")
    full_dataset = PCBDataset(transform=val_transform)
    print(f"Total images: {len(full_dataset)}")
    
    pos_weight = full_dataset.get_class_weights().to(device)
    print(f"📊 Positive class weight: {pos_weight.item():.4f}")
    
    # Split dataset: 80/10/10
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Training dataset with augmentation
    train_dataset_aug = PCBDataset(transform=train_transform)
    train_indices = train_dataset.indices
    train_dataset_aug = torch.utils.data.Subset(train_dataset_aug, train_indices)
    
    train_loader = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Initialize model
    if args.model == 'mlp':
        model = SimpleMLP().to(device)
    else:  # cnn
        model = CustomCNN().to(device)
    
    n_params = count_parameters(model)
    print(f"\n🧠 Model: {args.model.upper()}")
    print(f"   Parameters: {n_params:,}")
    
    # Optimizer - use SGD with momentum for CNN (more stable)
    if args.model == 'cnn':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        print(f"   Optimizer: SGD (momentum=0.9, weight_decay=1e-4)")
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"   Optimizer: Adam")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=learning_rate * 0.01)
    
    print(f"\n⚙️  Training config:")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Early stopping patience: {PATIENCE}")
    print(f"   Data augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    print(f"   Device: {device}")
    
    # Training loop
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    os.makedirs('outputs', exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            
            model_path = f"outputs/{args.model}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, model_path)
            print(f"✅ New best! Saved to {model_path}")
        else:
            epochs_without_improvement += 1
            print(f"⏳ No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        print()
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    checkpoint = torch.load(f"outputs/{args.model}_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📈 Test Results:")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall:    {test_rec:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    
    print_confusion_matrix(test_labels, test_preds, "Test Set Confusion Matrix")
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: outputs/{args.model}_best.pth")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
