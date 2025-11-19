# ==========================================
# Title:  train_classifier.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# ==========================================

__author__ = "Divyansh Gupta"

"""
Training script for PCB Defect Detection models.
Supports training MLP and CNN models.
"""

import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.dataset import PCBDataset
from src.models.mlp import SimpleMLP
from src.models.cnn import CustomCNN
from src.config import MLP_INPUT_SIZE, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS


def get_transforms(model_type):
    """
    Get appropriate transforms based on model type.
    
    Args:
        model_type: 'mlp' or 'cnn'
        
    Returns:
        transforms.Compose: Transform pipeline
    """
    if model_type == 'mlp':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(MLP_INPUT_SIZE),
            transforms.ToTensor()
        ])
    elif model_type == 'cnn':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Convert predictions to binary
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Train PCB Defect Detection Model')
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn'], required=True,
                        help='Model type to train: mlp or cnn')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    transform = get_transforms(args.model)
    
    # Instantiate dataset
    print("Loading dataset...")
    dataset = PCBDataset(transform=transform)
    print(f"Total images: {len(dataset)}")
    
    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    if args.model == 'mlp':
        model = SimpleMLP().to(device)
    elif args.model == 'cnn':
        model = CustomCNN().to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    print(f"\nTraining {args.model.upper()} model...")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Epochs: {EPOCHS}\n")
    
    # Training loop
    best_val_acc = 0.0
    os.makedirs('outputs', exist_ok=True)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}")
        print(f"Val Recall: {val_rec:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"outputs/{args.model}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\n✓ Saved best model (val_acc: {val_acc:.4f}) to {model_path}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

