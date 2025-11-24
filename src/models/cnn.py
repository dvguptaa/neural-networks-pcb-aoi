# ==========================================
# Title:  cnn.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Removed sigmoid for BCEWithLogitsLoss, optimized for 256x256
# ==========================================

__author__ = "Divyansh Gupta"

"""
Custom CNN (Convolutional Neural Network) model for PCB defect detection.
"""

import torch
import torch.nn as nn
from src.config import IMAGE_SIZE


class CustomCNN(nn.Module):
    """
    Custom CNN model for binary classification.
    
    Architecture:
    - 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    - Classifier head: Flatten -> Linear(256) -> ReLU -> Dropout -> Linear(1)
    - NO sigmoid at end (use BCEWithLogitsLoss for numerical stability)
    
    With IMAGE_SIZE=(256,256):
    - After 4 pools: 256 -> 128 -> 64 -> 32 -> 16
    - Feature map: 16 x 16 x 128 = 32,768 (much more manageable!)
    """
    
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First convolutional block: 3 -> 32 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolutional block: 32 -> 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolutional block: 64 -> 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fourth convolutional block: 128 -> 128 channels (added to reduce spatial size)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size after conv blocks
        # With IMAGE_SIZE = (256, 256):
        # After 4 MaxPool2d(2,2): 256 -> 128 -> 64 -> 32 -> 16
        # Final feature map: 16 x 16 x 128 = 32,768
        flattened_size = (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16) * 128
        
        # Classifier head (smaller than before)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
            # No sigmoid - using BCEWithLogitsLoss
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 1) - raw logits (no sigmoid)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x):
        """Get probability predictions (applies sigmoid)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CustomCNN()
    print(f"CustomCNN parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Show layer-by-layer spatial dimensions
    print("\nFeature map sizes:")
    test_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    x = model.conv_block1(test_input)
    print(f"After block1: {x.shape}")
    x = model.conv_block2(x)
    print(f"After block2: {x.shape}")
    x = model.conv_block3(x)
    print(f"After block3: {x.shape}")
    x = model.conv_block4(x)
    print(f"After block4: {x.shape}")
