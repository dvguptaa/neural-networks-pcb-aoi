# ==========================================
# Title:  mlp.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Removed sigmoid for BCEWithLogitsLoss compatibility
# ==========================================

__author__ = "Divyansh Gupta"

"""
Simple MLP (Multi-Layer Perceptron) model for PCB defect detection.
Baseline model for comparison.
"""

import torch
import torch.nn as nn
from src.config import MLP_INPUT_SIZE


class SimpleMLP(nn.Module):
    """
    Simple MLP model for binary classification.
    
    Architecture:
    - Input -> 512 -> 128 -> 1
    - ReLU activations between layers
    - NO sigmoid at end (use BCEWithLogitsLoss for numerical stability)
    """
    
    def __init__(self):
        super(SimpleMLP, self).__init__()
        
        # Calculate input features: height * width * channels (RGB)
        input_features = MLP_INPUT_SIZE[0] * MLP_INPUT_SIZE[1] * 3
        
        # Define layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Added batch norm for stability
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Added dropout for regularization
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 1) - raw logits (no sigmoid)
        """
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)  # Raw logits - no sigmoid
        
        return x
    
    def predict_proba(self, x):
        """Get probability predictions (applies sigmoid)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleMLP()
    print(f"SimpleMLP parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, MLP_INPUT_SIZE[0], MLP_INPUT_SIZE[1])
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
