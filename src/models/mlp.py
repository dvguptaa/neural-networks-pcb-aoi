# ==========================================
# Title:  mlp.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
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
    - Sigmoid activation at the end
    """
    
    def __init__(self):
        super(SimpleMLP, self).__init__()
        
        # Calculate input features: height * width * channels (RGB)
        input_features = MLP_INPUT_SIZE[0] * MLP_INPUT_SIZE[1] * 3
        
        # Define 3 Linear layers
        self.fc1 = nn.Linear(input_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

