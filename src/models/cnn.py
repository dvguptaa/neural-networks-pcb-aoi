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
    - 3 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    - Classifier head: Flatten -> Linear(512) -> ReLU -> Dropout -> Linear(1) -> Sigmoid
    """
    
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size after conv blocks
        # IMAGE_SIZE = (640, 640)
        # After 3 MaxPool2d(2,2): 640 -> 320 -> 160 -> 80
        # Final feature map: 80 x 80 x 128
        flattened_size = (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8) * 128
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Pass through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Pass through classifier
        x = self.classifier(x)
        
        return x

