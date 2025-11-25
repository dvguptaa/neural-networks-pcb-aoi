# ==========================================
# Title:  cnn.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Fixed custom CNN matching proposal specs
# ==========================================

__author__ = "Divyansh Gupta"

"""
Custom CNN for PCB defect detection.
Architecture: 3-4 conv layers with BatchNorm and Dropout (as per proposal)
"""

import torch
import torch.nn as nn
from src.config import IMAGE_SIZE


class CustomCNN(nn.Module):
    """
    Custom CNN model for binary classification.
    
    Architecture (matching proposal requirements):
    - 4 convolutional layers with BatchNorm and Dropout
    - Classifier head with Dropout
    
    Input: (batch, 3, 224, 224)
    Output: (batch, 1) - logits
    """
    
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Conv Block 1: 3 -> 32 channels, 224 -> 112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Conv Block 2: 32 -> 64 channels, 112 -> 56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Conv Block 3: 64 -> 128 channels, 56 -> 28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Conv Block 4: 128 -> 256 channels, 28 -> 14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Global Average Pooling - reduces 14x14 to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 256)
        
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing CustomCNN...")
    model = CustomCNN()
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    model.eval()  # Use eval mode for testing
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print("✅ CustomCNN works correctly!")
