# ==========================================
# Title:  dataset.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# ==========================================

__author__ = "Divyansh Gupta"

"""
Custom PyTorch Dataset for PCB Defect Detection.
Loads images from the DeepPCB dataset and assigns labels based on filename.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
from src.config import RAW_DATA_PATH


class PCBDataset(Dataset):
    """
    Custom Dataset for PCB defect detection.
    
    - Files ending with '_test.jpg' are labeled as class 1 (Defect)
    - Files ending with '_temp.jpg' are labeled as class 0 (Normal)
    """
    
    def __init__(self, transform=None):
        """
        Initialize the dataset.
        
        Args:
            transform: Optional transform to be applied on images
        """
        self.transform = transform
        self.data = []  # List of (file_path, label) tuples
        
        # Walk through RAW_DATA_PATH to find all image files
        for root, dirs, files in os.walk(RAW_DATA_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Filter logic: check filename endings
                if file.endswith('_test.jpg'):
                    # Class 1: Defect
                    self.data.append((file_path, 1))
                elif file.endswith('_temp.jpg'):
                    # Class 0: Normal
                    self.data.append((file_path, 0))
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (image, label) where image is a tensor and label is 0 or 1
        """
        file_path, label = self.data[idx]
        
        # Load image using cv2
        image = cv2.imread(file_path)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if it exists
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
    def __len__(self):
        """
        Return the total number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.data)


if __name__ == "__main__":
    # Test the dataset
    dataset = PCBDataset()
    print(f"Found {len(dataset)} images")
    
    # Get first image
    image, label = dataset[0]
    print(f"First image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
    print(f"First image label: {label}")

