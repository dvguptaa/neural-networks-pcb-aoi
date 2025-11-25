# ==========================================
# Title:  config.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Optimized settings for custom CNN training
# ==========================================

__author__ = "Divyansh Gupta"

"""
Configuration file for PCB Defect Detection project.
"""

# Random seed for reproducibility
SEED = 42

# Data paths
RAW_DATA_PATH = "DeepPCB_Raw/PCBData"
PROCESSED_DATA_PATH = "data/processed"

# Image configurations
IMAGE_SIZE = (224, 224)   # For CNN
MLP_INPUT_SIZE = (64, 64)  # For MLP

# Training hyperparameters
BATCH_SIZE = 32           # For MLP
CNN_BATCH_SIZE = 32       # Larger batch for stable BatchNorm

LEARNING_RATE = 0.001     # For MLP (Adam)
CNN_LEARNING_RATE = 0.01  # For CNN (SGD with momentum)

EPOCHS = 50
PATIENCE = 10
