# ==========================================
# Title:  config.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# Updated: Added seed, reduced CNN size for M1 compatibility
# ==========================================

__author__ = "Divyansh Gupta"

"""
Configuration file for PCB Defect Detection project.
Contains all hyperparameters and path configurations.
"""

# Random seed for reproducibility
SEED = 42

# Data paths
RAW_DATA_PATH = "DeepPCB_Raw/PCBData"
PROCESSED_DATA_PATH = "data/processed"

# Image configurations
IMAGE_SIZE = (256, 256)  # Reduced from 640x640 for M1 Pro compatibility
MLP_INPUT_SIZE = (64, 64)  # Smaller size for MLP baseline

# Training hyperparameters
BATCH_SIZE = 32  # Will be overridden to 8 for CNN in trainer
CNN_BATCH_SIZE = 8  # Smaller batch for CNN to fit in 16GB unified memory
LEARNING_RATE = 0.001
EPOCHS = 30  # Increased from 20

# Early stopping
PATIENCE = 7  # Stop if no improvement for 7 epochs
