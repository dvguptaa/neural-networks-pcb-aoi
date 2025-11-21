# ==========================================
# Title:  config.py
# Author: Divyansh Gupta
# Date:   18 Nov 2025
# ==========================================

__author__ = "Divyansh Gupta"

"""
Configuration file for PCB Defect Detection project.
Contains all hyperparameters and path configurations.
"""

# Data paths
RAW_DATA_PATH = "DeepPCB-master/PCBData"
PROCESSED_DATA_PATH = "data/processed"

# Image configurations
IMAGE_SIZE = (640, 640)  # Native resolution of DeepPCB
MLP_INPUT_SIZE = (64, 64)  # Smaller size for MLP baseline to avoid memory explosion

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

