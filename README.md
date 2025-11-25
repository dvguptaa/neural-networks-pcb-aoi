# PCB Defect Detection using Deep Learning

## 📋 Project Overview

This project compares **four deep learning approaches** for detecting defects in Printed Circuit Board (PCB) images:

1. **MLP (Multi-Layer Perceptron)** - Simple baseline classifier
2. **Custom CNN** - 4-layer convolutional neural network built from scratch
3. **ResNet (Transfer Learning)** - Pre-trained ResNet18 fine-tuned for classification
4. **YOLO** - Pre-trained YOLOv8 fine-tuned for defect detection (localization)

**Dataset:** DeepPCB (3,001 images, 6 defect types)

---

## 🏆 Results Summary

### Classification Models (Binary: Normal vs Defect)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP | 51.5% | 52.2% | 89.2% | 68.6% |
| Custom CNN | 52.2% | 52.3% | 99.4% | 68.6% |
| **ResNet18** | **98.0%** | **98.1%** | **98.1%** | **98.1%** |

### Object Detection Model (YOLO - Localization)

| Metric | Score |
|--------|-------|
| mAP@0.5 | **92.8%** |
| mAP@0.5:0.95 | 55.7% |
| Precision | 90.5% |
| Recall | 86.9% |

### YOLO Per-Class Performance

| Defect Type | Precision | Recall | Description |
|-------------|-----------|--------|-------------|
| short | 82.2% | 94.9% | Short circuit |
| mousebite | 83.1% | 84.4% | Mouse bite defect |
| spur | 97.3% | 73.3% | Spur/protrusion |
| copper | 92.2% | 87.2% | Copper defect |
| pinhole | 97.5% | 94.7% | Pin hole |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS with M1/M2 chip (MPS support) OR Linux/Windows with CUDA

### Installation

```bash
# Clone the repository
git clone https://github.com/dvguptaa/neural-networks-pcb-aoi.git
cd neural-networks-pcb-aoi

# Switch to the development branch
git checkout fix/raj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Ensure DeepPCB dataset is placed at:
```
DeepPCB_Raw/PCBData/
```

---

## 🏋️ Training Models

### 1. Train MLP (Baseline)

```bash
python -m src.train_classifier --model mlp
```

- **Runtime:** ~5 minutes
- **Output:** `outputs/mlp_best.pth`

### 2. Train Custom CNN

```bash
# Option 1: Use GPU (MPS on Mac)
python -m src.train_classifier --model cnn

# Option 2: Use CPU (more stable if GPU has issues)
python -m src.train_classifier --model cnn --cpu
```

- **Runtime:** ~20-30 minutes
- **Output:** `outputs/cnn_best.pth`

### 3. Train ResNet (Recommended for Classification)

```bash
python -m src.train_classifier --model resnet
```

- **Runtime:** ~30-45 minutes
- **Output:** `outputs/resnet_best.pth`

### 4. Train YOLO (For Object Detection/Localization)

First, convert annotations to YOLO format:
```bash
python -m src.convert_to_yolo
```

Then train:
```bash
python -m src.train_yolo --train --model s --epochs 50
```

- **Runtime:** ~1-2 hours
- **Output:** `outputs/yolo_pcb/weights/best.pt`

To evaluate:
```bash
python -m src.train_yolo --eval
```

---

## 📁 Project Structure

```
neural-networks-pcb-aoi/
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters and settings
│   ├── dataset.py             # PCBDataset class
│   ├── train_classifier.py    # MLP/CNN/ResNet training script
│   ├── convert_to_yolo.py     # Convert annotations to YOLO format
│   ├── train_yolo.py          # YOLO training script
│   └── models/
│       ├── __init__.py
│       ├── mlp.py             # MLP architecture
│       └── cnn.py             # Custom CNN & ResNet architectures
├── DeepPCB_Raw/               # Dataset (not in git)
│   └── PCBData/
├── data/
│   └── yolo/                  # YOLO-formatted dataset
├── outputs/                   # Trained models and logs
│   ├── mlp_best.pth
│   ├── cnn_best.pth
│   ├── resnet_best.pth
│   └── yolo_pcb/
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

Key settings in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEED` | 42 | Random seed for reproducibility |
| `IMAGE_SIZE` | (224, 224) | Input size for CNN/ResNet |
| `MLP_INPUT_SIZE` | (64, 64) | Input size for MLP |
| `BATCH_SIZE` | 32 | Batch size for MLP |
| `CNN_BATCH_SIZE` | 32 | Batch size for CNN |
| `LEARNING_RATE` | 0.001 | MLP learning rate |
| `CNN_LEARNING_RATE` | 0.01 | CNN learning rate |
| `EPOCHS` | 50 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |

---

## 🏗️ Model Architectures

### MLP (Multi-Layer Perceptron)
```
Input (64×64×3 = 12,288) 
    → Linear(512) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(128) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(1) → Output
```
**Parameters:** ~6.4M

### Custom CNN
```
Input (224×224×3)
    → Conv(32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)  [112×112]
    → Conv(64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)  [56×56]
    → Conv(128) + BatchNorm + ReLU + MaxPool + Dropout(0.25) [28×28]
    → Conv(256) + BatchNorm + ReLU + MaxPool + Dropout(0.25) [14×14]
    → GlobalAvgPool → Linear(128) + Dropout(0.5) → Linear(1) → Output
```
**Parameters:** ~460K

### ResNet18 (Transfer Learning)
- Pre-trained on ImageNet
- Final FC layer replaced for binary classification
- **Parameters:** ~11M

### YOLO (YOLOv8-small)
- Pre-trained on COCO dataset
- Fine-tuned on DeepPCB for 6-class detection
- **Parameters:** ~11M

---

## 📊 Key Findings

1. **MLP Limitations:** Flattening images destroys spatial information, limiting accuracy to ~50%

2. **CNN Challenges:** Training from scratch on a small dataset (~2,400 training images) is difficult; the model struggles to learn meaningful features without pre-trained weights

3. **Transfer Learning Wins:** ResNet18 with ImageNet pre-training achieves **98% accuracy**, demonstrating the power of transfer learning for small datasets

4. **YOLO Performance:** Object detection with YOLO achieves **92.8% mAP**, enabling precise localization of defects

5. **Best Detected Defects:** Pinhole (97.5% precision) and Spur (97.3% precision) are easiest to detect

6. **Challenging Defects:** Short circuits have lower precision (82.2%) but high recall (94.9%)

---

## 👥 Team Contributions

| Member | Contributions |
|--------|--------------|
| Raj | MLP/CNN implementation, training pipeline, model debugging |
| Divyansh | Initial codebase setup, data pipeline |
| Siddhant | YOLO implementation, GitHub management |
| Anurag | Dataset preparation, evaluation |

---

## 📝 Branch Information

- **main:** Original baseline code
- **fix/raj:** Complete implementation with all models trained

### Changes in fix/raj branch:
- Added data augmentation (flip, rotation, color jitter)
- Added ImageNet normalization for transfer learning
- Fixed CNN architecture (4 conv layers + BatchNorm + Dropout)
- Added ResNet18 transfer learning option
- Added YOLO training pipeline
- Added early stopping and learning rate scheduling
- Added comprehensive logging and metrics

---

## 🔗 References

1. [DeepPCB Dataset](https://github.com/tangsanli5201/DeepPCB)
2. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
3. [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

This project is for educational purposes (ECE 539 - Fall 2025, UW-Madison).

