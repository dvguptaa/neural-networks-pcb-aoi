# PCB Defect Detection - Updated Code

## Changes Made

### 1. `config.py`

- Added `SEED = 42` for reproducibility
- **Reduced `IMAGE_SIZE` from (640, 640) to (256, 256)** — critical for your M1 Pro
- Added `CNN_BATCH_SIZE = 8` (smaller batch for CNN)
- Added `PATIENCE = 7` for early stopping
- Increased `EPOCHS` from 20 to 30

### 2. `dataset.py`

- Added error handling for missing/corrupt images
- Added `_log_distribution()` to show class balance
- Added `get_class_weights()` for handling imbalanced data
- Changed label dtype to `float32` for BCEWithLogitsLoss

### 3. `mlp.py`

- **Removed sigmoid from output** (now outputs raw logits)
- Added BatchNorm layers for training stability
- Added Dropout(0.3) for regularization
- Added `predict_proba()` method for inference
- Added parameter counting utility

### 4. `cnn.py`

- **Removed sigmoid from output** (now outputs raw logits)
- **Added 4th conv block** to reduce flatten size
- With 256×256 input: flatten size is now 32,768 (was 819,200!)
- Reduced classifier hidden layer from 512 to 256
- Parameter count drops from ~419M to ~8.5M

### 5. `train_classifier.py`

- Added **seed pinning** for reproducibility
- Added **MPS device detection** (Apple Silicon GPU)
- Added **ImageNet normalization**
- Added **data augmentation** (flip, rotate, color jitter)
- Switched to **BCEWithLogitsLoss** with class weighting
- Changed to **80/10/10 split** (train/val/test)
- Added **early stopping** based on F1 score
- Added **confusion matrix** printing
- Saves full checkpoint (model + optimizer state)
- Separate transforms for train (with aug) vs val/test (no aug)

---

## Installation

Replace your existing `src/` folder with this one:

```bash
# From your project root
cd /Users/raaj/Documents/CS/cs539/neural-networks-pcb-aoi

# Backup old code (optional)
mv src src_backup

# Copy new code
# (drag the src folder from this download to your project)
```

---

## Training Commands

### MLP (start here to validate pipeline)

```bash
python -m src.train_classifier --model mlp
```

Expected runtime: ~3-5 minutes on M1 Pro

### CNN (after MLP works)

```bash
python -m src.train_classifier --model cnn
```

Expected runtime: ~15-30 minutes on M1 Pro with MPS

### Disable augmentation (for debugging)

```bash
python -m src.train_classifier --model mlp --no-augment
```

---

## Expected Output

```
🎲 Random seed set to: 42
🍎 Using Apple MPS (Metal Performance Shaders)

📂 Loading dataset...

📊 Dataset Distribution:
   Normal (0): 1500 (50.0%)
   Defect (1): 1501 (50.0%)
   Total: 3001

📊 Positive class weight: 0.9993

📊 Data split:
   Train: 2400 samples
   Val:   300 samples
   Test:  301 samples

🧠 Model: MLP
   Parameters: 6,423,425

⚙️  Training config:
   Batch size: 32
   Learning rate: 0.001
   Epochs: 30
   Early stopping patience: 7
   Data augmentation: Enabled

======================================================================
Starting training...
======================================================================

Epoch 1/30
--------------------------------------------------
Train Loss: 0.6521 | Train Acc: 0.6234
Val Loss:   0.5832 | Val Acc:   0.6800
Val Precision: 0.7123 | Recall: 0.6543 | F1: 0.6821
✅ New best! Saved to outputs/mlp_best.pth
...
```

---

## Key Differences from Original

| Aspect          | Original | Updated             |
| --------------- | -------- | ------------------- |
| CNN input size  | 640×640  | 256×256             |
| CNN params      | ~419M    | ~8.5M               |
| Loss function   | BCELoss  | BCEWithLogitsLoss   |
| Class weighting | None     | Automatic           |
| Normalization   | None     | ImageNet stats      |
| Augmentation    | None     | Flip, rotate, color |
| Split           | 80/20    | 80/10/10            |
| Early stopping  | None     | Patience=7          |
| Seed            | None     | 42 (fixed)          |
| Best metric     | Accuracy | F1 Score            |

---

## Troubleshooting

### "MPS not available" but you have M1

Make sure you have PyTorch 2.0+ installed:

```bash
pip install torch torchvision --upgrade
```

### Out of memory on CNN

Reduce batch size in config.py:

```python
CNN_BATCH_SIZE = 4  # or even 2
```

### Model predicts all one class

This usually means learning rate is too high or no normalization. The updated code should fix this, but you can try:

```python
LEARNING_RATE = 0.0001  # in config.py
```
