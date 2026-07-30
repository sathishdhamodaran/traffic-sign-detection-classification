# GTSRB Traffic Sign Detection & Recognition
 

---

## Project Overview
This project develops a machine learning pipeline for the detection and
recognition of German traffic signs using the GTSRB dataset. Two models
were implemented and compared:
- **Custom CNN + ANN** (trained from scratch) — 99.19% test accuracy
- **EfficientNetB0** (transfer learning) — 88.70% test accuracy

The Custom CNN+ANN model exceeds the human benchmark of 98.84%
(Stallkamp et al., 2012).

---

## Requirements

### Environment
- Python 3.11
- macOS with Apple Silicon (M4 Pro) or any system with GPU support
- Conda environment recommended

### Install Dependencies
```bash
conda create -n ml_env python=3.11
conda activate ml_env

# Mac (Apple Silicon) only:
pip install tensorflow-macos tensorflow-metal

# All other systems:
pip install tensorflow

# Common dependencies:
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn kagglehub Pillow
```

---

## Dataset

Download the GTSRB dataset using kagglehub:
```python
import kagglehub
path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Dataset path:", path)
```

Or download manually from:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The dataset contains:
- 39,209 training images across 43 classes
- 12,630 official test images

---

## Configuration

Before running either notebook, update the paths in the
Configuration cell at the top of each notebook:

TRAIN_PATH = "your/path/to/GTSRB/Final_Training/Images"
TEST_PATH  = "your/path/to/test/dataset"
MODEL_PATH = "your/path/to/save/models"

---

## How to Run

### Notebook 1 — Custom CNN + ANN
**File:** GTSRB_CustomCNN_final.ipynb

This notebook trains a custom CNN + ANN architecture from scratch
with no pretrained weights.

Run cells in order:
1. Imports & setup
2. Dataset loading
3. Data generator (CLAHE + normalisation + augmentation)
4. Model architecture (Custom CNN + ANN)
5. Training — 30 epochs with early stopping (approx. 2-3 hours on GPU)
6. Training curves — best val accuracy 99.99% at epoch 26
7. Load saved model
8. Validation evaluation
9. Confusion matrix & classification report
10. Real world testing — UK traffic signs (domain gap analysis)
11. Official test set evaluation — 99.19% accuracy
12. Live demo — real-time prediction with confidence scores

To skip retraining: Go directly to step 7 (Load saved model)
and ensure best_cnn_ann.keras is available in your MODEL_PATH.

---

### Notebook 2 — EfficientNet Transfer Learning
**File:** GTSRB_Phase3.ipynb

This notebook implements EfficientNetB0 with two-stage transfer learning
using ImageNet pretrained weights.

Run cells in order:
1. Configuration & imports
2. Dataset loading & bounding box preparation
3. Model architecture — EfficientNetB0 + custom classification head
4. Data generators
5. Stage 1 training — frozen base, train head only (approx. 1 hour)
6. Stage 2 fine-tuning — top 30 layers unfrozen (approx. 1 hour)
7. Load test data
8. Official test set evaluation — 88.70% accuracy
9. Comparison chart — CNN+ANN vs EfficientNet vs Human
10. Training curves
11. Load saved model
12. Confusion matrix & classification report

To skip retraining: Go directly to step 11 (Load saved model)
and ensure efficientnet_stage2_final.keras is available in your MODEL_PATH.

---

## Key Results

Model                          | Val Accuracy | Test Accuracy | vs Human
Custom CNN + ANN (from scratch)| 99.99%       | 99.19%        | Exceeds
EfficientNet B0 (transfer)     | 97.76%       | 88.70%        | Below
Human benchmark (Stallkamp)    | —            | 98.84%        | —

---

## Preprocessing Pipeline

Both models use the following preprocessing steps:
1. Pre-preprocessing validation (corrupted file check, brightness analysis)
2. Resizing to target size using bicubic interpolation (INTER_CUBIC)
3. CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8x8)
4. Normalisation to [0, 1]
5. Stratified 80/20 train/validation split
6. Data augmentation (rotation +/-15, zoom +/-10%, shifts +/-10%, shear)
7. Class weighting (range: 0.405 to 4.342) for 10.7x imbalance ratio

---

## Saved Models

File                              | Description                    | Size
best_cnn_ann.keras                | Custom CNN+ANN best weights    | ~68 MB
efficientnet_stage2_final.keras   | EfficientNet Stage 2 weights   | ~32 MB

---

## Project Structure

G3_Code/
├── README.md
├── GTSRB_CustomCNN_final.ipynb
├── GTSRB_Phase3.ipynb
└── images/
    ├── confusion_matrix_cnn_ann.png
    ├── confusion_matrix_efficientnet.png
    ├── confusion_matrix_full.png
    ├── efficientnet_training_curves.png
    ├── cnn_ann_training_curves.png
    ├── classification_report_cnn_ann.png
    ├── model_comparison.png
    ├── live_demo_result.png
    └── prediction_preprocessing.png
