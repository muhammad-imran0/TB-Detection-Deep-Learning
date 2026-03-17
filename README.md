# Tuberculosis Detection from Chest X-Rays Using Deep Learning

**Student ID:** 3004795  
**Module:** CN7023 — Artificial Intelligence & Machine Vision  
**University of Hertfordshire — March 2026**

---

## Overview

This project builds and compares three CNN-based models to classify chest X-ray images as **Normal** or **Tuberculosis (TB)**:

| # | Model | Type | Test Accuracy | F1-Score | AUC |
|---|-------|------|--------------|----------|-----|
| 1 | Custom CNN | Built from scratch | 95.40% | 0.9540 | 0.9926 |
| 2 | **GoogLeNet** | Transfer learning | **100.00%** | **1.0000** | **1.0000** |
| 3 | ResNet-18 | Transfer learning | 99.52% | 0.9952 | 0.9998 |

GoogLeNet achieved the best results, correctly classifying all 630 test images.

## Dataset

**Tuberculosis (TB) Chest X-ray Database** from Kaggle  
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

- 4,200 images (3,500 Normal, 700 TB)
- Imbalanced — addressed via oversampling of TB training images
- Split: 70% train / 15% validation / 15% test

> **Note:** The dataset is not included in this repository due to its size (~700 MB). Download it from Kaggle and place it in a folder called `TB_Chest_Radiography_Database/` with subfolders `Normal/` and `Tuberculosis/`.

## Repository Structure

```
TB-Detection-Deep-Learning/
├── README.md                          # This file
├── TB_Detection_CNN.m                 # Main MATLAB script (all 3 experiments)
├── 3004795_TB_Detection_Report.md     # Coursework report (Markdown)
├── TB_Detection_Flowchart.html        # Project pipeline flowchart
├── output/
│   └── MUHAMMAD_IMRAN_Final.pdf       # Published MATLAB output with all figures
└── .gitignore
```

## How to Run

### Requirements
- **MATLAB R2023b** or later (also works on MATLAB Online)
- **Deep Learning Toolbox**
- **GoogLeNet support package** (`googlenet`)
- **ResNet-18 support package** (`resnet18`)

### Steps

1. Download the dataset from Kaggle and extract it so the folder structure is:
   ```
   TB_Chest_Radiography_Database/
   ├── Normal/        (3,500 images)
   └── Tuberculosis/  (700 images)
   ```

2. Place the dataset folder in the same directory as `TB_Detection_CNN.m`

3. Open `TB_Detection_CNN.m` in MATLAB

4. Update the dataset path on **line 30** if needed:
   ```matlab
   datasetPath = fullfile(pwd, 'TB_Chest_Radiography_Database');
   ```

5. Run the script section by section (each `%%` block), or run all at once

### Expected Output
- 13 figures (accuracy curves, confusion matrices, ROC curves, comparison charts, sample predictions)
- Console output with all metrics (accuracy, precision, recall, F1, AUC)
- Training takes approximately 30–90 minutes on CPU depending on hardware

## Three Experiments

### Experiment 1 — Custom CNN (Shallow Network)
- 4 convolutional blocks (16→32→64→128 filters)
- BatchNorm + ReLU + MaxPool + Dropout (50%)
- Adam optimiser, learning rate 0.001, 20 epochs

### Experiment 2 — GoogLeNet (Transfer Learning)
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Inception modules process multiple filter sizes in parallel
- Only 5M parameters (efficient for CPU training)
- SGDM optimiser, learning rate 0.0001, 15 epochs

### Experiment 3 — ResNet-18 (Transfer Learning)
- Pretrained on ImageNet
- Residual (skip) connections prevent vanishing gradients
- 18 layers, ~11M parameters
- SGDM optimiser, learning rate 0.0001, 15 epochs

## Key Techniques

- **Oversampling** — TB training images duplicated from 490 to 2,450 to balance classes
- **Data Augmentation** — rotation (±15°), translation (±10px), horizontal flip, scale (0.9–1.1)
- **Transfer Learning** — pretrained ImageNet weights with fine-tuning (new layers learn 10× faster)
- **Early Stopping** — validation patience of 5 epochs to prevent overfitting

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score (per-class and overall)
- Confusion matrices with row/column summaries
- ROC curves and AUC values
- Training/validation accuracy and loss curves

## Reference

Rahman, T., Khandakar, A., Kadir, M.A., et al. (2020). "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization." *IEEE Access*, 8, 191586-191601.
