# Phase 2 Report: EfficientNet-B0 From-Scratch Baseline

**Project Title:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Phase:** Phase 2 – EfficientNet-B0 Baseline (From-Scratch Implementation and Evaluation)
**Date:** November 2026
**Primary Dataset Manifests:** `data/splits_primary/` (`train.csv`, `val.csv`, `test.csv`)
**Trained Model Checkpoint:** `models/efficientnet_b0_scratch_best.pth`
**Test Evaluation Metrics:** `reports/phase2_efficientnet_metrics.json`
**Training History Log:** `reports/phase2_training_history.csv`

---

## 1. Executive Summary

This report documents the implementation, from-scratch training, and independent test evaluation of the **EfficientNet-B0 convolutional baseline** for binary oral cancer screening.

Per strict research instructions:
- **No pretrained weights or external checkpoints were used.** The network was initialized strictly from scratch with random Gaussian weights (`weights=None`).
- **The model was trained exclusively on the approved 703-image duplicate-collapsed primary dataset** (`data/splits_primary/`), which guarantees zero scene duplication across splits.
- **The independent test set ($N = 109$) was evaluated strictly once** using the best checkpoint selected by validation loss.
- **Primary Test Performance (Positive Class = Cancer):**
  - **Accuracy:** **80.73%** (88/109 correct)
  - **Sensitivity / Recall (Cancer):** **80.00%** (60/75 true malignancies detected)
  - **Specificity (Non-Cancer):** **82.35%** (28/34 healthy/benign cases correctly identified)
  - **Precision (Cancer):** **90.91%** (60/66 predicted malignancies were true positives)
  - **F1-Score (Cancer):** **0.8511** (Macro F1: 0.7892)
  - **ROC-AUC:** **0.8912**
  - **PR-AUC (Average Precision):** **0.9176**

---

## 2. Dataset Distribution & Class Imbalance

The approved primary duplicate-collapsed dataset (`data/splits_primary/`) was used exclusively:

| Split | Cancer (Label 1) | Non-Cancer (Label 0) | Total Images | Class Proportion (C : NC) | Split Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 322 | 164 | **486** | 66.26% : 33.74% | 69.13% |
| **Validation** | 73 | 35 | **108** | 67.59% : 32.41% | 15.36% |
| **Testing** | 75 | 34 | **109** | 68.81% : 31.19% | 15.50% |
| **Total** | **470** | **233** | **703** | **66.86% : 33.14%** | **100.00%** |

### Class Imbalance Handling
The training set exhibits a ~1.96 : 1 class imbalance (322 Cancer vs. 164 Non-Cancer).
To prevent the model from defaulting to the majority malignant class, inverse class frequency weighting was applied strictly to the CrossEntropyLoss:
$$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$
- **Non-Cancer Weight ($w_0$):** $\frac{486}{2 \times 164} = \mathbf{1.4817}$
- **Cancer Weight ($w_1$):** $\frac{486}{2 \times 322} = \mathbf{0.7547}$

*Note:* Class weights were computed exclusively from training labels; zero test-set or validation-set information was used.

---

## 3. Preprocessing and Augmentation Pipeline

### 3.1 Normalization
Because the model was trained from scratch without ImageNet transfer learning, input normalization used the training-set-specific statistics computed in Phase 1:
- **Mean:** $[R = 0.6432, G = 0.4537, B = 0.4275]$
- **Standard Deviation:** $[R = 0.2189, G = 0.2118, B = 0.2090]$
- *Physiological Justification:* Centering on the oral mucosal red-dominant spectrum prevents initial positive bias on the red channel.

### 3.2 Augmentation Strategy
- **Training Set (Conservative Clinical Augmentation):**
  - Bilinear Resize to $224 \times 224$ px
  - `RandomHorizontalFlip(p=0.5)` (Preserves bilateral oral symmetry)
  - `RandomRotation(degrees=10)` (Simulates natural handheld camera/head tilt)
  - `ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0)` (Simulates lighting variation without altering diagnostic erythema/leukoplakia contrast)
  - *No vertical flips* (excluded due to anatomical asymmetry between palate and floor of mouth)
  - *No saturation/hue distortion*
- **Validation and Testing Sets (Deterministic):**
  - Bilinear Resize to $224 \times 224$ px
  - ToTensor $\rightarrow$ Normalize with primary training statistics
  - Zero random augmentation; 100% deterministic.

---

## 4. Model Architecture & Initialization Verification

### 4.1 Architecture Specification
- **Base Architecture:** EfficientNet-B0 (MBConv inverted residual blocks with Squeeze-and-Excitation).
- **Classification Head:**
  - Standard 1000-class linear projection was replaced with a binary classification head:
    $$\text{Dropout}(p=0.2) \longrightarrow \text{Linear}(\text{in\_features}=1280, \text{out\_features}=2)$$
  - Linear layer weights initialized via Kaiming normal (`fan_out`, `relu`), bias initialized to 0.0.
- **Logit Outputs:** Shape $[B, 2]$, where index 0 represents Non-Cancer and index 1 represents Cancer.

### 4.2 Parameter Count & Initialization Verification
- **Total Parameters:** **4,010,110**
- **Trainable Parameters:** **4,010,110** (100% trainable)
- **Non-trainable Parameters:** 0
- **Pretrained Weights Verification:**
  - Instantiated with `torchvision.models.efficientnet_b0(weights=None)`.
  - First convolutional layer weight distribution: $\mu = 0.001261, \sigma = 0.083520$, confirming standard random Gaussian initialization.
  - Zero external weight files, checkpoints, or pretrained embeddings were accessed.

---

## 5. Hardware and Software Environment

- **Python Version:** 3.11.9
- **PyTorch Version:** 2.6.0+cu124
- **Torchvision Version:** 0.21.0+cu124
- **Compute Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **CUDA Driver:** CUDA 12.4
- **Operating System:** Windows 11

---

## 6. Training Configuration & Hyperparameters

- **Random Seed:** `42` (Fixed across Python `random`, NumPy, PyTorch CPU, and CUDA)
- **Batch Size:** 16 (31 batches per epoch)
- **Optimizer:** `AdamW`
  - Base Learning Rate: $3 \times 10^{-4}$
  - Weight Decay: $1 \times 10^{-2}$ ($\beta_1 = 0.9, \beta_2 = 0.999$)
  - Gradient Clipping: $\text{max\_norm} = 5.0$
- **Learning Rate Scheduler:** `ReduceLROnPlateau`
  - Mode: `min` (monitoring validation loss)
  - Reduction Factor: $0.5$
  - Patience: 4 epochs
  - Minimum LR: $1 \times 10^{-6}$
- **Early Stopping:**
  - Monitored Metric: Validation Loss
  - Patience: 12 consecutive epochs without improvement
- **Maximum Epochs:** 50

---

## 7. Training Progression & Best Epoch Selection

### 7.1 Epoch-by-Epoch History Summary
Training ran for **25 epochs** before early stopping was triggered (790.0 seconds total elapsed time):

- **Epochs 1–6 ($LR = 3.0 \times 10^{-4}$):** Initial adaptation from random initialization. Initial validation loss dropped to $0.9857$ at Epoch 1, with temporary fluctuations typical of deep networks trained from scratch.
- **Epochs 7–11 ($LR = 1.5 \times 10^{-4}$):** Training accuracy rose steadily from 62.3% to 69.3%.
- **Epochs 12–18 ($LR = 7.5 \times 10^{-5}$):**
  - **Epoch 13 (BEST CHECKPOINT):**
    - **Validation Loss:** **0.92289** (Global Minimum)
    - **Validation Accuracy:** **80.56%**
    - **Validation Precision (Cancer):** **85.14%**
    - **Validation Recall (Cancer):** **86.30%**
    - **Validation F1-Score (Cancer):** **0.8571**
    - **Training Loss / Acc:** $1.5404$ / $74.49\%$
- **Epochs 19–25 ($LR = 3.75 \times 10^{-5} \rightarrow 1.88 \times 10^{-5}$):**
  - Training loss decreased further to $0.6399$ (training accuracy $81.5\% - 83.5\%$).
  - Validation loss plateaued between $2.0$ and $4.0$, indicating the onset of overfitting on the 486 training samples.
- **Early Stopping Trigger:** Epoch 25 (12 consecutive epochs without surpassing Epoch 13's validation loss).

*Training history is saved in [reports/phase2_training_history.csv](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_training_history.csv).*
*Learning curves plot is saved in [reports/phase2_learning_curves.png](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_learning_curves.png).*

---

## 8. Independent Primary Test Set Evaluation

The best checkpoint (`Epoch 13`, selected exclusively by validation loss) was loaded and evaluated **strictly once** on the independent primary test set ($N = 109$).

### 8.1 Core Classification Performance Metrics

| Metric | Measured Value | Formula / Basis |
| :--- | :---: | :--- |
| **Accuracy** | **80.73%** | $(TP + TN) / \text{Total} = (60 + 28) / 109$ |
| **Sensitivity / Recall (Cancer)** | **80.00%** | $TP / (TP + FN) = 60 / 75$ |
| **Specificity (Non-Cancer)** | **82.35%** | $TN / (TN + FP) = 28 / 34$ |
| **Precision (Cancer)** | **90.91%** | $TP / (TP + FP) = 60 / 66$ |
| **Precision (Non-Cancer)** | **65.12%** | $TN / (TN + FN) = 28 / 43$ |
| **F1-Score (Cancer, Positive Class)** | **0.8511** | $2 \cdot (P \cdot R) / (P + R) = 2 \cdot (0.9091 \cdot 0.8000) / 1.7091$ |
| **F1-Score (Non-Cancer)** | **0.7273** | $2 \cdot (0.6512 \cdot 0.8235) / 1.4747$ |
| **Macro F1-Score** | **0.7892** | Unweighted average of Cancer and Non-Cancer F1 |
| **Weighted F1-Score** | **0.8125** | Weighted average by class support |
| **ROC-AUC** | **0.8912** | Area under the Receiver Operating Characteristic curve |
| **PR-AUC (Average Precision)** | **0.9176** | Area under the Precision-Recall curve |

*Prediction Method:* Discrete predictions were generated using standard two-class argmax ($\hat{y} = \arg\max_c z_c$); no post-hoc threshold tuning on test labels was performed.

---

### 8.2 Confusion Matrix Analysis

```
                      PREDICTED
                 Non-Cancer (0)    Cancer (1)
ACTUAL
Non-Cancer (0)        TN = 28        FP =  6     │ Total Non-Cancer: 34
Cancer (1)            FN = 15        TP = 60     │ Total Cancer:     75
─────────────────────────────────────────────────
Total Predicted:          43             66      │ Total Test:      109
```

- **True Positives ($TP = 60$):** 60 out of 75 oral malignancies were correctly identified.
- **True Negatives ($TN = 28$):** 28 out of 34 benign/normal cases were correctly identified.
- **False Positives ($FP = 6$):** 6 benign cases were flagged as suspicious (17.65% false alarm rate).
- **False Negatives ($FN = 15$):** 15 cancer cases were missed (20.00% false negative rate).

*Confusion matrix heatmap is saved in [reports/phase2_confusion_matrix.png](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_confusion_matrix.png).*

---

## 9. Observations: Overfitting and Class Imbalance

1. **Impact of Class Weighting:**
   The training-set class weights ($1.4817$ for Non-Cancer vs. $0.7547$ for Cancer) successfully prevented the model from simply predicting the majority Cancer class. The model achieved a balanced specificity of **82.35%** alongside an **80.00%** sensitivity, avoiding the common pathology failure mode where all cases are flagged positive.
2. **Overfitting Dynamics on 486 Training Images:**
   Training a 4-million parameter deep convolutional network from scratch on 486 samples poses an inherent risk of memorization.
   - During early epochs, loss decreased steadily on both train and validation sets.
   - By Epoch 13, the model reached optimal generalizability (val loss $0.9229$, val acc $80.56\%$).
   - In subsequent epochs (14–25), training loss dropped further to $0.6399$ while validation loss began creeping upward to $2.2–4.0$. Early stopping effectively halted training before severe over-parameterized memorization occurred.

---

## 10. Research Integrity & Comparison with Reference Literature

### 10.1 Key Distinctions from Reference Paper
| Dimension | Reference Research Paper | Our Phase 2 Baseline Experiment |
| :--- | :--- | :--- |
| **Model Initialization** | **Pretrained on ImageNet-1k** (1.28M external images) | **Trained From Scratch** (`weights=None`, zero external weights) |
| **Dataset Configuration** | 750 images with unaddressed exact duplicates & contradictory label pair | **703 unique scenes** (duplicate-collapsed, contradictory collision excluded) |
| **Data Leakage Risk** | High (duplicate images randomly split across train and test) | **Zero** (connected-component deduplication, zero split overlap) |
| **Reported Metric Equivalence** | Paper reports ~90%+ accuracy with pretrained features | Scratch baseline achieves **80.73% Accuracy / 0.8912 ROC-AUC** |

### 10.2 Scientific Interpretation
The reference paper's reported accuracy relies heavily on transfer learning from millions of natural images and is evaluated on a dataset split that contains data leakage from duplicate copies. Our Phase 2 result of **80.73% accuracy and 0.8912 ROC-AUC from scratch** establishes an honest, rigorous, leak-free baseline. This provides the exact foundation needed to evaluate whether the Vision Transformer + Graph Attention Network (ViT+GAT) in Phase 3 brings true architectural improvement.

### 10.3 Clinical Disclaimer
This project is an academic research prototype developed for benchmarking machine learning architectures. It has not undergone clinical trials and cannot be used as an autonomous medical diagnostic tool.

---

## 11. Artifacts Generated

1. [models/efficientnet_b0_scratch_best.pth](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/models/efficientnet_b0_scratch_best.pth) – Best model checkpoint saved at Epoch 13 with full metadata.
2. [src/models/efficientnet_baseline.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/models/efficientnet_baseline.py) – Scratch EfficientNet-B0 architecture with binary head.
3. [src/train_efficientnet.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/train_efficientnet.py) – Training pipeline with class-weighted loss, scheduler, and early stopping.
4. [src/evaluate_efficientnet.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/evaluate_efficientnet.py) – Independent test evaluation pipeline.
5. [reports/phase2_efficientnet_metrics.json](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_efficientnet_metrics.json) – Complete test evaluation metrics in machine-readable JSON format.
6. [reports/phase2_training_history.csv](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_training_history.csv) – Epoch-by-epoch loss, accuracy, precision, recall, and learning rate history.
7. [reports/phase2_confusion_matrix.png](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_confusion_matrix.png) – Confusion matrix visualization.
8. [reports/phase2_learning_curves.png](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/phase2_learning_curves.png) – Training and validation loss, accuracy, and F1 curve plots.
