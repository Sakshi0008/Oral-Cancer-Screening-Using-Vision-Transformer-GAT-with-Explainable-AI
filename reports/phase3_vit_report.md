# Phase 3 Scientific Report: Custom Vision Transformer (From-Scratch Baseline)

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Author:** Antigravity AI Research System
**Date:** November 13, 2026
**Status:** Completed and Validated

---

## 1. Executive Summary

In Phase 3, a custom Vision Transformer (ViT) was designed, implemented, verified, trained, and evaluated entirely from scratch using PyTorch. In strict adherence to institutional and faculty guidelines, **no pretrained weights, no ImageNet checkpoints, and no external embeddings** were utilized at any stage.

The model was trained exclusively on the approved primary duplicate-collapsed oral cancer screening dataset (`data/splits_primary/`, $N=703$ images, 486 train, 108 val, 109 test). Training converged with validation-loss-based early stopping at Epoch 46 (best checkpoint at **Epoch 34**, validation loss $0.4880$, validation accuracy $80.56\%$).

On the independent primary test set ($N=109$), the from-scratch ViT baseline achieved **$75.23\%$ Accuracy**, **$76.00\%$ Sensitivity** ($57/75$ cancer cases detected), **$73.53\%$ Specificity** ($25/34$ non-cancer cases detected), **$86.36\%$ Precision**, and an **$F_1$-Score of $0.8085$**, with an **ROC-AUC of $0.8141$** and **PR-AUC of $0.8957$**. Crucially, the model interface cleanly exposes both CLS token embeddings ($[B, 192]$) and all 196 patch-token representations ($[B, 196, 192]$), establishing the direct feature foundation for graph construction in Phase 4 and hybrid fusion in Phase 5.

---

## 2. Vision Transformer Architecture

### 2.1 Architectural Specifications

A tailored, lightweight Vision Transformer was constructed to mitigate overfitting when learning from scratch on a clinical dataset without external pretraining:

| Component | Parameter / Specification | Justification |
|---|---|---|
| **Input Image Size** | $224 \times 224 \times 3$ RGB | Standard clinical input dimension matching Phase 1 & 2 |
| **Patch Size ($P$)** | $16 \times 16$ pixels | Yields $14 \times 14 = 196$ spatial patches |
| **Patch Count ($N$)** | $196$ patches | Captures localized epithelial tissue texture |
| **Embedding Dimension ($D$)** | $192$ channels | Compact representation prevents over-parameterization |
| **Encoder Depth ($L$)** | $4$ Transformer blocks | Sufficient expressive capacity for scratch training |
| **Attention Heads ($H$)** | $3$ heads ($d_h = 64$ per head) | Balanced subspace diversity ($3 \times 64 = 192$) |
| **MLP Hidden Dimension** | $768$ ($4 \times$ expansion ratio) | Standard Feed-Forward expansion ratio |
| **Dropout / Attention Dropout** | $0.1$ / $0.1$ | Regularization against small-dataset overfitting |
| **Classification Classes** | $2$ ($0$: Non-Cancer, $1$: Cancer) | Binary oral cancer screening objective |
| **Total Parameters** | **$1,965,890$** ($~1.97\text{M}$) | Trainable in $<1\text{ GB}$ GPU VRAM |

### 2.2 Mathematical Pipeline

1. **Patch Partitioning & Linear Projection:**
   The input image $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ is partitioned into non-overlapping patches $x_p \in \mathbb{R}^{B \times 196 \times (16 \cdot 16 \cdot 3)}$. Projection is implemented via a 2D convolution (`kernel_size=16, stride=16`):
   $$z_0 = [x_{\text{class}}; \, x_p^1 E; \, x_p^2 E; \, \dots; \, x_p^{196} E] + E_{\text{pos}}$$
   where $E \in \mathbb{R}^{(16 \cdot 16 \cdot 3) \times 192}$, $x_{\text{class}} \in \mathbb{R}^{1 \times 1 \times 192}$ is a learnable CLS token, and $E_{\text{pos}} \in \mathbb{R}^{1 \times 197 \times 192}$ is the learnable 1D spatial positional embedding.

2. **Transformer Encoder Blocks (Pre-LayerNorm):**
   Each of the 4 encoder blocks processes the sequence of 197 tokens using pre-normalization:
   $$z'_l = \text{MHSA}(\text{LayerNorm}(z_{l-1})) + z_{l-1}, \quad l = 1, \dots, 4$$
   $$z_l = \text{MLP}(\text{LayerNorm}(z'_l)) + z'_l, \quad l = 1, \dots, 4$$
   where $\text{MHSA}$ denotes Multi-Head Self-Attention with scaled dot-product:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{64}}\right) V$$
   and $\text{MLP}$ consists of two linear transformations with GELU activation:
   $$\text{MLP}(u) = (\text{GELU}(u W_1 + b_1)) W_2 + b_2, \quad W_1 \in \mathbb{R}^{192 \times 768}, \, W_2 \in \mathbb{R}^{768 \times 192}$$

3. **Classification & Representation Interface:**
   Following the final LayerNorm, the representations are partitioned:
   - **CLS Representation:** $h_{\text{CLS}} = z_4[:, 0] \in \mathbb{R}^{B \times 192}$
   - **Patch Representations:** $H_{\text{patches}} = z_4[:, 1:] \in \mathbb{R}^{B \times 196 \times 192}$
   - **Classification Logits:** $\hat{y} = h_{\text{CLS}} W_{\text{head}} + b_{\text{head}} \in \mathbb{R}^{B \times 2}$

---

## 3. Parameter Initialization & Pretrained Independence

All weights were initialized randomly under strict zero-pretrained verification:

```python
# From src/models/vit_scratch.py lines 192-208
trunc_normal_(self.pos_embed, std=0.02)
trunc_normal_(self.cls_token, std=0.02)

for m in self.modules():
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
```

- **Truncated Normal Initialization:** Draws parameters from a Gaussian distribution with $\mu = 0.0$ and $\sigma = 0.02$, redrawing any values outside $[-2\sigma, +2\sigma] = [-0.04, +0.04]$. This bounds initial gradient variances, critical for Transformer convergence from scratch without warmup.
- **LayerNorm:** Scale initialized to $1.0$, shift to $0.0$.
- **Explicit Check:** `model.is_pretrained == False` verified programmatically. Zero weights were loaded from external sources.

---

## 4. Training Configuration & Experimental Setup

Training was executed on an NVIDIA GeForce RTX 4050 Laptop GPU under fixed seed $42$:

| Hyperparameter | Value | Description |
|---|---|---|
| **Dataset Splits** | Primary (Collapsed) | Train: 486, Val: 108, Test: 109 ($N=703$) |
| **Class Mapping** | $0$: Non-Cancer, $1$: Cancer | Identical to Phase 1 & 2 |
| **Normalization** | Dataset Primary | $\mu = [0.6432, 0.4537, 0.4275], \sigma = [0.2189, 0.2118, 0.2090]$ |
| **Augmentation (Train)** | Conservative Clinical | Horizontal flip ($p=0.5$), rotation ($\pm 10^\circ$), brightness/contrast ($\pm 5\%$) |
| **Augmentation (Val/Test)** | None | Strictly deterministic bilinear resize and normalization |
| **Loss Function** | Cross-Entropy | Balanced weights: Non-Cancer $= 1.4817$, Cancer $= 0.7547$ |
| **Optimizer** | AdamW | $\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-2}$, $\beta = (0.9, 0.999)$ |
| **Learning Rate Schedule** | ReduceLROnPlateau | Factor $= 0.5$, patience $= 4$ epochs, min $\text{lr} = 10^{-6}$ |
| **Gradient Clipping** | $\text{max\_norm} = 5.0$ | Prevents gradient explosion in self-attention |
| **Batch Size** | 16 | Fitted comfortably in $<1.2\text{ GB}$ VRAM |
| **Early Stopping** | Patience $= 12$ epochs | Monitored validation loss; halted at Epoch 46 |
| **Total Duration** | $1092.9$ seconds ($18.2$ min) | Average $23.7$ seconds/epoch |

---

## 5. Learning Dynamics & Convergence

Epoch-by-epoch training logs are saved in `reports/phase3_vit_training_history.csv` and visualized in `reports/phase3_learning_curves.png`.

| Epoch Milestone | Train Loss | Train Acc | Val Loss | Val Acc | Val $F_1$ | Dynamic Notes |
|---|---|---|---|---|---|---|
| **Epoch 1** | $0.7854$ | $54.12\%$ | $0.7180$ | $42.59\%$ | $0.4638$ | Cold initialization; stochastic token alignment |
| **Epoch 5** | $0.6698$ | $61.93\%$ | $0.6473$ | $63.89\%$ | $0.7483$ | Gradual attention head specialization |
| **Epoch 13** | $0.6122$ | $66.87\%$ | $0.5795$ | $72.22\%$ | $0.7973$ | LR reduced to $1.5 \times 10^{-4}$; first stabilization |
| **Epoch 26** | $0.4771$ | $78.60\%$ | $0.5343$ | $78.70\%$ | $0.8369$ | LR reduced to $7.5 \times 10^{-5}$; solid separation |
| **Epoch 34 (BEST)** | **$0.4765$** | **$79.63\%$** | **$0.4880$** | **$80.56\%$** | **$0.8531$** | **Global minimum validation loss; checkpoint saved** |
| **Epoch 40** | $0.4243$ | $80.45\%$ | $0.5634$ | $75.93\%$ | $0.8243$ | LR reduced to $3.75 \times 10^{-5}$; onset of overfitting |
| **Epoch 46 (Halt)**| $0.4057$ | $82.92\%$ | $0.5580$ | $79.63\%$ | $0.8514$ | 12 epochs without improvement; early stopping halts |

### Key Observations:
1. **Steady Transformer Convergence:** Unlike convolutional architectures which exhibit rapid early convergence due to built-in translational equivariance, the scratch ViT required approximately 15 epochs to develop localized self-attention filters before loss began descending steadily.
2. **Optimal Checkpoint Selection:** The model achieved its lowest validation loss ($0.4880$) and peak validation accuracy ($80.56\%$) at Epoch 34. Beyond Epoch 34, training loss continued to decline to $0.3632$, but validation loss hovered between $0.50$ and $0.58$, confirming the timely intervention of early stopping.

---

## 6. Primary Test Set Evaluation

The frozen checkpoint from Epoch 34 was evaluated once on `data/splits_primary/test.csv` ($N=109$).

### 6.1 Quantitative Test Metrics

| Metric | Score | Notes / Formulation |
|---|---|---|
| **Accuracy** | **$75.23\%$** ($82/109$) | Total correctly classified clinical images |
| **Sensitivity (Cancer Recall)** | **$76.00\%$** ($57/75$) | True Positive Rate ($TP / [TP + FN]$) |
| **Specificity (Non-Cancer Recall)** | **$73.53\%$** ($25/34$) | True Negative Rate ($TN / [TN + FP]$) |
| **Precision (Cancer)** | **$86.36\%$** ($57/66$) | Positive Predictive Value ($TP / [TP + FP]$) |
| **Precision (Non-Cancer)** | **$58.14\%$** ($25/43$) | Negative Predictive Value |
| **$F_1$-Score (Cancer)** | **$0.8085$** | Harmonic mean of Cancer Precision & Sensitivity |
| **$F_1$-Score (Non-Cancer)** | **$0.6494$** | Harmonic mean of Non-Cancer Precision & Specificity |
| **Macro $F_1$-Score** | **$0.7289$** | Unweighted average across both classes |
| **Weighted $F_1$-Score** | **$0.7589$** | Support-weighted average across both classes |
| **ROC-AUC** | **$0.8141$** | Evaluated on continuous Softmax Cancer probabilities |
| **PR-AUC (Average Precision)** | **$0.8957$** | Evaluated on continuous Softmax Cancer probabilities |

### 6.2 Test Confusion Matrix

```
                      Predicted Non-Cancer (0)    Predicted Cancer (1)
Actual Non-Cancer (0):        TN = 25                     FP =  9
Actual Cancer (1):            FN = 18                     TP = 57
```

- **True Negatives ($TN=25$):** Non-cancer oral mucosa correctly classified as benign.
- **False Positives ($FP=9$):** Benign inflammatory or normal lesions misclassified as malignant.
- **False Negatives ($FN=18$):** Malignant lesions missed by the model.
- **True Positives ($TP=57$):** Confirmed oral squamous cell carcinomas accurately detected.

---

## 7. Observed Strengths and Limitations

### Strengths:
1. **Patch-Level Feature Granularity:** Unlike global CNN pooling, the ViT outputs 196 distinct $192$-dimensional patch embeddings that preserve spatial and morphological information across the entire $14 \times 14$ oral tissue grid.
2. **High Precision:** At $86.36\%$ Cancer precision, positive screening predictions carry high diagnostic reliability.
3. **Robust Separation:** ROC-AUC ($0.8141$) and PR-AUC ($0.8957$) demonstrate well-calibrated continuous confidence ranking.

### Limitations:
1. **Lack of Inductive Bias:** Convolutional networks inherently assume locality and spatial translation invariance. Standard Vision Transformers have no intrinsic inductive bias for spatial adjacency; all spatial relationships must be learned from scratch through the positional embeddings and self-attention weights. With only 486 training images, this results in slightly lower standalone test accuracy ($75.23\%$) compared to the inductive-bias-rich EfficientNet-B0 baseline ($80.73\%$).
2. **Need for Graph Relational Modeling:** Because self-attention computes dense, unconstrained all-to-all interactions ($196 \times 196$), it can disperse attention over irrelevant background pixels. Constraining patch interactions using explicit topological graphs (spatial grids and feature-similarity graphs in Phase 4) is precisely the theoretical motivation for introducing Graph Attention Networks.
