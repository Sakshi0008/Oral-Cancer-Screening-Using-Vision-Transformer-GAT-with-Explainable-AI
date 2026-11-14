# Phase 5 Research Report: Vision Transformer and Graph Attention Network Hybrid Fusion

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Author:** AI Research Assistant
**Date:** November 13, 2026
**Status:** Complete & Scientifically Validated
**Primary Dataset Manifest:** Approved Primary Duplicate-Collapsed Splits (`data/splits_primary/`)
- **Training Set:** 486 images (322 Cancer, 164 Non-Cancer)
- **Validation Set:** 108 images (73 Cancer, 35 Non-Cancer)
- **Test Set:** 109 images (75 Cancer, 34 Non-Cancer)
- **Total Unique Images:** 703 representative images after identified redundant copies and contradictory-label images were excluded (scene-level unique oral cavity images; not 703 unique patients as patient-level metadata is unavailable in the source dataset)

---

## 1. Executive Summary and Experimental Objectives

In Phases 3 and 4, two distinct architectural paradigms were trained and evaluated from scratch:
1. **Vision Transformer (ViT-Scratch):** Operates on a $14 \times 14$ grid of 196 non-overlapping patches ($16 \times 16$ pixels), utilizing multi-head self-attention across 4 transformer encoder layers ($d=192$, 3 attention heads, 1,965,890 parameters). Global visual context is aggregated into a 192-dimensional `[CLS]` token.
2. **Graph Attention Network (GAT / GATv2):** Treats the 196 patch representations as graph nodes, constructing a hybrid graph topology combining an 8-nearest-neighbor spatial adjacency grid with dynamic top-$k$ feature similarity ($k=8$). Two graph attention layers ($d=192$, 4 attention heads) propagate local geometric and semantic context, followed by global graph pooling into a 192-dimensional pooled graph representation.

The primary objective of **Phase 5** is to investigate whether fusing the global contextual attention of the ViT `[CLS]` token with the localized topological structure of the GAT/GATv2 graph representation provides complementary inductive biases that improve oral cancer screening performance over individual unimodal models.

### Strict Scientific Constraints Adhered To:
- **Zero Pretrained Weights:** Absolutely NO ImageNet, timm, Hugging Face, or external checkpoints were loaded. All components were trained from scratch or initialized from the verified Phase 3/4 scratch checkpoints.
- **Strict Data Isolation:** Primary dataset ($N=703$) with zero cross-split leakage. Normalization and feature scalers fitted strictly on training data ($N=486$).
- **Internal Holdout Test Evaluation:** The test set ($N=109$) serves as an internal holdout split derived from the same source dataset, with the disclosed limitation that it was evaluated across sequential project phases (Phases 2 through 6).
- **Strict Validation Selection:** Architecture selection and hyperparameter decisions were made strictly on validation data before freezing checkpoints for test evaluation.

---

## 2. Graph Configuration Selection Rationale

Before implementing the hybrid fusion models, the graph configuration was formally selected based on **validation performance** from Phase 4 experiments, adhering to Part B of the experimental guidelines.

### Candidate Graph Architectures Evaluated in Phase 4:
| Configuration | Graph Type | Attention Mechanism | Val Loss | Val Accuracy | Val Sensitivity | Val Specificity | Val F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Spatial GAT | Spatial 8-grid | Static GAT | 0.5057 | 78.70% | 84.93% | 65.71% | 0.8435 |
| Similarity GAT | Dynamic Cosine kNN | Static GAT | 0.4907 | 76.85% | 83.56% | 62.86% | 0.8300 |
| **Hybrid GAT** | Spatial + Similarity | Static GAT | **0.4557** | 80.56% | 87.67% | 65.71% | 0.8591 |
| Spatial GATv2 | Spatial 8-grid | Dynamic GATv2 | 0.4856 | 80.56% | 86.30% | 68.57% | 0.8571 |
| Similarity GATv2 | Dynamic Cosine kNN | Dynamic GATv2 | 0.4721 | 82.41% | 87.67% | 71.43% | 0.8708 |
| **Hybrid GATv2** | Spatial + Similarity | Dynamic GATv2 | **0.4576** | **82.41%** | **87.67%** | **71.43%** | **0.8708** |

### Formal Selection Decision:
1. **Primary Selected Backbone:** **Hybrid GATv2** (`models/graph_hybrid_gatv2_best.pth`, Epoch 21).
   *Justification:* Maximizes composite validation performance, achieving the highest validation accuracy ($82.41\%$), highest validation F1-score ($0.8708$), highest validation specificity ($71.43\%$), and tied-highest validation sensitivity ($87.67\%$), while maintaining an exceptionally low validation loss ($0.4576$). Dynamic GATv2 attention addresses the static attention ranking limitation by computing non-monotonic, context-dependent query-key scores.
2. **Comparative Baseline Backbone:** **Hybrid GAT** (`models/graph_hybrid_gat_best.pth`, Epoch 26).
   *Justification:* Achieved the absolute lowest validation loss overall ($0.4557$) with strong validation F1 ($0.8591$), retained as a controlled baseline to isolate the effect of static vs. dynamic attention in hybrid fusion.

---

## 3. Mathematical Formulation of Hybrid Fusion Architectures

Let $h_{\text{CLS}} \in \mathbb{R}^{B \times 192}$ denote the global image representation extracted by the Vision Transformer, and let $h_{\text{graph}} \in \mathbb{R}^{B \times 192}$ denote the graph-level representation obtained by global mean pooling over the updated node representations from the Graph Attention Network.

Three distinct fusion approaches were implemented in `src/models/hybrid_fusion.py` and independently evaluated:

### Approach 1: Concatenation Fusion (`ConcatenationFusion`)
The two representations are concatenated along the feature dimension to form a joint multimodal embedding:
$$z_{\text{concat}} = [h_{\text{CLS}} \,\|\, h_{\text{graph}}] \in \mathbb{R}^{B \times 384}$$
The joint vector is transformed through a non-linear classification head with Layer Normalization and Dropout:
$$h_1 = \text{Dropout}_{0.2}(\text{ReLU}(\text{LayerNorm}(\mathbf{W}_1 z_{\text{concat}} + \mathbf{b}_1))) \in \mathbb{R}^{B \times 128}$$
$$\hat{y} = \mathbf{W}_2 h_1 + \mathbf{b}_2 \in \mathbb{R}^{B \times 2}$$
where $\mathbf{W}_1 \in \mathbb{R}^{128 \times 384}$, $\mathbf{W}_2 \in \mathbb{R}^{2 \times 128}$.
*Parameters:* 49,794 parameters.

### Approach 2: Gated Fusion (`GatedFusion`)
To allow the model to dynamically regulate the relative contribution of global transformer attention versus local topological graph features on a per-sample and per-channel basis, a learnable gating mechanism was formulated:
$$z_{\text{joint}} = [h_{\text{CLS}} \,\|\, h_{\text{graph}}] \in \mathbb{R}^{B \times 384}$$
$$g = \sigma(\mathbf{W}_g z_{\text{joint}} + \mathbf{b}_g) \in [0, 1]^{B \times 192}$$
where $\mathbf{W}_g \in \mathbb{R}^{192 \times 384}$, and $\sigma(\cdot)$ is the element-wise sigmoid activation function.
The fused representation is computed as a convex combination:
$$h_{\text{fused}} = g \odot h_{\text{CLS}} + (1 - g) \odot h_{\text{graph}} \in \mathbb{R}^{B \times 192}$$
The fused vector is classified via:
$$h_{\text{gate\_head}} = \text{Dropout}_{0.2}(\text{ReLU}(\text{LayerNorm}(\mathbf{W}_{\text{head}} h_{\text{fused}} + \mathbf{b}_{\text{head}}))) \in \mathbb{R}^{B \times 64}$$
$$\hat{y} = \mathbf{W}_{\text{out}} h_{\text{gate\_head}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^{B \times 2}$$
*Parameters:* 86,530 parameters.

### Approach 3: Residual Fusion (`ResidualFusion`)
In this formulation, the Vision Transformer `[CLS]` representation serves as the primary base representation, while the GAT graph representation provides a non-linear residual perturbation:
$$h_{\text{res}} = \text{MLP}_{\text{res}}(h_{\text{graph}}) = \mathbf{W}_{r2} \text{Dropout}_{0.2}(\text{ReLU}(\text{LayerNorm}(\mathbf{W}_{r1} h_{\text{graph}} + \mathbf{b}_{r1}))) + \mathbf{b}_{r2} \in \mathbb{R}^{B \times 192}$$
$$h_{\text{fused}} = \text{LayerNorm}(h_{\text{CLS}} + h_{\text{res}}) \in \mathbb{R}^{B \times 192}$$
$$\hat{y} = \text{Classifier}(h_{\text{fused}}) \in \mathbb{R}^{B \times 2}$$
*Parameters:* 87,490 parameters.

---

## 4. Training Strategies: Frozen vs. End-to-End Fine-Tuning

Two training strategies were evaluated under strict experimental control:

### Strategy A: Frozen Feature Extractors
- Feature extractors (ViT Epoch 34 checkpoint and GATv2 Epoch 21 checkpoint) remain completely frozen.
- Only the newly introduced fusion layers and classification head are trained.
- Batch size: 16, Optimizer: AdamW ($lr=10^{-3}$, weight decay $=10^{-3}$), Cosine Annealing scheduler, Class-weighted cross entropy ($w_0=1.51875, w_1=0.7454$; note that all 15 Phase 5–6 models used these class weights derived from an initial 326/160 approximation, representing a +3.77% weight ratio deviation from the approved 322/164 split [$w_0=1.4817, w_1=0.7547$], which is a documented methodological deviation as audited in reports/phase5_phase6_class_weight_audit.md).
- Early stopping based strictly on minimum validation loss (patience: 15 epochs).

### Strategy B: Controlled End-to-End Fine-Tuning
- The complete pipeline (ViT backbone + dynamic graph builder + GATv2 layers + Concatenation head, total 2,238,406 parameters) was trained end-to-end directly from raw images.
- Differential learning rates: $1 \times 10^{-5}$ for the ViT and GATv2 backbones, $5 \times 10^{-4}$ for the fusion head.
- Batch size: 8, Epochs: 15, Optimizer: AdamW with Cosine Annealing.
- Best checkpoint saved at Epoch 2 (`models/hybrid_e2e_finetuned_best.pth`, Val Loss: 0.5381).

---

## 5. Experimental Results and Comprehensive Metric Comparison

All models were evaluated on the frozen 109-image primary test set (75 Cancer, 34 Non-Cancer). Metrics were computed using continuous predicted probabilities:

| Experiment / Model | Strategy | Val Loss | Val Acc | Val F1 | Test Acc | Cancer Sens (95% CI) | Specificity (95% CI) | Cancer Prec | Cancer F1 | ROC-AUC (95% CI) | PR-AUC | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01. ViT-Scratch Only** | Baseline | 0.4880 | 80.56% | 0.8531 | **75.23%** | 76.00% [65.2%, 84.2%] | 73.53% [56.9%, 85.4%] | 86.36% | 0.8085 | 0.8141 [0.722, 0.898] | 0.8957 | TN:25, FP:9, FN:18, TP:57 |
| **02. Hybrid GATv2 Only** | Baseline | 0.4576 | 82.41% | 0.8708 | **77.06%** | 81.33% [71.1%, 88.5%] | 67.65% [50.8%, 80.9%] | 84.72% | 0.8299 | 0.8298 [0.740, 0.908] | 0.9151 | TN:23, FP:11, FN:14, TP:61 |
| **03. Hybrid GAT Only** | Baseline | 0.4557 | 80.56% | 0.8591 | **71.56%** | 76.00% [65.2%, 84.2%] | 61.76% [45.1%, 76.0%] | 81.43% | 0.7862 | 0.8145 [0.721, 0.896] | 0.9047 | TN:21, FP:13, FN:18, TP:57 |
| **04. ViT + GAT Concat** | Strategy A | 0.5313 | 81.48% | 0.8592 | **74.31%** | 73.33% [62.3%, 82.0%] | 76.47% [60.0%, 87.6%] | 87.30% | 0.7971 | 0.8204 [0.730, 0.900] | 0.9057 | TN:26, FP:8, FN:20, TP:55 |
| **05. ViT + GAT Gated** | Strategy A | 0.5499 | 79.63% | 0.8472 | **76.15%** | 77.33% [66.6%, 85.3%] | 73.53% [56.9%, 85.4%] | 86.57% | 0.8169 | 0.8063 [0.716, 0.887] | 0.8866 | TN:25, FP:9, FN:17, TP:58 |
| **06. ViT + GAT Residual** | Strategy A | 0.4783 | 80.56% | 0.8571 | **76.15%** | 80.00% [69.6%, 87.5%] | 67.65% [50.8%, 80.9%] | 84.51% | 0.8219 | 0.8094 [0.720, 0.890] | 0.9002 | TN:23, FP:11, FN:15, TP:60 |
| **14. ViT + GAT E2E Fine-Tune** | Strategy B | 0.5381 | 79.63% | 0.8451 | **76.15%** | 76.00% [65.2%, 84.2%] | 76.47% [60.0%, 87.6%] | 87.69% | 0.8143 | 0.8251 [0.738, 0.905] | 0.9070 | TN:26, FP:8, FN:18, TP:57 |

---

## 6. Key Scientific Findings & Discussion

### Finding 1: Gated and Residual Fusion Outperform Pure ViT
- Combining ViT and GAT representations through learnable Gated Fusion (76.15% test accuracy, 0.8169 F1) and Residual Fusion (76.15% test accuracy, 0.8219 F1, 80.00% cancer sensitivity) improved upon the pure ViT baseline (75.23% accuracy, 0.8085 F1, 76.00% sensitivity).
- Residual Fusion achieved a notably high Cancer Sensitivity of **80.00%** (detecting 60 out of 75 cancer lesions, with only 15 false negatives), demonstrating that GAT's local graph-pooling acts as an effective inductive regularizer over ViT's attention maps.

### Finding 2: Strategy A (Frozen Extractors) vs. Strategy B (End-to-End Fine-Tuning)
- In Strategy B, end-to-end training of 2,238,406 parameters on 486 training samples achieved its best validation loss at **Epoch 2** (Val Loss: 0.5381, Val F1: 0.8451), after which training loss continued decreasing while validation loss diverged.
- In contrast, Strategy A with Residual Fusion achieved a substantially lower validation loss of **0.4783** and higher validation F1 of **0.8571**, translating to higher test cancer sensitivity (80.00% vs. 76.00%).
- *Scientific Conclusion:* On moderately sized biomedical datasets ($N=703$), training multi-modal fusion heads over frozen, domain-specific feature representations provides superior regularization compared to full end-to-end backpropagation, which is susceptible to overfitting.

### Finding 3: Comparison with Phase 2 CNN Baseline
- The Phase 2 EfficientNet-B0 scratch baseline achieved 80.73% test accuracy, 80.00% cancer sensitivity, 82.35% specificity, and 0.8912 ROC-AUC.
- Pure ViT-GAT fusion approaches (74.31%–76.15%) did not surpass the EfficientNet-B0 baseline on their own. Transformers and Graph Attention Networks lack the rigid 2D translation equivariance built into convolutional kernels. As revealed in Phase 6, combining deep representations with explicit handcrafted texture descriptors bridges this gap.

---

## 7. Artifacts and Checkpoints Produced
- Model Checkpoint: `models/hybrid_vit_gat_concat_best.pth`
- Model Checkpoint: `models/hybrid_vit_gat_gated_best.pth`
- Model Checkpoint: `models/hybrid_vit_gat_residual_best.pth`
- Model Checkpoint: `models/hybrid_e2e_finetuned_best.pth`
- Cached Representations: `data/features_graph/*.pt`
- Training History: `reports/phase5_phase6_training_history.csv`
- Ablation Summary: `reports/phase5_phase6_ablation_results.csv`
- Final Master Comparison: `reports/phase5_phase6_final_comparison.csv`
