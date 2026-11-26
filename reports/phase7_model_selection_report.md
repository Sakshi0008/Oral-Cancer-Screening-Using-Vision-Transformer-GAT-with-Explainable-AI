# Phase 7 Stage 2: Model Selection and Architecture Audit Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Phase:** Phase 7 (Final Research Application, Explainable AI, User-Friendly Reporting, and Frontend)  
**Stage:** Stage 2 (Model Selection and Architecture Audit)  
**Date:** November 24, 2026  
**Auditor / Assistant:** Antigravity AI Assistant  
**Selection Mandate:** Select the primary ViT-GAT hybrid model strictly using validation performance, architecture compatibility, and XAI readiness without utilizing test set data.

---

## 1. Executive Summary

In accordance with the faculty research mandate and Phase 7 specifications, the primary screening engine of the upcoming **ORALSCAN AI** platform must embody the integration of **Vision Transformers (ViT)** and **Graph Attention Networks (GAT/GATv2)**.

This report evaluates all candidate hybrid configurations saved in `models/` against seven strict technical criteria:
1. Architecture correctness.
2. Local checkpoint integrity and availability.
3. Validation-set performance (strictly using `data/splits_primary/val.csv`).
4. Training configuration and class-weight documentation.
5. Class mapping alignment (`0: NON CANCER`, `1: CANCER`).
6. Preprocessing pipeline compatibility ($224 \times 224$, dataset normalization).
7. Availability of intermediate tensors for Explainable AI (ViT self-attention rollout, GATv2 graph edge attention, and multimodal fusion attributions).

**Primary Selection Outcome:**
The **ViT + GATv2 Concatenation Hybrid Model (`models/hybrid_vit_gat_concat_best.pth`)** is formally selected as the primary clinical screening model for the Phase 7 ORALSCAN AI application, supported by the **ViT + GATv2 Gated Fusion Model (`models/hybrid_vit_gat_gated_best.pth`)** as an alternate research ablation in the frontend.

---

## 2. Evaluation of Hybrid Candidates (Validation Set Performance)

Model selection was conducted strictly using validation metrics from `data/splits_primary/val.csv` ($N=108$ samples: 73 Cancer, 35 Non-Cancer). **The test split was not accessed or used for selection or threshold tuning.**

| Candidate Architecture | Checkpoint File | Best Epoch | Validation Loss | Validation Accuracy | Validation Sensitivity | Validation Specificity | Validation Precision | Validation F1 | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ViT + GATv2 Concatenation** | `hybrid_vit_gat_concat_best.pth` | **4** | 0.5313 | **81.48%** | 83.56% | **77.14%** | **88.41%** | **0.8592** | **SELECTED PRIMARY** |
| **ViT + GATv2 Gated Fusion** | `hybrid_vit_gat_gated_best.pth` | 4 | 0.5499 | 79.63% | 83.56% | 71.43% | 85.92% | 0.8472 | Alternate Option |
| **ViT + GATv2 Residual Fusion** | `hybrid_vit_gat_residual_best.pth` | 3 | 0.4783 | 80.56% | 86.30% | 68.57% | 85.14% | 0.8571 | Alternate Option |
| **End-to-End Fine-Tuned Hybrid** | `hybrid_e2e_finetuned_best.pth` | 2 | 0.5381 | 79.63% | 82.19% | 74.29% | 86.96% | 0.8451 | Alternate Option |
| **Multi-Scale ViT + GATv2** | `multiscale_hybrid_best.pth` | 8 | 0.5152 | 78.70% | 84.93% | 65.71% | 83.78% | 0.8435 | Ablation Baseline |
| **Tri-Modal Concat (ViT+GAT+HC)** | `trimodal_vit_gat_hc_concat_best.pth` | 1 | 0.4951 | 79.63% | 86.30% | 65.71% | 84.00% | 0.8514 | Ablation Baseline |

### Why `hybrid_vit_gat_concat_best.pth` Was Selected:
1. **Highest Validation Accuracy (81.48%):** Outperforms all other hybrid architectures on the validation cohort.
2. **Highest Validation Specificity (77.14%):** Maximizes the identification of true non-cancer oral tissues, significantly minimizing false-positive screening alarms.
3. **Highest Validation Precision (88.41%) and F1 (0.8592):** Yields the most reliable positive predictive value among all hybrid candidates.
4. **Structural Decoupling for XAI:** Because the representation fusion operates on a direct concatenation $[h_{\text{cls}} \parallel h_{\text{graph}}] \in \mathbb{R}^{384}$, the relative influence of global visual attention versus localized relational graph topology can be cleanly attributed and visualized.

---

## 3. Candidate Architectural Details & Pipeline Compatibility

The selected primary hybrid model operates through a coordinated three-stage pipeline:

```text
Input Image [1, 3, 224, 224] (RGB, Dataset Normalized)
  │
  ├──► [Stage 1] Vision Transformer Scratch Backbone (4 layers, 3 heads, d=192)
  │      ├── Global CLS Token: [1, 192]
  │      └── Patch Tokens:      [1, 196, 192] (14 x 14 grid of 16x16 patches)
  │             │
  │             └──► [Stage 2] Graph Construction & GATv2 Backbone
  │                    ├── Spatial 8-NN Grid Adjacency (Chebyshev metric)
  │                    ├── Feature Similarity 8-NN Adjacency (Cosine metric)
  │                    ├── 2-Layer GATv2 Multi-Head Attention (4 heads, hidden=48)
  │                    └── Global Mean Pooling ──► Graph Representation: [1, 192]
  │                                                      │
  └──────────────────────────────────────────────────────┴────────► [Stage 3] Concatenation Fusion Head
                                                                      ├── Fused Representation: [1, 384]
                                                                      ├── Linear(384, 128) -> LayerNorm -> ReLU -> Dropout(0.2)
                                                                      └── Linear(128, 2) ──► Logits [1, 2] ──► Softmax Probability
```

### Preprocessing and Interface Specifications:
- **Image Input Format:** RGB image, resized to $224 \times 224$ via PIL Bicubic interpolation (deterministic test mode).
- **Pixel Normalization:** Channel-wise standardization with dataset statistics:
  - Mean: $[0.485, 0.456, 0.406]$
  - Std: $[0.229, 0.224, 0.225]$
- **Class Index Mapping:**
  - Index `0`: `NON CANCER` (Benign / Normal Oral Tissue)
  - Index `1`: `CANCER` (Oral Squamous Cell Carcinoma / Suspicious Lesion)
- **Decision Threshold:** Fixed at standard $0.50$ ($p \ge 0.50 \implies \text{CANCER}$).

---

## 4. Class-Weight Deviation Disclosure

As audited in `reports/internal_archive/phase5_phase6_class_weight_audit.md`, during Phases 5 and 6, training scripts utilized loss weights derived from an initial $326 / 160$ count estimation:
- **Trained Class Weights:** $w_0 = 1.51875$, $w_1 = 0.7454$ (Weight ratio: $w_0 / w_1 = 2.0375$).
- **Approved Manifest Class Weights ($322 / 164$):** $w_0 = 1.481707$, $w_1 = 0.754658$ (Weight ratio: $w_0 / w_1 = 1.9634$).
- **Discrepancy:** $+3.77\%$ ratio deviation.

### Impact Analysis for Selected Checkpoint:
- All 15 Phase 5 & 6 models (including `hybrid_vit_gat_concat_best.pth`, `hybrid_vit_gat_gated_best.pth`, and `handcrafted_all_143_best.pth`) were trained with these identical weights, ensuring internal consistency across all comparative ablation runs.
- The $+3.77\%$ deviation slightly favored the minority class (Non-Cancer), encouraging higher sensitivity towards distinguishing normal tissue without introducing label leakage.
- For Phase 7 application inference, the trained weights in the checkpoint are frozen. Loss weights only affect gradient updates during training and have no effect during inference forward passes.

---

## 5. Explainable AI (XAI) Compatibility Assessment

The selected hybrid configuration provides full architectural access to the intermediate representations required for Phase 7 XAI:

1. **ViT Patch-Level Attention Rollout:**
   - The 4 Transformer encoder blocks expose multi-head self-attention weight tensors $[1, 3, 197, 197]$.
   - Attention Rollout recursively multiplies attention matrices across layers to map how information from all 196 image patches flows into the final `[CLS]` token.
2. **GATv2 Graph Relational Explainability:**
   - The 2 GATv2 layers calculate dynamic attention coefficients across all active spatial and similarity edges ($196 \times 196$ adjacency matrix).
   - Enables visualization of the graph topology, highlighting which spatial tissue clusters exchange the strongest structural attention.
3. **Multimodal Fusion Attribution:**
   - The concatenated vector $[h_{\text{cls}} \parallel h_{\text{graph}}] \in \mathbb{R}^{384}$ feeds into the first fusion linear layer.
   - Saliency gradients or integrated gradients can directly partition decision attribution between the global visual pathway ($[0:192]$) and the topological graph pathway ($[192:384]$).

---

## 6. Scientific Position on Handcrafted Features

The handcrafted feature model (`models/handcrafted_all_143_best.pth`, test accuracy $87.16\%$) will be maintained in the project codebase and showcased within the application's **"Comparative Baselines & Research Benchmark"** section.

**It will NOT replace the ViT-GAT hybrid model** for the following verified reasons:
1. **Academic Mandate:** The primary research question set by the faculty is evaluating Vision Transformers and Graph Attention Networks for structured oral image screening.
2. **Representation Learning:** Handcrafted descriptors calculate static mathematical statistics; they cannot learn deep hierarchical visual representations or dynamic topological graphs.
3. **No Image-to-Graph Interpretability:** Handcrafted features do not generate attention heatmaps or structural tissue graphs.

---

## 7. Conclusion & Selected Model Summary

| Attribute | Specification |
| :--- | :--- |
| **Selected Checkpoint** | `models/hybrid_vit_gat_concat_best.pth` |
| **ViT Backbone Checkpoint** | `models/vit_scratch_best.pth` |
| **GATv2 Backbone Checkpoint** | `models/graph_hybrid_gatv2_best.pth` |
| **Total Trainable Parameters** | 49,794 (Fusion Head) + 1,965,890 (ViT) + 222,722 (GATv2) = 2,238,406 total |
| **Validation Performance** | Acc: **81.48%**, Sens: **83.56%**, Spec: **77.14%**, F1: **0.8592** |
| **Deployment Mode** | Primary Screening Model for ORALSCAN AI Application |
| **Status** | Verified Compatible & Ready for Stage 3 XAI Implementation |
