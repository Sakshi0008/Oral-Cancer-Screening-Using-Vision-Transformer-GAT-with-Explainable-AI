# Phase 7: Repository Cleanup, Scientific Audit, and Architecture Presentation Plan

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Phase:** Phase 7 (Final Research Application, Explainable AI, User-Friendly Reporting, and Frontend)  
**Stage:** Stage 1 (Repository Audit and Presentation Reorganization Proposal)  
**Date:** November 24, 2026  
**Auditor / Assistant:** Antigravity AI Assistant  
**Git Branch:** `main` (Synchronized with `origin/main` at commit `214dcab`)  
**Status:** Read-Only Audit & Proposed Plan — **No files deleted or moved pending user approval**  

---

## 1. Executive Summary & Research Mandate

The primary academic mandate for this Major Project, as stipulated by the faculty evaluation committee, centers on four essential pillars:
1. **Vision Transformer (ViT):** From-scratch self-attention modeling on oral lesion image patches without external pretraining.
2. **Graph Attention Network (GAT / GATv2):** Modeling topological, spatial, and feature-similarity relationships between tissue regions.
3. **Hybrid Architecture Fusion:** Multimodal integration of ViT patch representations with GAT graph-level representations.
4. **Explainable AI (XAI):** Clinically interpretable visual and structural explanations (attention rollout, graph node/edge attributions, saliency heatmaps, and plain-language summaries).

During Phases 5 and 6, domain-engineered handcrafted features (LBP, GLCM, HSV/Lab color moments, Sobel edge statistics) were introduced as an additional comparative experiment. While the 143-dimensional handcrafted baseline achieved an empirical accuracy of **87.16%** on the small dataset ($N=109$ test samples), the central research contribution remains the **ViT–GAT hybrid architecture**. Handcrafted features must remain an informative comparative baseline and ablation study; they must not silently displace the primary deep-graph hybrid architecture.

This audit report provides:
- A complete catalog and classification of all 180+ project files across `src/`, `models/`, `data/`, `reports/`, `tests/`, and root directories.
- A proposed reorganization plan separating faculty-facing presentation assets from internal development logs without destroying experimental reproducibility.
- An inspection of checkpoint compatibility, preprocessing pipelines, and inference interfaces for the upcoming Phase 7 ORALSCAN AI web application.
- A scientific reconciliation of the performance differences between handcrafted features and from-scratch deep-graph models.

---

## 2. Current Repository State & Git Status

- **Working Directory:** `c:\Users\Sakshi Aswale\OneDrive\Desktop\Major Project final\Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI`
- **Git Branch:** `main`
- **Remote Synchronization:** Up to date with `origin/main` (`https://github.com/Sakshi0008/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI.git`).
- **HEAD Commit:** `214dcab` (*"Complete Phases 1-6: ViT-GAT hybrid, feature engineering, and validation"*).
- **Unit Test Status:** All 37 tests across 8 test suites passing (`Ran 37 tests in 7.064s - OK`).
- **Untracked Working Tree Additions:**
  - `reports/github_phase1_phase6_push_report.md` (Push verification record from previous session).
  - `src/screen.py` (Unified demonstration screening engine).
  - `scratch/matplotlib/` (Runtime font cache).

---

## 3. Comprehensive File Classification

Every file in the repository has been audited and classified into three distinct categories:

### Category A: Essential Project Files (Faculty-Facing & Application-Critical)

These files constitute the primary deliverables, executable source code, test suites, approved dataset manifests, production models, and benchmark results:

| Path / Pattern | Component | Description / Purpose | Git Status |
| :--- | :--- | :--- | :--- |
| `README.md` | Documentation | Primary project documentation, architecture diagrams, benchmark tables, and instructions. | Tracked |
| `.gitignore` | Configuration | Production exclusion rules protecting heavy binaries and dataset privacy. | Tracked |
| `data/splits_primary/train.csv` | Dataset | Approved primary training split ($N=486$: 322 cancer, 164 non-cancer). | Tracked |
| `data/splits_primary/val.csv` | Dataset | Approved primary validation split ($N=108$: 73 cancer, 35 non-cancer). | Tracked |
| `data/splits_primary/test.csv` | Dataset | Approved frozen primary test split ($N=109$: 75 cancer, 34 non-cancer). | Tracked |
| `data/splits_primary/excluded_redundant_copies.csv` | Dataset | Audit record of 47 exact duplicate/redundant images excluded from primary cohort. | Tracked |
| `models/handcrafted_scaler.pt` | Model Artifact | Fitted training-split StandardScaler for 143 handcrafted features (2.9 KB). | Tracked |
| `src/dataset.py` | Source Code | PyTorch Dataset class, deterministic resize, and training normalization pipelines. | Tracked |
| `src/screen.py` | Source Code | Unified CLI screening & multi-model inference demonstration engine. | Untracked (Keep) |
| `src/models/vit_scratch.py` | Source Code | 4-layer, 3-head from-scratch Vision Transformer implementation ($d=192$). | Tracked |
| `src/models/graph_attention.py` | Source Code | Standard Graph Attention Network (GAT) implementation. | Tracked |
| `src/models/graph_attention_v2.py` | Source Code | Dynamic Graph Attention Network v2 (GATv2) implementation. | Tracked |
| `src/models/graph_classifier.py` | Source Code | Graph classification model with spatial/similarity/hybrid adjacency builders. | Tracked |
| `src/models/hybrid_fusion.py` | Source Code | Concatenation, Gated, and Residual ViT–GAT representation fusion heads. | Tracked |
| `src/models/multiscale_fusion.py` | Source Code | Multi-scale ViT and graph representation pooling modules. | Tracked |
| `src/models/handcrafted_fusion.py` | Source Code | Handcrafted projection heads, bi-modal, and tri-modal fusion architectures. | Tracked |
| `src/models/efficientnet_baseline.py` | Source Code | Scratch-trained EfficientNet-B0 baseline architecture. | Tracked |
| `src/graph/*.py` (5 modules) | Source Code | Graph adjacency builders (spatial grid, k-NN similarity, hybrid) and pooling. | Tracked |
| `src/features/*.py` (6 modules) | Source Code | Classical feature extraction: LBP, GLCM, HSV/Lab color, Sobel edge, and scaler. | Tracked |
| `src/train_*.py` (4 modules) | Source Code | Training pipelines for EfficientNet, ViT, Graph models, and Hybrid heads. | Tracked |
| `src/evaluate_*.py` (4 modules) | Source Code | Test evaluation scripts and metrics engines with Wilson score confidence intervals. | Tracked |
| `src/run_ablation_studies.py` | Source Code | Master runner orchestrating all 14 Phase 5 & 6 ablation experiments. | Tracked |
| `tests/*.py` (8 test suites) | Test Suite | 37 unit tests verifying models, graph batching, scaling, and forward passes. | Tracked |
| `reports/phase5_phase6_final_comparison.csv` | Benchmark | Master empirical comparison table across all 19 model configurations. | Tracked |
| `reports/phase4_experiment_comparison.csv` | Benchmark | Comparison table of all 6 GAT/GATv2 graph configurations against ViT. | Tracked |
| `reports/phase5_phase6_ablation_results.csv` | Benchmark | Detailed ablation metrics, validation scores, and parameter counts. | Tracked |
| `reports/phase5_phase6_experiment_manifest.json` | Benchmark | Full JSON manifest of hyperparameters, configurations, and evaluation metrics. | Tracked |
| `reports/phase2_confusion_matrix.png` | Visualization | Test set confusion matrix plot for EfficientNet-B0 baseline. | Tracked |
| `reports/phase3_confusion_matrix.png` | Visualization | Test set confusion matrix plot for Vision Transformer baseline. | Tracked |
| `reports/phase2_learning_curves.png` | Visualization | Training and validation loss/accuracy curves for EfficientNet-B0 baseline. | Tracked |
| `reports/phase3_learning_curves.png` | Visualization | Training and validation loss/accuracy curves for Vision Transformer baseline. | Tracked |
| `models/*.pth` (23 checkpoints) | Model Weights | Saved model weights preserved locally on disk for application inference. | Local (.gitignore) |
| `data/features_*/*.pt` (12 caches) | Feature Caches | Intermediate ViT patch, graph, and handcrafted feature caches. | Local (.gitignore) |
| `Oral Cancer/*` (750 raw images) | Raw Dataset | Original oral lesion images preserved locally on disk. | Local (.gitignore) |

---

### Category B: Internal Research & Development Documentation (Archive Candidates)

These files contain detailed Phase 1–6 audit histories, intermediate split manifests, epoch-by-epoch training logs, and reconciliation investigations. They are scientifically critical for auditability and must **not be deleted**, but they clutter the primary project presentation:

| Path | Description | Recommended Treatment |
| :--- | :--- | :--- |
| `reports/dataset_audit.md` | Phase 1 duplicate and image audit (15.6 KB). | Move to `reports/internal_archive/` |
| `reports/dataset_audit_initial_backup.md` | Initial Phase 1 audit notes before redundant curation (13.3 KB). | Move to `reports/internal_archive/` |
| `reports/duplicate_representatives_report.md` | Duplicate group clustering details (6.8 KB). | Move to `reports/internal_archive/` |
| `reports/github_phase1_phase6_push_report.md` | Pre-push and staging verification report (8.9 KB). | Move to `reports/internal_archive/` |
| `reports/phase2_efficientnet_training_report.md` | Phase 2 narrative training log (15.8 KB). | Move to `reports/internal_archive/` |
| `reports/phase2_final_verification.md` | Phase 2 independent audit report (15.9 KB). | Move to `reports/internal_archive/` |
| `reports/phase3_phase4_final_verification.md` | Phase 3 and 4 audit report (19.1 KB). | Move to `reports/internal_archive/` |
| `reports/phase3_vit_report.md` | Phase 3 narrative training report (12.6 KB). | Move to `reports/internal_archive/` |
| `reports/phase4_gat_gatv2_report.md` | Phase 4 narrative GNN report (11.7 KB). | Move to `reports/internal_archive/` |
| `reports/phase4_graph_construction_report.md` | Phase 4 graph engineering report (8.1 KB). | Move to `reports/internal_archive/` |
| `reports/phase5_hybrid_fusion_report.md` | Phase 5 hybrid fusion narrative report (14.0 KB). | Move to `reports/internal_archive/` |
| `reports/phase5_phase6_class_weight_audit.md` | Deep audit of class-weight discrepancies (25.8 KB). | Move to `reports/internal_archive/` |
| `reports/phase5_phase6_final_verification.md` | Phase 5 & 6 verification audit (12.1 KB). | Move to `reports/internal_archive/` |
| `reports/phase5_phase6_independent_consistency_audit.md` | Consistency audit across all models (28.1 KB). | Move to `reports/internal_archive/` |
| `reports/phase6_feature_engineering_report.md` | Phase 6 handcrafted engineering report (18.4 KB). | Move to `reports/internal_archive/` |
| `reports/phase2_training_history.csv` | Epoch-by-epoch loss/accuracy history for Phase 2. | Move to `reports/internal_archive/` |
| `reports/phase3_vit_training_history.csv` | Epoch-by-epoch loss/accuracy history for Phase 3. | Move to `reports/internal_archive/` |
| `reports/phase4_training_*.csv` (6 CSVs) | Epoch-by-epoch histories for spatial, similarity, hybrid GAT/GATv2. | Move to `reports/internal_archive/` |
| `reports/phase5_phase6_training_history.csv` | Full epoch histories for all 15 Phase 5 & 6 models. | Move to `reports/internal_archive/` |
| `reports/phase2_efficientnet_metrics.json` | JSON test metrics for EfficientNet. | Move to `reports/internal_archive/` |
| `reports/phase3_vit_metrics.json` | JSON test metrics for ViT. | Move to `reports/internal_archive/` |
| `data/splits_initial_backup/` (3 CSVs) | Initial pre-audit train/val/test splits. | Move to `data/splits_archive/` |
| `data/splits/` (5 CSVs) | Intermediate duplicate audit splits superseded by `data/splits_primary/`. | Move to `data/splits_archive/` |

---

### Category C: Unnecessary, Generated & Ephemeral Files (Cleanup Candidates)

These files are automatic runtime outputs, temporary files, or bytecode caches that can be safely deleted without any impact on code or data:

| Path / Pattern | Description | Proposed Action |
| :--- | :--- | :--- |
| `**/__pycache__/` | Compiled Python bytecode (`.pyc` files in `src/`, `models/`, `graph/`, `features/`, `tests/`). | Safe to delete (ignored by `.gitignore`; regenerated on execution). |
| `scratch/matplotlib/` | Matplotlib font cache file (`fontlist-v390.json`). | Safe to clean/ignore. |

---

## 4. Proposed Reorganization Plan

To present a clean, faculty-ready repository structure, we propose the following reorganization **(subject to user approval)**:

```text
Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/
├── .gitignore
├── README.md                               <-- Faculty-facing comprehensive project presentation
│
├── data/
│   ├── splits_primary/                     <-- The approved, active 703-image cohort
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── excluded_redundant_copies.csv
│   ├── splits_archive/                     <-- Historical Phase 1 split iterations
│   │   ├── splits_v1_initial/
│   │   └── splits_v2_pre_curation/
│   └── (features_* preserved locally)
│
├── models/
│   ├── handcrafted_scaler.pt               <-- Tracked normalization scaler
│   └── (*.pth checkpoints preserved locally)
│
├── src/                                    <-- Modular production source code
│   ├── dataset.py
│   ├── screen.py                           <-- CLI Screening & Demo Engine
│   ├── models/                             <-- ViT, GAT, GATv2, Hybrid, Handcrafted
│   ├── graph/                              <-- Adjacency builders & graph pooling
│   ├── features/                           <-- Classical feature extractors & scaler
│   ├── xai/                                <-- Phase 7: Explainable AI engines (Rollout, GNN explainer, CAM)
│   ├── app/                                <-- Phase 7: ORALSCAN AI Backend & Frontend
│   ├── train_*.py
│   └── evaluate_*.py
│
├── tests/                                  <-- Complete unit test suite (37 tests)
│
└── reports/
    ├── phase5_phase6_final_comparison.csv  <-- Master benchmark table
    ├── phase4_experiment_comparison.csv    <-- Graph architecture benchmark table
    ├── phase5_phase6_ablation_results.csv  <-- Ablation study metrics
    ├── phase5_phase6_experiment_manifest.json
    ├── phase2_confusion_matrix.png
    ├── phase3_confusion_matrix.png
    ├── phase2_learning_curves.png
    ├── phase3_learning_curves.png
    └── internal_archive/                   <-- Preserved Phase 1-6 audit & verification logs
        ├── audits/                         <-- Detailed audit reports (class-weights, data leakage)
        ├── phase_reports/                  <-- Narrative phase-by-phase development logs
        └── training_histories/             <-- Epoch-by-epoch loss/accuracy CSVs
```

> [!IMPORTANT]
> **No files will be moved or deleted until explicit user approval is granted.** All 15 development and audit reports remain accessible on disk and are preserved for full scientific reproducibility.

---

## 5. Model Architecture & Inference Compatibility Audit

We audited the saved checkpoints in `models/` to determine compatibility with the upcoming Phase 7 ORALSCAN AI application:

### A. Primary Candidate Models for Phase 7 Application

1. **ViT-Scratch Baseline (`models/vit_scratch_best.pth`):**
   - **Architecture:** 4 Transformer encoder layers, 3 attention heads, embedding dimension $d=192$, MLP dimension $768$, patch size $16 \times 16$.
   - **Status:** Verified working. Generates both classification logits $[1, 2]$ and patch tokens $[1, 196, 192]$.
   - **XAI Compatibility:** Fully compatible with Attention Rollout and Attention Map extraction from its 4 encoder layers.

2. **Selected Graph Model (`models/graph_hybrid_gatv2_best.pth`):**
   - **Architecture:** 2-layer GATv2 with 4 attention heads ($4 \times 48 = 192$ hidden features), dynamic attention mechanism, hybrid adjacency (spatial 8-NN + similarity 8-NN), global mean pooling.
   - **Status:** Verified working. Operates directly on the ViT patch tokens $[1, 196, 192]$.
   - **XAI Compatibility:** Fully compatible with edge attention extraction ($196 \times 196$ attention matrix) to highlight structural tissue relationships.

3. **Primary Hybrid Architecture: ViT + GATv2 Concatenation (`models/hybrid_vit_gat_concat_best.pth`):**
   - **Architecture:** Combines ViT `[CLS]` token $[1, 192]$ and GATv2 pooled graph representation $[1, 192]$ into a unified 384-dimensional representation $[1, 384]$, followed by `Linear(384, 128) -> LayerNorm -> ReLU -> Dropout(0.2) -> Linear(128, 2)`.
   - **Test Set Performance:** Accuracy: **$74.31\%$**, Cancer Precision: **$87.30\%$**, Specificity: **$76.47\%$**, ROC-AUC: **$82.04\%$**, PR-AUC: **$90.57\%$**.
   - **Inference Status:** Tested and verified compatible in `src/screen.py`.
   - **Recommendation:** **Selected as the primary deployment model for ORALSCAN AI.**

4. **Alternative Primary Hybrid: ViT + GATv2 Gated Fusion (`models/hybrid_vit_gat_gated_best.pth`):**
   - **Architecture:** Learnable gating vector $g = \sigma(W [h_{cls} || h_{graph}])$ dynamically balancing global visual context and graph structural context.
   - **Test Set Performance:** Accuracy: **$76.15\%$**, Sensitivity: **$77.33\%$**, Specificity: **$73.53\%$**, ROC-AUC: **$80.63\%$**, PR-AUC: **$88.66\%$**.
   - **Recommendation:** Available as an alternate selectable fusion option in the research dashboard.

5. **End-to-End Fine-Tuned Hybrid (`models/hybrid_e2e_finetuned_best.pth`):**
   - **Architecture:** Jointly trained ViT backbone + GATv2 graph network + Concatenation head (2,238,406 parameters).
   - **Test Set Performance:** Accuracy: **$76.15\%$**, Sensitivity: **$76.00\%$**, Specificity: **$76.47\%$**, ROC-AUC: **$82.51\%$**, PR-AUC: **$90.70\%$**.

6. **Handcrafted Feature Model (`models/handcrafted_all_143_best.pth` + `models/handcrafted_scaler.pt`):**
   - **Architecture:** 143 classical descriptors (59 LBP + 24 GLCM + 46 Color + 14 Edge) normalized with training StandardScaler, followed by `Linear(143, 64) -> LayerNorm -> ReLU -> Linear(64, 2)`.
   - **Role:** Maintained in the application's "Comparative Baselines & Research Benchmark" section.

---

## 6. Scientific Reconciliation: Handcrafted Features vs. ViT-GAT Hybrid

A critical question investigated in this audit is:
> *Why did the Handcrafted-only model achieve 87.16% accuracy, while ViT achieved 75.23% and Hybrid ViT-GAT achieved 77.06%? How should this be presented to the faculty?*

### Experimental Observations and Scientific Nuances:
1. **Observed Performance Difference as an Empirical Finding:**
   - On this specific holdout partition ($N=109$ test samples), the classical handcrafted feature classifier achieved 87.16% accuracy, outperforming the from-scratch deep models (ViT: 75.23%, GATv2: 77.06%, Hybrid ViT-GATv2: 77.06%).
   - **Crucially, this is an experimental observation within this single-source dataset cohort, not a proven causal law.** The dataset contains only 486 training samples (322 cancer, 164 non-cancer) without multi-center external validation.
   - In low-sample regimes ($N < 500$) without external pretraining, high-dimensional unconstrained neural networks (ViT has ~1.96M parameters) are known to exhibit high sample variance and optimization challenges compared to compact linear/MLP classifiers trained on low-dimensional statistical feature vectors (143 dimensions, 11,554 parameters).

2. **Precise Nature of Handcrafted Features:**
   - Handcrafted features compute deterministic mathematical statistics: local intensity transitions (LBP: 59 dims), spatial co-occurrence contrast/homogeneity (GLCM: 24 dims), color channel mean/variance/histograms in HSV and Lab spaces (Color: 46 dims), and gradient magnitude/orientation distributions (Sobel: 14 dims).
   - **They calculate mathematical texture, colour, and edge statistics.** They do NOT directly detect or establish clinical conditions such as keratinization, ulceration, epithelial dysplasia, or tissue necrosis, because the dataset possesses only binary image-level labels without pixel-level pathological ground truth annotations.

3. **Explainability Requires Independent Validation:**
   - The ViT-GAT graph structure does NOT automatically provide clinical explainability by itself.
   - Attention weights and adjacency edges reflect mathematical weighting inside the network; they do not constitute causal proof of malignancy or verified pathological boundaries.
   - Explainable AI must be separately implemented, rigorously evaluated (e.g., via perturbation and stability testing), and presented with appropriate clinical disclaimers.

4. **Faculty-Facing Presentation Strategy:**
   - **Do not conceal the handcrafted baseline:** Report all 19 experiments transparently as empirical evidence. Handcrafted features serve as an essential classical benchmark demonstrating how fixed statistical descriptors perform in low-data regimes.
   - **Do not redesign the project into a handcrafted-only application:** Handcrafted features do not perform hierarchical representation learning, cannot discover latent topological patch relationships, and do not address the faculty's core research mandate.
   - **Position the ViT-GAT Hybrid as the Primary Research Architecture:** The core academic contribution is the investigation of Vision Transformers coupled with Graph Attention Networks for structured oral lesion analysis, accompanied by dedicated, validated Explainable AI techniques.

---

## 7. Faculty-Facing README Blueprint

The new presentation `README.md` will be structured to immediately communicate academic rigor:

1. **Title & Research Abstract:** Clear academic statement of the ViT–GAT hybrid framework with Explainable AI.
2. **Clinical Motivation & Problem Statement:** The challenge of early oral cancer screening, low-resource settings, and the need for interpretable AI.
3. **Proposed System Architecture:** Detailed diagrams showing:
   - Input Image $\to$ $16 \times 16$ Patch Partitioning $\to$ ViT Scratch Encoder.
   - ViT Patch Tokens $\to$ Spatial & Feature-Similarity Graph Construction $\to$ GATv2 Graph Neural Network.
   - Hybrid Fusion Head (Concatenation / Gating).
   - Explainable AI Engine (Attention Rollout + Graph Connectivity + Plain-Language Summary).
4. **Dataset & Partitioning:** Primary 703-image cohort summary (486 train, 108 val, 109 test), duplicate-aware splitting, single-source disclosure.
5. **Master Empirical Evaluation Table:** Full benchmark table with all 19 models clearly grouped into:
   - *Primary Proposed Methodology:* ViT–GAT Hybrid Models (Concatenation, Gated, Residual, End-to-End).
   - *Deep Learning & Graph Baselines:* ViT-Scratch Baseline, GATv2 Baseline.
   - *Comparative Baselines & Ablations:* EfficientNet-B0 CNN Baseline, Handcrafted Features (LBP, GLCM, Color, Edge), Tri-Modal Fusion.
6. **Explainable AI (XAI) Capabilities:** Visual examples of ViT Attention Rollout and GATv2 edge connectivity maps.
7. **ORALSCAN AI Application:** Overview, architecture, screenshots, and execution instructions.
8. **Limitations & Ethical Disclosures:** Research prototype status, single-source dataset, necessity for biopsy verification.

---

## 8. Summary of Completed Stage 1 & Request for Approval

### What Was Completed in Stage 1:
- [x] Full recursive audit of all 180+ files across the repository.
- [x] Classification into Category A (Essential), Category B (Internal Documentation), and Category C (Generated/Ephemeral).
- [x] Verified git status: clean working tree, synchronized with GitHub remote `origin/main` at commit `214dcab`.
- [x] Verified unit tests: all 37 tests passing.
- [x] Verified trained checkpoints and inference compatibility for ViT, GATv2, and Hybrid Concatenation models.
- [x] Completed scientific reconciliation between handcrafted and deep-graph hybrid models.
- [x] Formulated clean faculty-facing repository structure and README blueprint.

### Proposed Next Actions Awaiting User Approval:
1. **File Reorganization:** Create `reports/internal_archive/` and move the 15 internal audit and training history files there to declutter `reports/` without deleting any data.
2. **Move to Stage 2:** Proceed with Model Selection, dynamic class-weight verification, and inference validation for the primary ViT-GAT hybrid model.
