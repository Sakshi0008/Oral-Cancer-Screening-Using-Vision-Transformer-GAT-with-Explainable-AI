# Phase 5 and Phase 6 Final Verification Audit Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Audit Date:** November 13, 2026
**Auditor:** Automated Rigorous Verification Harness
**Scope:** Phase 5 (ViT + GAT Hybrid Representation Fusion) and Phase 6 (Handcrafted Image Feature Engineering and Controlled Ablation Studies)
**Overall Audit Status:** **PASS**

---

## 1. Executive Summary & Verification Verdict

A comprehensive, read-only technical and scientific audit was conducted across all source files, unit test suites, saved checkpoints, feature caches, split manifests, and evaluation deliverables produced during Phase 5 and Phase 6.

### Verdict: **PASS**
- **Zero Critical Issues Detected.**
- **Zero Data Leakage:** All normalization parameters and feature scalers (`StandardScaler`) were fitted strictly on the 486 primary training samples.
- **Zero Pretrained Weights:** All newly introduced layers were initialized from scratch; deep backbones originated strictly from scratch-trained Phase 3 and Phase 4 project checkpoints.
- **Strict Dataset Discipline:** Evaluated exclusively on the approved 703-image primary duplicate-collapsed dataset (`data/splits_primary/`).
- **Complete Test Suite Success:** All 37 unit tests across four test suites passed in 3.823 seconds.
- **Strict Phase Boundary Respected:** Phase 7 (Explainable AI / Grad-CAM / Attention Rollout) has NOT been started.

---

## 2. Checkpoint and Weight Initialization Audit

| Checkpoint File | Status | Size | Expected Source / Initialization | Pretrained Weights Detected? |
| :--- | :---: | :---: | :--- | :---: |
| `models/handcrafted_scaler.pt` | PASS | 2.9 KB | Fitted on `data/splits_primary/train.csv` ($N=486$) | None (Empirical Stats) |
| `models/hybrid_vit_gat_concat_best.pth` | PASS | 202 KB | ConcatenationFusion ($d=384 \to 128 \to 2$), Kaiming normal | None |
| `models/hybrid_vit_gat_gated_best.pth` | PASS | 350 KB | GatedFusion ($d=384 \to 192 \to 64 \to 2$), Kaiming normal | None |
| `models/hybrid_vit_gat_residual_best.pth` | PASS | 355 KB | ResidualFusion ($d=192 + \text{MLP}(192)$), Kaiming normal | None |
| `models/handcrafted_all_143_best.pth` | PASS | 50 KB | HandcraftedOnlyClassifier ($143 \to 64 \to 32 \to 2$) | None |
| `models/vit_plus_handcrafted_best.pth` | PASS | 127 KB | BiModalHandcraftedFusion ($192 + 64 = 256 \to 64 \to 2$) | None |
| `models/gat_plus_handcrafted_best.pth` | PASS | 127 KB | BiModalHandcraftedFusion ($192 + 64 = 256 \to 64 \to 2$) | None |
| `models/trimodal_vit_gat_hc_concat_best.pth` | PASS | 326 KB | TriModalConcatFusion ($192 + 192 + 64 = 448 \to 128 \to 2$) | None |
| `models/trimodal_vit_gat_hc_gated_best.pth` | PASS | 416 KB | TriModalGatedFusion (Modality-attention softmax) | None |
| `models/multiscale_vit_best.pth` | PASS | 335 KB | MultiScaleViTClassifier ($192 \times 3 = 576 \to 128 \to 2$) | None |
| `models/multiscale_hybrid_best.pth` | PASS | 433 KB | MultiScaleHybridClassifier ($576 + 192 = 768 \to 128 \to 2$) | None |
| `models/hybrid_e2e_finetuned_best.pth` | PASS | 8.98 MB | ViT-Scratch (Epoch 34) + Hybrid GATv2 (Epoch 21) + Concat | None |

---

## 3. Data Leakage and Split Isolation Audit

### 3.1 Training-Only Standardization
- `src/features/feature_scaler.py` strictly raises a `ValueError` if an attempt is made to fit on `'val'` or `'test'`.
- Inspection of `models/handcrafted_scaler.pt` confirms:
  - `n_samples_seen`: **486** (matches `train.csv` size exactly).
  - `feature_dim`: **143**.
  - `fitted_on`: `"data/splits_primary/train.csv"`.
  - Zero validation or test samples influenced feature scaling parameters.

### 3.2 Feature Cache Alignment
Every split was checked for 1-to-1 correspondence across all cached files:
1. `data/splits_primary/{split}.csv`
2. `data/features_vit_scratch/{split}_patch_features.pt`
3. `data/features_handcrafted/{split}_handcrafted_features.pt`
4. `data/features_graph/{split}_graph_hybrid_gatv2.pt`

**Verification Results:**
- **Train Split ($N=486$):** 100% filename and label alignment verified across all 4 files.
- **Validation Split ($N=108$):** 100% filename and label alignment verified across all 4 files.
- **Test Split ($N=109$):** 100% filename and label alignment verified across all 4 files.

---

## 4. Preservation of Prior Phase Artifacts

All artifacts from Phase 1 through Phase 4 were inspected to ensure zero regression or accidental overwriting:
- `models/efficientnet_b0_scratch_best.pth`: Verified present (48.6 MB, Epoch 13 checkpoint).
- `models/vit_scratch_best.pth`: Verified present (23.6 MB, Epoch 34 checkpoint).
- `models/graph_hybrid_gatv2_best.pth`: Verified present (897 KB, Epoch 21 checkpoint).
- `models/graph_hybrid_gat_best.pth`: Verified present (308 KB, Epoch 26 checkpoint).
- `reports/dataset_audit.md`: Verified unmodified.
- `reports/phase2_final_verification.md`: Verified unmodified.
- `reports/phase3_phase4_final_verification.md`: Verified unmodified.
- `reports/phase4_experiment_comparison.csv`: Verified unmodified.

---

## 5. Unit Test Suite Execution Audit

The complete project test suite was executed via `python -m unittest discover -s tests -p "test_*.py"`:

```
.....................................
----------------------------------------------------------------------
Ran 37 tests in 3.823s

OK
```

### Breakdown of Verified Test Modules:
1. `tests/test_handcrafted_features.py`:
   - `test_lbp_extractor_dimensions`: PASS (59 bins, sums to 1.0)
   - `test_glcm_extractor_dimensions`: PASS (24 features, no NaNs/Infs)
   - `test_color_extractor_dimensions`: PASS (46 features)
   - `test_edge_extractor_dimensions`: PASS (14 features)
   - `test_unified_extractor_dimensions_and_consistency`: PASS (143 features, numpy/PIL parity)
   - `test_feature_scaler_training_only_enforcement`: PASS (rejects val/test, zero leakage)
2. `tests/test_hybrid_fusion.py`:
   - `test_concat_fusion`: PASS (forward, backward, grad flow, $[B, 384] \to [B, 2]$)
   - `test_gated_fusion`: PASS (sigmoid gate $[B, 192] \in [0, 1]$, fused $[B, 192]$)
   - `test_residual_fusion`: PASS (residual LayerNorm, $[B, 192]$)
   - `test_handcrafted_only_classifier`: PASS (MLP $[B, 143] \to [B, 2]$)
   - `test_tri_modal_concat`: PASS ($[B, 448] \to [B, 2]$)
   - `test_tri_modal_gated`: PASS (modality softmax weights sum to 1.0)
   - `test_batch_size_one`: PASS (handles $B=1$ evaluation cleanly)
3. `tests/test_multiscale_fusion.py`:
   - `test_multiscale_patch_pooler`: PASS ($14\times14 \to 7\times7$ pooling $\to$ fine & regional summaries)
   - `test_multiscale_vit_classifier`: PASS (fine + regional + CLS $= 576$ dims)
   - `test_multiscale_hybrid_classifier`: PASS ($576 + 192 = 768$ dims)
4. `tests/test_vit.py`:
   - 21 tests for Vision Transformer patch embeddings, pos embeddings, self-attention, blocks, forward passes: ALL PASS.

---

## 6. Summary of Master Experimental Results

| Model Configuration | Category | Test Accuracy | Cancer Sens | Specificity | Cancer Prec | Cancer F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **07_handcrafted_all_143** | Handcrafted | **87.16%** | **86.67%** | **88.24%** | **94.20%** | **0.9028** | **0.9322** | **0.9654** |
| **07a_handcrafted_lbp_only** | Sub-family | **83.49%** | **81.33%** | **88.24%** | **93.85%** | **0.8714** | **0.9161** | **0.9608** |
| **08_vit_plus_handcrafted** | Bi-Modal | **80.73%** | **80.00%** | **82.35%** | **90.91%** | **0.8511** | **0.8761** | **0.9399** |
| **09_gat_plus_handcrafted** | Bi-Modal | **80.73%** | **80.00%** | **82.35%** | **90.91%** | **0.8511** | **0.8690** | **0.9213** |
| **Phase 2 Baseline (EffNet-B0)** | CNN Base | **80.73%** | **80.00%** | **82.35%** | **90.91%** | **0.8511** | **0.8912** | **0.9176** |
| **02_selected_gatv2_hybrid** | Graph Base | **77.06%** | **81.33%** | **67.65%** | **84.72%** | **0.8299** | **0.8298** | **0.9151** |
| **10_trimodal_concat** | Tri-Modal | **77.06%** | **77.33%** | **76.47%** | **87.88%** | **0.8227** | **0.8451** | **0.9261** |
| **05_hybrid_vit_gat_gated** | Hybrid | **76.15%** | **77.33%** | **73.53%** | **86.57%** | **0.8169** | **0.8063** | **0.8866** |
| **06_hybrid_vit_gat_residual** | Hybrid | **76.15%** | **80.00%** | **67.65%** | **84.51%** | **0.8219** | **0.8094** | **0.9002** |
| **12_multiscale_vit_only** | Multi-Scale | **76.15%** | **78.67%** | **70.59%** | **85.51%** | **0.8194** | **0.8000** | **0.8847** |
| **13_multiscale_hybrid** | Multi-Scale | **76.15%** | **80.00%** | **67.65%** | **84.51%** | **0.8219** | **0.8141** | **0.9016** |
| **14_hybrid_e2e_finetuned** | Strategy B | **76.15%** | **76.00%** | **76.47%** | **87.69%** | **0.8143** | **0.8251** | **0.9070** |
| **07c_color_only** | Sub-family | **76.15%** | **76.00%** | **76.47%** | **87.69%** | **0.8143** | **0.8678** | **0.9296** |
| **01_vit_scratch_only** | ViT Base | **75.23%** | **76.00%** | **73.53%** | **86.36%** | **0.8085** | **0.8141** | **0.8957** |
| **04_hybrid_vit_gat_concat** | Hybrid | **74.31%** | **73.33%** | **76.47%** | **87.30%** | **0.7971** | **0.8204** | **0.9057** |
| **11_trimodal_gated** | Tri-Modal | **74.31%** | **73.33%** | **76.47%** | **87.30%** | **0.7971** | **0.8482** | **0.9228** |
| **07b_glcm_only** | Sub-family | **71.56%** | **68.00%** | **79.41%** | **87.93%** | **0.7669** | **0.8106** | **0.9038** |
| **07d_edge_only** | Sub-family | **71.56%** | **73.33%** | **67.65%** | **83.33%** | **0.7801** | **0.7886** | **0.8749** |
| **03_comparative_gat_hybrid** | Graph Base | **71.56%** | **76.00%** | **61.76%** | **81.43%** | **0.7862** | **0.8145** | **0.9047** |

---

## 7. Deliverables Checklist & File Inventory

- [x] `src/features/lbp_features.py` (59-bin uniform LBP texture extractor)
- [x] `src/features/glcm_features.py` (24-dim rotational-invariant Haralick GLCM extractor)
- [x] `src/features/color_features.py` (46-dim HSV & CIELAB moments and histograms)
- [x] `src/features/edge_features.py` (14-dim Sobel magnitude moments, densities, & orientations)
- [x] `src/features/handcrafted_extractor.py` (143-dim unified modular extractor)
- [x] `src/features/feature_scaler.py` (leakage-free `TrainingFeatureScaler` fitted on train only)
- [x] `src/extract_handcrafted_features.py` (feature extraction and cache generation)
- [x] `src/models/hybrid_fusion.py` (Concatenation, Gated, and Residual ViT+GAT fusion models)
- [x] `src/models/handcrafted_fusion.py` (Projection, Bi-Modal, and Tri-Modal fusion models)
- [x] `src/models/multiscale_fusion.py` (MultiScalePatchPooler, MultiScaleViT, MultiScaleHybrid)
- [x] `src/train_hybrid.py` (Unified trainer with early stopping on validation loss)
- [x] `src/evaluate_hybrid.py` (Evaluation engine with Wilson score & bootstrap CIs)
- [x] `src/run_ablation_studies.py` (Ablation runner for all 14 configurations + Strategy B)
- [x] `tests/test_handcrafted_features.py` (Unit tests for extractors and scaler)
- [x] `tests/test_hybrid_fusion.py` (Unit tests for hybrid, bi-modal, and tri-modal fusion)
- [x] `tests/test_multiscale_fusion.py` (Unit tests for multi-scale pooling and classifiers)
- [x] `reports/phase5_hybrid_fusion_report.md` (Detailed Phase 5 research report)
- [x] `reports/phase6_feature_engineering_report.md` (Detailed Phase 6 research report)
- [x] `reports/phase5_phase6_ablation_results.csv` (Full numerical ablation results)
- [x] `reports/phase5_phase6_final_comparison.csv` (Master comparison across all project phases)
- [x] `reports/phase5_phase6_training_history.csv` (Epoch-by-epoch training and validation loss/acc)
- [x] `reports/phase5_phase6_experiment_manifest.json` (Structured JSON experiment manifest)
- [x] `reports/phase5_phase6_final_verification.md` (This verification audit report)

---

## 8. Stop Condition Acknowledgment

In accordance with user instructions in Part L:
- **Phase 5 and Phase 6 are 100% complete and verified.**
- **Phase 7 (Explainable AI / Grad-CAM / Attention Rollout / GNN Explainer) has NOT been initiated.**
- All deliverables and reports are frozen and ready for user review and approval.
