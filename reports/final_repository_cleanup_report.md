# Final Repository Cleanup Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**System Name:** ORALSCAN AI  
**Date:** November 25, 2026  
**Auditor / Agent:** Antigravity AI Engine  
**Status:** Cleanup Executed Successfully  

---

## 1. Summary of Actions Taken

Following the comprehensive categorization documented in `reports/final_repository_cleanup_audit.md`, a strictly controlled cleanup operation was performed across the repository to ensure high presentation readiness for faculty evaluation without compromising experimental reproducibility or clinical pipeline stability.

### 2. Actions Executed

1. **Python Cache and Bytecode Cleanup:**
   - Recursively scanned and purged all `__pycache__/` directories across `backend/`, `src/`, `src/features/`, `src/graph/`, `src/models/`, `src/xai/`, and `tests/`.
   - Purged all transient `.pyc`, `.pyo`, and `.pyd` compiled files.

2. **Transient Test Uploads and Visualizations Removal:**
   - Cleared residual test JPEG images from `uploads/` generated during initial Stage 6 API unit tests.
   - Cleared residual PDF reports from `outputs/reports/` generated during Stage 6 test suites.
   - Cleared residual PNG visualization overlays from `outputs/visualizations/` generated during XAI test suites.
   - Placed `.gitkeep` files in `uploads/`, `outputs/reports/`, and `outputs/visualizations/` to ensure clean folder structure persistence.

3. **Scratch and Cache Cleanup:**
   - Cleared the Matplotlib runtime font cache folder `scratch/matplotlib/`.
   - Verified that all research helper scripts in `scratch/` remain untouched and are ignored by `.gitignore`.

4. **Repository Invariants Protected (Preservation Verification):**
   - **All 23 Checkpoints Preserved:** Confirmed all 23 `.pth` files in `models/` remain present, unaltered, and intact.
   - **Approved Primary Models:** `models/vit_scratch_best.pth`, `models/graph_hybrid_gatv2_best.pth`, and `models/hybrid_vit_gat_concat_best.pth` verified with zero byte modifications.
   - **Dataset Manifests Intact:** Confirmed `data/splits_primary/` (`train.csv`, `val.csv`, `test.csv`) and archive splits are completely unaltered.
   - **Feature Caches Intact:** `data/features_vit_scratch/`, `data/features_graph/`, and `data/features_handcrafted/` remain intact.
   - **Historical Documentation Intact:** All prior phase reports, confusion matrices, and ablation histories are safely preserved.

5. **Configuration and Environment Updates:**
   - Created `requirements.txt` specifying exact runtime dependencies (`torch`, `torchvision`, `fastapi`, `uvicorn`, `pydantic`, `python-multipart`, `pillow`, `matplotlib`, `scikit-learn`, `pandas`, `numpy`).
   - Updated `.gitignore` to explicitly ignore runtime databases (`*.db`, `data/*.db`), screening uploads (`uploads/*`), and screening outputs (`outputs/*`), preventing clutter in version control.

---

## 3. Inventory of Preserved Model Checkpoints

| # | Checkpoint Filename | Size | Purpose / Architecture |
| :---: | :--- | :--- | :--- |
| 1 | `models/vit_scratch_best.pth` | 23.7 MB | **Primary Application:** Vision Transformer from scratch (1.96M params) |
| 2 | `models/graph_hybrid_gatv2_best.pth` | 897 KB | **Primary Application:** GATv2 on hybrid spatial+similarity graph (222K params) |
| 3 | `models/hybrid_vit_gat_concat_best.pth` | 203 KB | **Primary Application:** Concatenation Fusion Head (49.8K params) |
| 4 | `models/efficientnet_b0_scratch_best.pth` | 48.6 MB | Phase 2 Comparative Baseline (CNN Transfer) |
| 5 | `models/graph_hybrid_gat_best.pth` | 308 KB | Phase 4 Ablation: GAT v1 on hybrid graph |
| 6 | `models/graph_similarity_gat_best.pth` | 308 KB | Phase 4 Ablation: GAT v1 on cosine similarity graph |
| 7 | `models/graph_similarity_gatv2_best.pth` | 897 KB | Phase 4 Ablation: GATv2 on cosine similarity graph |
| 8 | `models/graph_spatial_gat_best.pth` | 308 KB | Phase 4 Ablation: GAT v1 on 8-neighbor spatial graph |
| 9 | `models/graph_spatial_gatv2_best.pth` | 897 KB | Phase 4 Ablation: GATv2 on 8-neighbor spatial graph |
| 10 | `models/hybrid_e2e_finetuned_best.pth` | 8.98 MB | Phase 5 Ablation: End-to-end finetuned hybrid |
| 11 | `models/hybrid_vit_gat_gated_best.pth` | 350 KB | Phase 5 Ablation: Gated attention hybrid fusion |
| 12 | `models/hybrid_vit_gat_residual_best.pth`| 355 KB | Phase 5 Ablation: Residual connection hybrid fusion |
| 13 | `models/multiscale_vit_best.pth` | 335 KB | Phase 5 Ablation: Multiscale patch transformer |
| 14 | `models/multiscale_hybrid_best.pth` | 433 KB | Phase 5 Ablation: Multiscale hybrid graph fusion |
| 15 | `models/07a_handcrafted_lbp_only_best.pth`| 29 KB | Phase 6 Benchmark: Local Binary Patterns classifier |
| 16 | `models/07b_handcrafted_glcm_only_best.pth`| 20 KB | Phase 6 Benchmark: Gray-Level Co-occurrence Matrix |
| 17 | `models/07c_handcrafted_color_only_best.pth`| 26 KB | Phase 6 Benchmark: HSV/Lab color moments |
| 18 | `models/07d_handcrafted_edge_only_best.pth` | 18 KB | Phase 6 Benchmark: Sobel edge gradient histograms |
| 19 | `models/handcrafted_all_143_best.pth` | 51 KB | Phase 6 Benchmark: Combined 143-dim handcrafted MLP (87.16% acc) |
| 20 | `models/vit_plus_handcrafted_best.pth` | 127 KB | Phase 6 Benchmark: ViT + Handcrafted multimodal |
| 21 | `models/gat_plus_handcrafted_best.pth` | 127 KB | Phase 6 Benchmark: GATv2 + Handcrafted multimodal |
| 22 | `models/trimodal_vit_gat_hc_concat_best.pth`| 327 KB | Phase 6 Benchmark: Trimodal ViT + GATv2 + Handcrafted Concat |
| 23 | `models/trimodal_vit_gat_hc_gated_best.pth` | 416 KB | Phase 6 Benchmark: Trimodal ViT + GATv2 + Handcrafted Gated |
| -- | `models/handcrafted_scaler.pt` | 2.9 KB | Scikit-learn StandardScaler normalization parameters |

---

## 4. Current Git Status Post-Cleanup

The repository remains cleanly synchronized with `origin/main`. No commits or pushes have been staged or made. All disposable files have been removed, and runtime output paths are safely insulated from accidental tracking.

The application workspace is fully prepared for live backend startup and browser verification.
