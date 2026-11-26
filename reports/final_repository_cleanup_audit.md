# Final Repository Cleanup and Architecture Audit

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**System Name:** ORALSCAN AI  
**Date:** November 25, 2026  
**Auditor:** Antigravity AI Engine  
**Repository Branch:** `main` (Tracking `origin/main` at commit `214dcab`)  
**Status:** Pre-Demonstration Audit Completed  

---

## 1. Executive Summary & Objective

This audit establishes a rigorous inventory and categorization of all files, directories, models, datasets, reports, and transient artifacts within the repository prior to faculty demonstration and live application verification.

The repository supports dual objectives:
1. **Academic Research Archive:** Complete reproducibility of the comparative research across Vision Transformer (ViT from scratch), Graph Attention Networks (GAT/GATv2), Concatenation Hybrid Fusion, and comparative Handcrafted Feature benchmarks (LBP, GLCM, Color, Sobel).
2. **Interactive Clinical Application (ORALSCAN AI):** A production-grade, local web screening application integrating the verified hybrid deep-graph checkpoint pipeline, Explainable AI visual attributions (Rollout, GAT subgraph connectivity, Integrated Gradients fusion attribution, Clinical Narrative), SQLite authentication/history management, and publication-quality vector PDF report generation.

---

## 2. Comprehensive 5-Category File Classification

Every path in the repository is cataloged under one of five designated categories:

### Category A: Essential Application Files
*Production components required to execute the ORALSCAN AI web application, API, inference pipeline, and XAI suite.*

| Path / Component | Description | Parameter Count / Size |
| :--- | :--- | :--- |
| `backend/app.py` | Primary FastAPI application server, endpoints, static mounts, and auth routing | ~14.8 KB |
| `backend/database.py` | SQLite database manager, SHA-256 hashed clinician credentials, screening records | ~11.6 KB |
| `backend/inference_engine.py` | Singleton thread-safe inference & XAI orchestrator with checkpoint integrity verification | ~10.7 KB |
| `backend/pdf_generator.py` | Two-page publication-quality vector PDF clinical report generator (Matplotlib PdfPages) | ~12.1 KB |
| `backend/__init__.py` | Backend package initializer | 50 B |
| `frontend/index.html` | Semantic HTML5 Single-Page Application (8 views: landing, about, architecture, screening, results, history, dashboard, auth) | ~30.8 KB |
| `frontend/styles.css` | Clinical responsive design system, dark navy accents, accessible typography | ~23.2 KB |
| `frontend/app.js` | Modular SPA controller, hash routing, drag-and-drop dropzone, live progress, XAI rendering | ~19.0 KB |
| `src/verify_hybrid_inference.py`| Verified inference pipeline class reproducing training normalization and graph topology | ~10.3 KB |
| `src/screen.py` | Standalone CLI screening engine for terminal demonstrations | ~16.0 KB |
| `src/dataset.py` | Image transforms, dataset loader, normalization constants (Mean: 0.485, 0.456, 0.406; Std: 0.229, 0.224, 0.225) | ~11.7 KB |
| `src/models/vit_scratch.py` | 6-layer ViT from scratch, 192 embed dim, 3 attention heads, 196 patch tokens | 1,965,890 params |
| `src/models/graph_attention_v2.py`| 3-layer GATv2 architecture, dynamic multi-head relational attention | 222,722 params |
| `src/models/graph_classifier.py` | GraphClassifier wrapper supporting mean-pooling, GAT, and GATv2 topologies | -- |
| `src/models/hybrid_fusion.py` | ConcatenationFusion head (Linear 384->128 -> BatchNorm1d -> ReLU -> Dropout -> Linear 128->2) | 49,794 params |
| `src/graph/construction.py` | Hybrid graph generator (8-neighbor spatial adjacency + Top-8 cosine feature similarity) | ~7.2 KB |
| `src/graph/spatial.py` | 8-connected regular grid 2D spatial adjacency matrix builder | ~3.1 KB |
| `src/graph/similarity.py` | k-NN cosine similarity graph construction between ViT patch tokens | ~4.1 KB |
| `src/graph/batching.py` | Disjoint block-diagonal adjacency batching for PyTorch geometric tensor processing | ~2.1 KB |
| `src/xai/pipeline.py` | Unified coordinator generating Rollout, GAT overlays, Integrated Gradients, and Narratives | ~8.9 KB |
| `src/xai/vit_rollout.py` | ViT Attention Rollout with identity residual flow across all 6 transformer blocks | ~4.8 KB |
| `src/xai/gat_attribution.py` | GATv2 node degree centrality and Top-K relational edge attention extractor | ~5.3 KB |
| `src/xai/fusion_attribution.py` | Gauss-Legendre 50-step Integrated Gradients attribution with [0.05]^384 baseline | ~9.7 KB |
| `src/xai/clinical_narrative.py` | Rule-based deterministic clinical interpretation generator with risk stratification | ~5.5 KB |
| `src/xai/visualization.py` | High-resolution 4-panel diagnostic composite figure rendering | ~6.4 KB |
| `models/vit_scratch_best.pth` | Approved ViT checkpoint (SHA-256: `1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf`) | ~23.7 MB |
| `models/graph_hybrid_gatv2_best.pth`| Approved GATv2 checkpoint (SHA-256: `04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f`) | ~897 KB |
| `models/hybrid_vit_gat_concat_best.pth`| Approved Fusion checkpoint (SHA-256: `05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365`) | ~203 KB |
| `requirements.txt` | Runtime dependencies specification | ~0.5 KB |
| `.gitignore` | Exclusion patterns protecting runtime databases, uploads, and large weights | ~1.8 KB |

---

### Category B: Essential Research & Training Files
*Core academic research code, training routines, baseline models, dataset splits, manifests, and comparative ablation suites.*

| Path / Component | Description | Role / Status |
| :--- | :--- | :--- |
| `src/train_vit.py` | Vision Transformer from-scratch training routine | Research reproduction |
| `src/train_graph_models.py` | GAT and GATv2 graph network training routines | Research reproduction |
| `src/train_hybrid.py` | Concatenation and gated hybrid fusion training | Research reproduction |
| `src/train_efficientnet.py` | Transfer learning baseline (EfficientNet-B0) | Comparative baseline |
| `src/run_ablation_studies.py` | Systematic ablation study executor across all 15 architectures | Research benchmark |
| `src/evaluate_vit.py` | Test-set evaluation script for ViT models | Research evaluation |
| `src/evaluate_graph_models.py`| Test-set evaluation script for graph models | Research evaluation |
| `src/evaluate_hybrid.py` | Test-set evaluation script for hybrid models | Research evaluation |
| `src/evaluate_efficientnet.py`| Test-set evaluation script for EfficientNet | Research evaluation |
| `src/extract_patch_features.py`| Frozen ViT patch token feature extractor | Feature generation |
| `src/extract_graph_representations.py`| Frozen GATv2 graph representation extractor | Feature generation |
| `src/extract_handcrafted_features.py`| 143-dim LBP, GLCM, Color, Sobel extractor | Feature generation |
| `src/models/efficientnet_baseline.py`| Pretrained EfficientNet-B0 classifier architecture | Benchmark baseline |
| `src/models/graph_attention.py`| Original Veličković GAT v1 architecture | Ablation baseline |
| `src/models/handcrafted_fusion.py`| Multimodal fusion integrating deep and handcrafted features | Comparative baseline |
| `src/models/multiscale_fusion.py`| Multiscale patch token fusion architecture | Ablation baseline |
| `src/features/` | Handcrafted feature extraction modules (color, texture, edges) | Feature engineering |
| `models/*.pth` (20 other checkpoints)| All 20 historical ablation and baseline checkpoints | Preserved locally |
| `models/handcrafted_scaler.pt` | Scikit-learn StandardScaler parameters for 143-dim features | Tracked & preserved |
| `data/splits_primary/` | Approved 70:15:15 patient-stratified split manifests (`train.csv`, `val.csv`, `test.csv`) | Immutable benchmark |
| `data/splits/` | Primary dataset split links | Immutable benchmark |
| `data/splits_archive/` | Historical splits archive | Reproducibility |
| `Oral Cancer/` | Raw clinical image dataset (Cancer and Non-Cancer images) | Preserved on disk |
| `data/features_*/` | Cached binary tensor features for rapid ablation | Preserved locally |
| `reports/phase5_phase6_*.csv` | Master ablation benchmark results and training history | Research evidence |
| `reports/phase5_phase6_experiment_manifest.json`| Cryptographic metadata of all 15 trained architectures | Research evidence |
| `tests/test_*.py` (14 test suites)| Complete unit test suite (60 automated unit/integration tests) | Automated QA |

---

### Category C: Documentation & Reports
*Academic thesis documentation, user guides, API specifications, and formal audit records.*

| Path / Component | Description | Audience |
| :--- | :--- | :--- |
| `README.md` | Master thesis and conference documentation, architecture, benchmark tables | Faculty / Reviewers |
| `docs/system_architecture.md` | Comprehensive system design, data flows, thread-safety, XAI math | Faculty / Engineers |
| `docs/application_user_guide.md` | End-to-end clinician user manual with step-by-step walkthrough | Clinical Users |
| `docs/model_inference_pipeline.md` | Technical inference specification, tensor dimensions, normalization | Technical Evaluators |
| `docs/xai_methodology.md` | Detailed mathematical formulations of Rollout, GAT centrality, IG | Research Evaluators |
| `docs/limitations_and_disclaimers.md` | Formal clinical disclaimer, ethical AI statement, failure modes | Regulatory / Ethical |
| `reports/phase7_backend_*.md` | Backend implementation, API documentation, and testing reports | System Documentation |
| `reports/phase7_xai_*.md` | XAI implementation, independent audits, IG verification, limitations | XAI Documentation |
| `reports/phase7_model_selection_report.md` | Formal model selection and clinical risk evaluation | Research Documentation |
| `reports/phase7_checkpoint_architecture_provenance_audit.md` | Parameter count resolution and cryptographic provenance audit | Provenance Audit |
| `reports/phase7_final_integration_report.md` | Complete system integration report | Stage Verification |
| `reports/phase7_final_testing_report.md` | Complete automated testing verification record | QA Documentation |
| `reports/internal_archive/` | Historical phase 1-6 reports, dataset audit records, push reports | Internal Archive |
| `reports/xai_examples/` | Generated baseline XAI visualization artifacts | Verification Artifacts |
| `reports/*.png` (confusion matrices, learning curves) | Master publication figures | Thesis Figures |

---

### Category D: Temporary & Generated Files
*Runtime artifacts produced during screening operations, test executions, or local caching.*

| Path / Component | Description | Recommended Disposition |
| :--- | :--- | :--- |
| `data/oralscan.db` | Local SQLite database created during app testing | Preserve for demonstration; exclude from Git |
| `uploads/*.jpeg` | Clinical image uploads from test runs | Clean test uploads; preserve directory with `.gitkeep` |
| `outputs/visualizations/*.png` | Saliency maps, graph plots, composite figures from test runs | Clean test runs; preserve directory with `.gitkeep` |
| `outputs/reports/*.pdf` | Generated test clinical PDF reports | Clean test runs; preserve directory with `.gitkeep` |
| `scratch/matplotlib/` | Runtime font cache generated by Matplotlib | Disposable; delete |
| `scratch/*.py` | Audit helper scripts used during Phase 7 audits | Preserve in `scratch/`; already ignored by Git |
| `__pycache__/` | Python bytecode directories across 7 packages | Clean across repository |

---

### Category E: Disposable & Unwanted Files
*Redundant, transient, or clutter files that must be removed prior to demonstration.*

| Path / Component | Reason for Removal | Safety Assessment |
| :--- | :--- | :--- |
| All `__pycache__/` directories | Python compiled bytecode; regenerated automatically on import | Completely safe to delete |
| All `*.pyc` files | Transient compiled Python bytecode | Completely safe to delete |
| `uploads/*.jpeg` | Residual test upload files from development testing | Completely safe to clear |
| `outputs/reports/*.pdf` | Residual test PDF reports from development testing | Completely safe to clear |
| `outputs/visualizations/*.png` | Residual test visual overlays from development testing | Completely safe to clear |
| `scratch/matplotlib/` | Local user font cache | Completely safe to clear |

---

## 3. Preservation Guarantees

In accordance with strict project rules:
1. **Zero Checkpoint Loss:** All 23 model weight files (`models/*.pth`, including all ablation baselines and `models/handcrafted_scaler.pt`) are strictly preserved.
2. **Zero Dataset / Split Loss:** All original image directories (`Oral Cancer/`), split manifests (`data/splits_primary/`), and feature caches are untouched.
3. **Zero Historical Evidence Loss:** All research reports, confusion matrices, ablation tables, and training curves in `reports/` and `reports/internal_archive/` are preserved.
4. **Git Safety:** No commits or pushes will be made. `.gitignore` will be updated to ensure local runtime databases, uploads, and outputs are never accidentally staged.

Audit completed and approved for Phase 7 execution.
