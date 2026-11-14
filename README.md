# Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

An end-to-end, reproducible deep learning framework for binary oral cancer screening (CANCER vs. NON CANCER) combining **Vision Transformers (ViT)**, **Graph Attention Networks (GAT / GATv2)**, and **Domain-Engineered Handcrafted Features (LBP, GLCM, Color Moments, Edge Statistics)**, trained strictly from scratch without external pretraining.

---

## Key Highlights & Scientific Principles

- **Zero Pretrained Weights:** Strictly initialized and trained from scratch with random weights (`weights=None`). No ImageNet, timm, or Hugging Face pretraining.
- **Duplicate-Aware Dataset Manifests:** Rigorous audit of the 750 original images resolved exact SHA-256 duplicates, near-duplicates, and contradictory labels, establishing a clean primary cohort of **703 representative images** (470 CANCER, 233 NON CANCER). Duplicate-aware image-level splitting was performed to prevent identified duplicate and near-duplicate groups from crossing dataset partitions. Patient-level independence could not be verified because patient identifiers were unavailable.
- **Multimodal Representations:** Integrates global visual context (ViT `[CLS]` token), localized topological context (8-NN spatial + feature-similarity graph via GATv2), and 143 classical image texture/color/edge descriptors.
- **Benchmarking Across 19 Model Configurations:** Comprehensive ablation study comparing unimodal baselines, dual-fusion heads, tri-modal architectures, multi-scale pooling, and end-to-end fine-tuning.

---

## Primary Dataset Structure

The dataset contains 703 representative images partitioned into stratified, duplicate-aware splits:

| Split Manifest | Total Images ($N$) | CANCER (label=1) | NON CANCER (label=0) | Percentage |
|---|---|---|---|---|
| `data/splits_primary/train.csv` | 486 | 322 | 164 | 69.1% |
| `data/splits_primary/val.csv` | 108 | 73 | 35 | 15.4% |
| `data/splits_primary/test.csv` | 109 | 75 | 34 | 15.5% |
| **Total Primary Dataset** | **703** | **470** | **233** | **100.0%** |

*Note: Raw clinical images are preserved locally and excluded from this repository. Download the original dataset into `Oral Cancer/Oral Cancer Dataset/` as structured in `data/splits_primary/train.csv`.*

---

## Requirements & Environment

- **Python:** 3.11+
- **PyTorch:** 2.6.0 (CUDA 12.4 compatible if GPU acceleration is available)
- **Key Python Packages:** `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `opencv-python`, `pillow`, `matplotlib`

### Installation
```bash
# Clone the repository
git clone https://github.com/Sakshi0008/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI.git
cd Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies (with CUDA 12.4 support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn opencv-python pillow matplotlib
```

---

## Step-by-Step Pipeline Execution

### Step 1: Feature Extraction & Standardization
To regenerate intermediate patch, graph, and handcrafted feature tensors from raw images:
```bash
python src/extract_patch_features.py          # Generates data/features_vit_scratch/*.pt
python src/extract_graph_representations.py   # Generates data/features_graph/*.pt
python src/extract_handcrafted_features.py    # Generates data/features_handcrafted/*.pt and fits scaler
```

### Step 2: Training Models from Scratch
- **Phase 2 (EfficientNet-B0 Scratch Baseline):**
  ```bash
  python src/train_efficientnet.py
  python src/evaluate_efficientnet.py
  ```
- **Phase 3 (Vision Transformer Scratch Baseline):**
  ```bash
  python src/train_vit.py
  python src/evaluate_vit.py
  ```
- **Phase 4 (Graph Construction, GAT & GATv2):**
  ```bash
  python src/train_graph_models.py
  python src/evaluate_graph_models.py
  ```
- **Phases 5 & 6 (Master Ablations & Hybrid Fusion):**
  ```bash
  python src/run_ablation_studies.py
  ```

### Step 3: Run Unit Test Suite
To verify model forward passes, graph batching, feature scalers, and dimensions across all 37 tests:
```bash
python -m unittest discover tests
```

---

## Summary of Results (Holdout Test Split, $N=109$)

| Model ID | Architecture / Strategy | Test Accuracy | Cancer Sensitivity | Specificity | Cancer F1 | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **07** | **Handcrafted Only (All 143 Features)** | **87.16%** | **86.67%** | **88.24%** | **0.9028** | **0.9322** | **0.9654** |
| **07a** | Handcrafted LBP Only (59 Dims) | **83.49%** | **81.33%** | **88.24%** | **0.8714** | **0.9161** | **0.9608** |
| **Phase 2** | EfficientNet-B0 (Scratch Baseline) | **80.73%** | **80.00%** | **82.35%** | **0.8511** | **0.8912** | **0.9176** |
| **08** | ViT [CLS] + Handcrafted Fusion | **80.73%** | **80.00%** | **82.35%** | **0.8511** | **0.8761** | **0.9399** |
| **09** | GATv2 Graph + Handcrafted Fusion | **80.73%** | **80.00%** | **82.35%** | **0.8511** | **0.8690** | **0.9213** |
| **02** | Hybrid GATv2 (Graph Baseline) | **77.06%** | **81.33%** | **67.65%** | **0.8299** | **0.8298** | **0.9151** |
| **10** | Tri-Modal Concat (ViT+GATv2+HC) | **77.06%** | **77.33%** | **76.47%** | **0.8227** | **0.8451** | **0.9261** |
| **14** | Strategy B: End-to-End Fine-Tuning | **76.15%** | **76.00%** | **76.47%** | **0.8143** | **0.8251** | **0.9070** |
| **05** | ViT + GATv2 Gated Fusion | **76.15%** | **77.33%** | **73.53%** | **0.8169** | **0.8063** | **0.8866** |
| **06** | ViT + GATv2 Residual Fusion | **76.15%** | **80.00%** | **67.65%** | **0.8219** | **0.8094** | **0.9002** |
| **01** | ViT-Scratch Baseline (CLS Only) | **75.23%** | **76.00%** | **73.53%** | **0.8085** | **0.8141** | **0.8957** |

---

## Checkpoints and Feature Caches Availability

Model checkpoints are not included in this repository. They will be made available separately when a release is published. Intermediate feature caches can be regenerated using the verified extraction scripts.
- Intermediate feature caches can be regenerated directly from raw images using the extraction scripts in `src/`.
- Training-only standardization parameters are tracked in this repository via `models/handcrafted_scaler.pt` to ensure exact, reproducible scaling.

---

## Methodological Disclosures

1. **Internal Holdout Partition:** The test set ($N=109$) represents an internal holdout split from the single-source dataset, evaluated across sequential project phases (Phases 2 through 6). Multi-center external cohort validation is required before clinical deployment.
2. **Class-Weight Implementation:** All 15 Phase 5 and 6 models were trained with loss weights $[w_0=1.51875, w_1=0.7454]$ derived from an initial 326/160 count estimation ($+3.77\%$ ratio deviation from the approved 322/164 split $[1.4817, 0.7547]$), as documented in `reports/phase5_phase6_class_weight_audit.md`.
3. **Image-Level Representation:** Annotations represent 703 non-redundant representative images after identified redundant copies and contradictory-label images were excluded, rather than 703 verified distinct patients, as patient hospital identifiers were unavailable in the public source dataset.
