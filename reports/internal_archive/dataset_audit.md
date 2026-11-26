# Comprehensive Dataset Audit & Preparation Report

**Project Title:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Phase:** Phase 1 – Final Dataset Verification, Duplicate Resolution & Preprocessing Pipeline
**Date:** November 2026
**Dataset Root Location:** `Oral Cancer/Oral Cancer Dataset`
**Initial Pre-Review Backup:** `data/splits_initial_backup/` and `reports/dataset_audit_initial_backup.md`

---

## 1. Executive Summary & Experimental Architecture

This final report establishes the definitive dataset baselines for the research project *"Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI"*.

To ensure scientific rigor, three distinct dataset configurations are formalized:

```
[A. Raw Benchmark Dataset] (750 Images: 500 Cancer, 250 Non-Cancer)
       │
       ▼  Exclusion of Contradictory Duplicate Collision (001.jpeg & 014.jpeg)
[B. Duplicate-Aware Dataset] (748 Images: 499 Cancer, 249 Non-Cancer)
       │  - Group-aware split (data/splits/)
       │  - Retained for Sensitivity Analysis
       │
       ▼  Deterministic Representative Selection (1 per duplicate group)
[C. Duplicate-Collapsed Primary Dataset] (703 Images: 470 Cancer, 233 Non-Cancer)
          - Primary Experimental Split (data/splits_primary/)
          - Train: 486 | Val: 108 | Test: 109
          - Primary benchmark for model training and evaluation
```

### Key Dataset Configurations
1. **Configuration A (Raw Benchmark, 750 Images):** Complete raw benchmark preserving all physical files as originally sourced.
2. **Configuration B (Duplicate-Aware Secondary Benchmark, 748 Images in `data/splits/`):**
   - Excludes the 2 contradictory-label collision images (`CANCER/001.jpeg` and `NON CANCER/014.jpeg`).
   - Retains same-class multi-copy duplicate scenes, keeping every duplicate group strictly within one split to prevent leakage.
   - **Purpose:** Retained as a secondary sensitivity-analysis benchmark to measure whether retaining repeated copies of clinical lesions affects model generalizability.
3. **Configuration C (Duplicate-Collapsed Primary Benchmark, 703 Images in `data/splits_primary/`):**
   - Retains **exactly one representative image** for each of the 35 same-class duplicate groups based on an objective, deterministic rule (highest resolution, file size, and filename tie-breaker).
   - Excludes the 45 redundant copies (logged in `data/splits_primary/excluded_redundant_copies.csv`).
   - **Purpose:** Primary experimental benchmark for model training and evaluation. Guarantees that no individual clinical scene receives disproportionate weighting or gradient influence during training, and ensures that test metrics reflect true image-level generalizability across distinct clinical scenes.

---

## 2. Dataset Counts & Class Distributions

| Dataset Tier | CANCER (1) | NON CANCER (0) | Total Images | Class Ratio (C : NC) | Purpose / Manifest Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A. Raw Benchmark** | 500 | 250 | **750** | 66.67% : 33.33% | Raw files preserved in `Oral Cancer/Oral Cancer Dataset/` |
| **B. Duplicate-Aware** | 499 | 249 | **748** | 66.71% : 33.29% | Sensitivity benchmark in `data/splits/` |
| **C. Duplicate-Collapsed Primary** | 470 | 233 | **703** | 66.86% : 33.14% | **Primary Experimental Benchmark** in `data/splits_primary/` |

- **Preservation:** 100% of the 750 original files remain intact on disk without modification or deletion.
- **Class Proportions:** The duplicate-collapsed primary dataset preserves the canonical ~2:1 class balance (66.86% malignant vs 33.14% non-malignant).

---

## 3. Primary Duplicate-Collapsed Dataset Structure

### 3.1 Split Distribution Summary (Primary Dataset, $N = 703$)

| Split Manifest | Path | CANCER (1) | NON CANCER (0) | Total Images | Share (%) | Target Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `data/splits_primary/train.csv` | 322 | 164 | **486** | **69.13%** | ~70.0% |
| **Validation** | `data/splits_primary/val.csv` | 73 | 35 | **108** | **15.36%** | ~15.0% |
| **Test** | `data/splits_primary/test.csv` | 75 | 34 | **109** | **15.50%** | ~15.0% |
| **Total Primary** | | **470** | **233** | **703** | **100.00%** | 100.0% |

- **Preservation of Original Split Assignments:**
  Every representative image was retained in the **exact same split** originally assigned during group-aware stratified splitting. No image was moved across splits to chase exact fractions.
- **Class Proportions Across Primary Splits:**
  - Train: 66.26% CANCER / 33.74% NON CANCER
  - Validation: 67.59% CANCER / 32.41% NON CANCER
  - Test: 68.81% CANCER / 31.19% NON CANCER
- **Uniqueness Guarantee:** Exactly **0 duplicate scenes** appear more than once in the primary dataset. Every group ID appears exactly once across `train.csv`, `val.csv`, and `test.csv`.

### 3.2 Representative Selection Methodology
For each of the 35 same-class multi-image duplicate groups, exactly one representative was selected based on an objective, deterministic rule:
1. **Primary Criterion:** Highest spatial resolution (total pixel count: $\text{width} \times \text{height}$).
2. **Secondary Tie-Breaker:** Largest file size in bytes (indicative of lower JPEG compression artifacts).
3. **Tertiary Tie-Breaker:** Lexicographically smallest filename (e.g. `019.jpeg` before `449.jpeg`).

The full audit log detailing all 35 groups, their representatives, and excluded copies is recorded in [reports/duplicate_representatives_report.md](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/reports/duplicate_representatives_report.md) and [data/splits_primary/excluded_redundant_copies.csv](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/data/splits_primary/excluded_redundant_copies.csv).

---

## 4. Resolution of Exact Duplicate Counts & Pairwise Combinations

Across all 750 images, SHA-256 hashing identified **725 unique hashes**, with **21 duplicated hashes** across 46 files:

| Duplicate Category | Group Size ($k$) | Number of Groups | Total Files ($\sum k$) | Pairs per Group $\binom{k}{2}$ | Total Pairwise Combinations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cross-Class Collision** | 2 | 1 | 2 | 1 | **1** |
| **CANCER Pairs** | 2 | 15 | 30 | 1 | **15** |
| **CANCER Triplets** | 3 | 3 | 9 | 3 | **9** |
| **NON CANCER Pairs** | 2 | 1 | 2 | 1 | **1** |
| **NON CANCER Triplets** | 3 | 1 | 3 | 3 | **3** |
| **Total Exact Duplicates** | – | **21 groups** | **46 files** | – | **29 pairs** |

- **Total Unique SHA-256 Hashes:** 725
- **Total Duplicate Groups ($k > 1$):** 21 groups
- **Total Files in Duplicate Groups:** $2 + 30 + 9 + 2 + 3 = \mathbf{46\text{ files}}$
- **Total Unique Pairwise Combinations:** $1 + 15 + 9 + 1 + 3 = \mathbf{29\text{ pairs}}$

---

## 5. Perceptual Hash Grouping Evidence & Transitivity Verification

Using 64-bit DCT perceptual hashing (`imagehash.phash`) at threshold $\le 6$:
- **Pairwise Distance Breakdown:**
  - Distance 0: 51 pairs (29 exact SHA-256 duplicate pairs + 22 visually identical near-duplicate pairs).
  - Distance 2: 4 pairs (minor compression differences with identical aspect ratios).
  - Distance 6: 3 pairs (re-scaled/cropped versions of the same clinical lesion, e.g., `CANCER/063.jpeg` [635×406] $\leftrightarrow$ `CANCER/092.jpeg` [635×388]).
  - Distances 1, 3, 4, 5: 0 pairs exist in the dataset.
- **Transitivity Verification:**
  - Connected component clustering was tested for runaway transitivity ($A \leftrightarrow B$ and $B \leftrightarrow C$ producing distant pairs $A \leftrightarrow C$).
  - **Empirical Result:** **Zero transitive expansion occurred.** The maximum internal pairwise distance within any connected component in the dataset is strictly $\le \mathbf{6}$.
- **Largest Connected Component:**
  - The largest component contains **4 images**: `['NON CANCER/167.jpeg', 'NON CANCER/175.jpeg', 'NON CANCER/187.jpeg', 'NON CANCER/208.jpeg']`.
  - Every single pairwise distance between these 4 images is **0** (they are identical multiscale copies of the same palate photograph).
- **Patient Identifier Disclaimer:**
  These groupings are established strictly to prevent data leakage of identical or cropped images across splits. We make **no claim** that perceptually grouped images belong to the same patient, as patient IDs are unavailable.

---

## 6. Preprocessing Architecture: Direct Resizing vs. Letterbox Padding

### 6.1 Training Set Aspect Ratio Distribution ($N = 486$)
- **Mean Aspect Ratio ($W/H$):** $1.151 \pm 0.297$
- **Median:** $1.089$
- **25th – 75th Percentile:** $[0.932, 1.366]$ (50% of images are within 15% of square)
- **Extreme Narrow Example:** `NON CANCER/201.jpeg` ($178 \times 283$, $\text{AR} = 0.629$)
- **Extreme Wide Example:** `CANCER/420.jpeg` ($551 \times 226$, $\text{AR} = 2.438$)
- **Square Example:** `CANCER/127.jpeg` ($256 \times 256$, $\text{AR} = 1.000$)

### 6.2 Comparison for Hybrid EfficientNet + ViT + GAT
1. **Direct Bilinear Resizing ($224 \times 224$):**
   - **Canvas Utilization:** 100% of all 196 patch tokens ($14 \times 14$) represent real oral mucosal tissue.
   - **GAT Graph Construction:** Every node in the graph represents anatomical mucosal tissue. Zero dummy padding nodes.
   - **Trade-off:** Anamorphic geometric distortion in extreme wide/tall images (e.g. `CANCER/420.jpeg` width compressed by ~59%).
2. **Aspect-Ratio Letterbox Padding ($224 \times 224$):**
   - **Geometric Fidelity:** 100% preserves lesion circularity and aspect ratio.
   - **Token Dilution:** For extreme wide images ($\text{AR} = 2.44$), **58.9% of the canvas and 112 out of 196 patch tokens are pure black padding zeros**.
   - **GAT Mitigation via Masking:** Padding does not invalidate GAT if attention masking is applied:
     $$\text{Attn}_{ij} = \text{softmax}\left(\text{Score}_{ij} + (1 - M_j) \cdot (-10^9)\right)$$
     where $M \in \{0, 1\}^{196}$ is the valid patch mask produced by `LetterboxPad.get_mask()`.
3. **Recommendation:**
   Direct resizing is the primary default mode in `src/dataset.py` (`resize_mode="direct"`), with letterbox padding (`resize_mode="letterbox"`) fully implemented for comparative evaluation during Phase 2 model experimentation.

---

## 7. Normalization Strategy for Training From Scratch

Because models in this project must be trained **from scratch** with random weight initialization (no pretrained weights):
- ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) do not reflect the red-dominant vascularized oral mucosa.
- Channel-wise mean and standard deviation computed **strictly on the 486 primary training images** (`data/splits_primary/train.csv`), with zero access to validation or test images:
  $$\text{Mean}_{\text{primary}} = [R = \mathbf{0.6432}, G = \mathbf{0.4537}, B = \mathbf{0.4275}]$$
  $$\text{Std}_{\text{primary}}  = [R = \mathbf{0.2189}, G = \mathbf{0.2118}, B = \mathbf{0.2090}]$$
- The red channel mean ($0.6432$) is significantly higher than ImageNet ($0.4850$), reflecting physiological mucosal erythema. Centering with these statistics standardizes input features around zero mean for scratch initialization.

---

## 8. Conservative Clinical Augmentation Review

| Augmentation Technique | Setting | Clinical Appropriateness | Decision & Rationale |
| :--- | :---: | :---: | :--- |
| **Horizontal Flip** | $p = 0.5$ | **High** | **Retained:** Preserves bilateral oral symmetry (left/right buccal mucosa, lateral tongue margins, bilateral palate). |
| **Vertical Flip** | $p = 0.5$ | **Unsuitable** | **REMOVED:** Inverting the hard/soft palate (superior) and floor of the mouth (inferior) creates anatomically impossible orientations. |
| **Rotation** | $\pm 10^\circ$ | **High** | **Retained ($\pm 10^\circ$):** Models natural handheld camera or patient head tilt without inverted orientation. |
| **Color Jitter** | Brightness/Contrast $\pm 0.05$ | **Sensitive** | **Restricted:** Hue and saturation shifts are set to **0.0** to preserve the critical visual distinction between **erythroplakia** (red velvet patches) and **leukoplakia** (white keratotic plaques). Brightness/contrast variation ($\pm 0.05$) safely models exposure differences. |

---

## 9. Automated Verification Results

All 6 automated tests on the primary dataset passed with zero errors:

| Check # | Test Description | Condition Tested | Result |
| :---: | :--- | :--- | :---: |
| **1** | Sample accounting | Primary (703) + Redundant (45) + Contradictory (2) = 750 | **PASSED (750/750)** |
| **2** | Scene uniqueness | Each duplicate group appears at most once in primary | **PASSED (Max count = 1)** |
| **3** | Path disjointness | Zero path intersection across train, val, test, redundant, contradictory | **PASSED (0 overlap)** |
| **4** | Path existence | 100% of 703 primary image paths resolve on disk | **PASSED (703/703 verified)** |
| **5** | Tensor output format | DataLoaders yield $(B, 3, 224, 224)$ float32 tensors | **PASSED (All splits)** |
| **6** | Normalization computation | Training stats computed strictly on 486 training images | **PASSED (Zero leakage)** |

---

## 10. Summary of Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **Primary Train Manifest** | `data/splits_primary/train.csv` | 486 unique clinical scenes (322 CANCER, 164 NON CANCER). |
| **Primary Val Manifest** | `data/splits_primary/val.csv` | 108 unique clinical scenes (73 CANCER, 35 NON CANCER). |
| **Primary Test Manifest** | `data/splits_primary/test.csv` | 109 unique clinical scenes (75 CANCER, 34 NON CANCER). |
| **Excluded Redundant Copies** | `data/splits_primary/excluded_redundant_copies.csv` | 45 redundant duplicate copies with group IDs and selection reasons. |
| **Representatives Report** | `reports/duplicate_representatives_report.md` | Detailed justification for the retained representative of each duplicate group. |
| **Secondary Benchmark** | `data/splits/` (`train.csv`, `val.csv`, `test.csv`) | 748-image duplicate-aware benchmark for sensitivity analysis. |
| **Contradictory Exclusions** | `data/splits/excluded_images.csv` | Record of `CANCER/001.jpeg` and `NON CANCER/014.jpeg`. |
| **Complete Group Catalog** | `data/splits/duplicate_groups.csv` | Catalog of all 704 groups across the original 750 images. |
| **Dataset Loading Pipeline** | `src/dataset.py` | PyTorch Dataset and DataLoader module supporting primary and secondary splits. |
| **Comprehensive Report** | `reports/dataset_audit.md` | This document. |
| **Backups** | `data/splits_initial_backup/`, `reports/dataset_audit_initial_backup.md` | Preserved initial manifests and audit reports. |

---

## 11. Remaining Limitations & Research Considerations
1. **Absence of Patient Metadata:** While duplicate clustering eliminates leakage between identical or slightly cropped images, separate photographs of the same patient taken from completely different angles or at different clinic visits ($p\text{Hash distance} > 6$) cannot be grouped without hospital medical record numbers.
2. **2:1 Class Imbalance:** CANCER represents 66.86% of primary images. Weighted cross-entropy or focal loss is recommended during training.
3. **Source Resolution Range:** Original dimensions range from $90 \times 89$ px to $9,248 \times 6,936$ px. Very low-resolution samples lose high-frequency textural detail when upscaled to $224 \times 224$.
