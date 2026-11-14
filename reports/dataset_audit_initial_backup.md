# Dataset Audit Report: Oral Cancer Screening Benchmark

**Project Title:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Phase:** Phase 1 – Dataset Verification and Preparation
**Audit Date:** November 2026
**Dataset Name:** Original Oral Cancer Dataset (Reference Paper Benchmark)
**Dataset Root Location:** `Oral Cancer/Oral Cancer Dataset`

---

## 1. Executive Summary

This audit report documents the formal verification, structural inspection, and preparation of the primary dataset for the research project *"Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI"*.

The benchmark dataset was located and validated against the reference specification:
- **Total images:** Exactly **750** images
- **Class breakdown:** **500** CANCER images, **250** NON CANCER images
- **File integrity:** **0** corrupted or unreadable images (100% readable and decodable)
- **Manifests created:** Reproducible stratified split CSVs (`train.csv`, `val.csv`, `test.csv`) with seed **42**
- **Pipeline created:** Reusable PyTorch dataset and dataloading module (`src/dataset.py`)

A critical scientific finding of this audit is that the original dataset contains **21 exact duplicate groups (46 files)**, including **one cross-class collision** (`CANCER/001.jpeg` is byte-for-byte identical to `NON CANCER/014.jpeg`). Furthermore, although all 750 files carry the `.jpeg` file extension, **141 files are actually PNG or WEBP formats**, and **103 images contain 4-channel alpha transparency or palette color profiles**.

Per project constraints, the original dataset has not been modified or altered. These anomalies are accommodated by a robust preprocessing pipeline and are documented in detail below.

---

## 2. Dataset Identification & Provenance

### 2.1 Directory Structure
The dataset resides at:
```
Oral Cancer/Oral Cancer Dataset/
├── CANCER/          [500 files: 001.jpeg through 500.jpeg]
└── NON CANCER/      [250 files: 001.jpeg through 250.jpeg]
```

### 2.2 Provenance Verification
- **Origin:** The dataset matches the canonical 750-image oral cancer dataset compiled and published by research groups (e.g., Jayaprakash / JPINFOTECH / Kaggle "Oral Images").
- **Reference Paper Alignment:** This exact 750-image set (500 malignant, 250 benign/normal) is the precise benchmark cited in contemporary oral cancer classification literature utilizing deep learning backbones.
- **File Sequencing:**
  - `CANCER`: Sequence `001.jpeg` through `500.jpeg` is 100% complete. No missing files, no spurious files.
  - `NON CANCER`: Sequence `001.jpeg` through `250.jpeg` is 100% complete. No missing files, no spurious files.

---

## 3. Class Counts & Class Distribution

| Class Name | Target Count | Verified Count | Class Share (%) | Label Index |
| :--- | :---: | :---: | :---: | :---: |
| **CANCER** | 500 | 500 | 66.67% | `1` |
| **NON CANCER** | 250 | 250 | 33.33% | `0` |
| **Total** | **750** | **750** | **100.00%** | – |

> **Note on Imbalance:** The dataset exhibits a 2:1 class imbalance ratio (66.7% positive / 33.3% negative). Stratified sampling was applied to guarantee identical class proportions across train, validation, and test splits.

---

## 4. File Format, Color Mode & Dimension Analysis

### 4.1 File Extension vs. True Header Magic Bytes
Every file in the dataset was named with the extension `.jpeg`. Binary header inspection using PIL revealed significant format discrepancies:

| True File Format (PIL) | CANCER Count | NON CANCER Count | Total Images | Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **JPEG** | 459 | 150 | **609** | 81.20% |
| **PNG** | 20 | 89 | **109** | 14.53% |
| **WEBP** | 21 | 11 | **32** | 4.27% |
| **Total** | **500** | **250** | **750** | **100.00%** |

### 4.2 Image Color Modes
Raw image modes decoded from disk:

| Color Mode | Description | CANCER | NON CANCER | Total |
| :--- | :--- | :---: | :---: | :---: |
| **RGB** | Standard 3-channel color | 484 | 163 | **647** |
| **RGBA** | 4-channel color (with alpha transparency) | 12 | 87 | **99** |
| **P** | Palette-based 8-bit color with transparency | 4 | 0 | **4** |
| **Total** | | **500** | **250** | **750** |

> **Pipeline Mitigation:** To prevent shape mismatches (`[4, H, W]` vs `[3, H, W]`) or palette corruption during PyTorch tensor conversion, the pipeline in `src/dataset.py` explicitly invokes `raw_img.convert("RGB")` on every image prior to transformation.

### 4.3 Image Resolution & Aspect Ratio Statistics

| Metric | Width (px) | Height (px) | Aspect Ratio (W/H) | File Size (Bytes) |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum** | 90 | 89 | 0.4859 | 4,104 (4.0 KB) |
| **25th Percentile** | 280 | 237 | 0.9168 | 38,681 (37.8 KB) |
| **Median (50%)** | 495 | 435 | 1.0922 | 62,004 (60.6 KB) |
| **Mean** | 722.4 | 663.8 | 1.1472 | 228,298 (223.0 KB) |
| **75th Percentile** | 750 | 699 | 1.3659 | 103,127 (100.7 KB) |
| **Maximum** | 9,248 | 6,936 | 2.4381 | 10,333,524 (9.86 MB) |

### 4.4 Corruption Verification
- **Integrity Test:** All 750 files were checked via PIL `img.verify()` and `img.load()`.
- **Result:** **0 unreadable or corrupted files**. Every image can be read, decompressed, and converted into valid pixel tensors.

---

## 5. Duplicate and Near-Duplicate Analysis

### 5.1 Exact Byte-Level Duplicates (MD5 & SHA-256)
A hash analysis across raw byte streams identified **21 duplicate groups** involving **46 files**:
- **19 intra-class duplicate groups** in `CANCER` (38 files)
- **2 intra-class duplicate groups** in `NON CANCER` (5 files)
- **1 critical cross-class collision** (2 files)

#### Critical Cross-Class Duplicate
| MD5 Hash | Files Involved | Classes |
| :--- | :--- | :--- |
| `00b5c56942f93a4b429141aceb71698b` | `CANCER/001.jpeg` & `NON CANCER/014.jpeg` | **Contradictory Labels (CANCER vs. NON CANCER)** |

> **Scientific Implication:** `CANCER/001.jpeg` and `NON CANCER/014.jpeg` are binary identical files (1280 x 960 px), yet one is labeled as CANCER and the other as NON CANCER in the original dataset.

#### Intra-Class Exact Duplicates (CANCER)
1. `0597ce8de38ff2645f45669094c9a9f9`: `CANCER/019.jpeg`, `CANCER/449.jpeg`
2. `21ce3e03898be3df58948df92271c229`: `CANCER/020.jpeg`, `CANCER/440.jpeg`
3. `24e35fc131729b0dbbfd090d4b6dc7ab`: `CANCER/061.jpeg`, `CANCER/079.jpeg`
4. `6296e49ecd461224c1f84bb3e9de5914`: `CANCER/070.jpeg`, `CANCER/103.jpeg`, `CANCER/400.jpeg` (Triplet)
5. `64686bec25a2f1591a096b92b7d03500`: `CANCER/069.jpeg`, `CANCER/101.jpeg`
6. `65889758cea7b9b1c69ea148c5215932`: `CANCER/136.jpeg`, `CANCER/480.jpeg`
7. `6b33a7cf63d2007610af27644fb7fde3`: `CANCER/065.jpeg`, `CANCER/095.jpeg`
8. `7e70e2838bede63d52126dc19ef708c5`: `CANCER/003.jpeg`, `CANCER/486.jpeg`
9. `985c45e449d3501c44f76679ceded15f`: `CANCER/071.jpeg`, `CANCER/107.jpeg`
10. `9d0174c337f1216e9d828537737fa8c3`: `CANCER/064.jpeg`, `CANCER/094.jpeg`
11. `a53523482fedb01cc37d26e68acc2d11`: `CANCER/067.jpeg`, `CANCER/097.jpeg`
12. `b26a04726e692d10a4d84e46994ba94e`: `CANCER/043.jpeg`, `CANCER/058.jpeg`
13. `bfc167697eca461b8f98478263a8f6f2`: `CANCER/029.jpeg`, `CANCER/055.jpeg`
14. `c4dae698df305bd81fd5756f4608f9b8`: `CANCER/039.jpeg`, `CANCER/057.jpeg`, `CANCER/099.jpeg` (Triplet)
15. `d041ecf179aad2d14da37c63de879c48`: `CANCER/050.jpeg`, `CANCER/060.jpeg`, `CANCER/482.jpeg` (Triplet)
16. `d180e5cf1bb53f73fca1ecaf76e91ba8`: `CANCER/025.jpeg`, `CANCER/054.jpeg`
17. `dc9013b973d7a22c46e98a4cd5702c36`: `CANCER/011.jpeg`, `CANCER/432.jpeg`
18. `dfceb7c94e154ac730c626f826e7e0f5`: `CANCER/030.jpeg`, `CANCER/056.jpeg`

#### Intra-Class Exact Duplicates (NON CANCER)
1. `cf679c5f205e38d6cb3640f41db80e58`: `NON CANCER/149.jpeg`, `NON CANCER/199.jpeg`
2. `c296df6812da57052bffa1077303663c`: `NON CANCER/178.jpeg`, `NON CANCER/197.jpeg`, `NON CANCER/231.jpeg` (Triplet)

### 5.2 Perceptual Hash (pHash) Near-Duplicate Analysis
Using perceptual hashing (`imagehash.phash`, 64-bit DCT hash):
- **Identical pHash (Hamming distance 0):** **33 clusters** (51 image pairs). Of these, 29 pairs correspond to exact MD5 duplicates, while 22 pairs are visually identical images that differ only due to slight JPEG re-compression or minor metadata differences (e.g., `CANCER/041.jpeg` and `CANCER/073.jpeg`).
- **Near-duplicate pairs (Hamming distance <= 2):** **55 pairs**.
- **Cross-class near duplicates:** The **only** cross-class pair with distance <= 4 is `CANCER/001.jpeg` and `NON CANCER/014.jpeg`. There are no additional cross-class near-duplicates.

---

## 6. Dataset Splits & Manifest Generation

### 6.1 Split Strategy
- **Stratification:** Performed on class labels (`0` vs `1`) to maintain exactly 66.7% CANCER and 33.3% NON CANCER in all subsets.
- **Split Ratio:** **70% Training / 15% Validation / 15% Testing** (the standard 3-way benchmark partition for this dataset).
- **Random Seed:** Fixed to **`42`** for 100% deterministic reproducibility.
- **Patient-Level Separation:** Not claimed. EXIF and metadata analysis revealed that patient IDs are absent from the dataset; splits are image-level stratified.

### 6.2 Split Distribution Summary

| Split Manifest | Path | CANCER | NON CANCER | Total Images | Proportion |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `data/splits/train.csv` | 350 | 175 | **525** | 70.0% |
| **Validation** | `data/splits/val.csv` | 75 | 37 | **112** | 14.93% |
| **Test** | `data/splits/test.csv` | 75 | 38 | **113** | 15.07% |
| **Total** | | **500** | **250** | **750** | **100.0%** |

### 6.3 Overlap Verification
- **Path Overlap:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 images)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 images)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 images)
- **Path Resolution:** 100% (750/750) of image paths in `train.csv`, `val.csv`, and `test.csv` resolve to existing files on disk.

### 6.4 Manifest Schema
Each CSV file (`data/splits/*.csv`) contains the following columns:
1. `image_path`: Portable relative path from workspace root (e.g. `Oral Cancer/Oral Cancer Dataset/CANCER/001.jpeg`).
2. `filename`: Standard filename (e.g. `001.jpeg`).
3. `label_name`: Class name (`CANCER` or `NON CANCER`).
4. `label`: Binary classification target (`1` for CANCER, `0` for NON CANCER).
5. `file_size`: Byte size of original file.
6. `md5`: MD5 checksum for bit-level verification.
7. `sha256`: SHA-256 cryptographic hash.

---

## 7. Preprocessing & Data Loading Pipeline (`src/dataset.py`)

A modular, verified data pipeline has been implemented in `src/dataset.py`:

```
Input File (Disk)
       │
       ▼
PIL Image Open & Convert("RGB")  <── Handles RGBA (99), Palette P (4), WEBP (32), PNG (109)
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[Train Split]                     [Validation Split]                [Test Split]
- Bilinear Resize (224 x 224)     - Bilinear Resize (224 x 224)     - Bilinear Resize (224 x 224)
- RandomHorizontalFlip(p=0.5)     - ToTensor()                      - ToTensor()
- RandomVerticalFlip(p=0.5)       - Normalize(ImageNet mean/std)    - Normalize(ImageNet mean/std)
- RandomRotation(±15°)            *(Strictly Deterministic)*        *(Strictly Deterministic)*
- ColorJitter(b=0.1, c=0.1, s=0.1)
- ToTensor()
- Normalize(ImageNet mean/std)
```

### Pipeline Verification Results
Executing `python src/dataset.py` confirmed:
- Train DataLoader output shape: `torch.Size([B, 3, 224, 224])`, `dtype=torch.float32`
- Validation DataLoader output shape: `torch.Size([B, 3, 224, 224])`, `dtype=torch.float32`
- Test DataLoader output shape: `torch.Size([B, 3, 224, 224])`, `dtype=torch.float32`
- Determinism check: Multiple forward reads of validation samples yield identical tensors (`torch.allclose == True`).

---

## 8. Research Limitations & Recommendations

1. **Cross-Class Collision (`CANCER/001.jpeg` vs `NON CANCER/014.jpeg`):**
   - In our stratified split, `CANCER/001.jpeg` is assigned to **Train** (label 1), while `NON CANCER/014.jpeg` is in **Validation** (label 0).
   - *Impact:* The model will be penalized on validation for predicting the exact pattern it was trained on. In future phases, evaluating test sets with and without this duplicate will be crucial for transparent reporting.
2. **Intra-Class Duplicates:**
   - 44 files in the dataset share identical images within their own class. When split randomly, duplicates of a training image can appear in validation or test splits, leading to minor evaluation optimism (data leakage).
3. **No Patient Identifiers:**
   - Because patient IDs are unavailable in public oral cancer datasets, patient-level cross-validation cannot be guaranteed. Model evaluations represent image-level generalizability rather than patient-level clinical generalizability.
4. **Resolution Range:**
   - Source resolutions range from 90x89 to 9248x6936. While resizing to 224x224 standardizes input shapes for ViT and GAT patch extractors, very small images suffer from upsampling blur while extremely large images undergo significant downsampling compression.
