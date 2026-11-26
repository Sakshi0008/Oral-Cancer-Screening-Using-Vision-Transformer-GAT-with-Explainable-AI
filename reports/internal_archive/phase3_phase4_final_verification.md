# Phase 3 & Phase 4 Final Technical and Scientific Verification Audit

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Audit Date:** November 13, 2026
**Auditor:** Antigravity AI Research Verification System
**Scope:** Strict, read-only technical and scientific audit of Phase 3 (Vision Transformer from Scratch) and Phase 4 (Graph Construction, GAT, and GATv2) source code, checkpoints, feature caches, test predictions, and scientific reports.
**Final Assessment:** **PASS WITH MINOR CORRECTIONS** (Methodological implementation is completely verified; scientific reporting requires cautious language regarding sample size constraints, operating trade-offs, and generalizability).

---

## 1. Files Inspected

The following source files, cached tensors, model checkpoints, and reports were audited directly:

| Category | File Path | Verified Status |
|---|---|---|
| **ViT Architecture** | `src/models/vit_scratch.py` | Verified (Random initialization, $1.97\text{M}$ params) |
| **ViT Training & Eval** | `src/train_vit.py`, `src/evaluate_vit.py` | Verified (Zero test leakage, class weights, seed 42) |
| **ViT Checkpoint** | `models/vit_scratch_best.pth` | Verified (Epoch 34, Val loss $0.4880$) |
| **Feature Extraction** | `src/extract_patch_features.py` | Verified (Deterministic eval mode, no augmentations) |
| **Feature Caches** | `data/features_vit_scratch/*.pt` | Verified (Train: 486, Val: 108, Test: 109) |
| **Graph Construction** | `src/graph/spatial_graph.py` | Verified (196 nodes, 924 directed edges) |
| | `src/graph/similarity_graph.py` | Verified (Cosine similarity, $k=8$, symmetric, self-loops) |
| | `src/graph/hybrid_graph.py` | Verified (Deduplicated set union clamp) |
| | `src/graph/graph_builder.py` | Verified (Batch isolation, dense & sparse representations) |
| | `src/graph/graph_pooling.py` | Verified (GlobalMeanPooling, AttentionGraphPooling) |
| **GNN Architectures** | `src/models/graph_attention.py` | Verified (Multi-head static GAT, pure PyTorch) |
| | `src/models/graph_attention_v2.py`| Verified (Multi-head dynamic GATv2, Brody et al.) |
| | `src/models/graph_classifier.py` | Verified (Modular end-to-end and cached pipelines) |
| **Graph Training & Eval**| `src/train_graph_models.py`, `src/evaluate_graph_models.py` | Verified (All 6 experiments, validation early stopping) |
| **Graph Checkpoints** | `models/graph_*_best.pth` (6 files)| Verified (Independently initialized, saved at best val loss) |
| **Test Predictions** | `reports/phase4_experiment_comparison.csv` | Verified (Exact test metrics, continuous AUC probabilities) |
| **Scientific Reports** | `reports/phase3_vit_report.md` | Audited for scientific phrasing |
| | `reports/phase4_graph_construction_report.md` | Audited for scientific phrasing |
| | `reports/phase4_gat_gatv2_report.md` | Audited for scientific phrasing |

---

## 2. Verified Technical Facts

1. **Zero Pretrained Weights:** All modules (`VisionTransformerScratch`, `GAT`, `GATv2`, `GraphClassifier`) use pure PyTorch constructors initialized with random distributions. `is_pretrained == False` is enforced in all configurations. No external weights, ImageNet parameters, timm, or Hugging Face checkpoints were loaded.
2. **Dataset Partition Integrity:** All experiments used exclusively the approved primary duplicate-collapsed splits (`data/splits_primary/`): Train $= 486$, Val $= 108$, Test $= 109$ ($N=703$). No images or labels were altered.
3. **Reproducibility:** Seed fixed at `42` across Python, NumPy, PyTorch CPU, and CUDA. CUDA deterministic flags were strictly enabled (`cudnn.deterministic = True`, `cudnn.benchmark = False`).
4. **Automated Unit Testing:** All 21 tests in `tests/test_vit.py`, `tests/test_graph_construction.py`, `tests/test_graph_batching.py`, `tests/test_gat.py`, and `tests/test_gatv2.py` pass cleanly.

---

## 3. ViT Checkpoint & Feature Cache Audit

### 3.1 Checkpoint Integrity (`models/vit_scratch_best.pth`)
- **Epoch:** Stored state corresponds strictly to **Epoch 34**.
- **Metrics at Epoch 34:** Validation Loss $= 0.48804$, Validation Accuracy $= 80.56\%$, Validation $F_1 = 0.8531$.
- **Early Stopping Trigger:** Training continued until Epoch 46 without improvement over Epoch 34 ($46 - 34 = 12$ consecutive epochs), halting training automatically.

### 3.2 Feature Extraction Integrity (`src/extract_patch_features.py`)
Code inspection of the feature extraction pipeline confirms:
- **Evaluation Mode & Dropout:** `model.eval()` was explicitly called. Dropout layers in the patch embedding, self-attention blocks, and MLP were deactivated.
- **Gradient Freezing:** All forward passes were wrapped in `with torch.no_grad():`.
- **Deterministic Preprocessing:** Transforms were created using `get_transforms("test", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")`.
  - Resize: Bilinear interpolation to $224 \times 224$.
  - Normalization: Dataset primary training mean $\mu = [0.6432, 0.4537, 0.4275]$ and standard deviation $\sigma = [0.2189, 0.2118, 0.2090]$.
  - Augmentation: Zero random flips, zero random rotations, and zero color jitter were applied. All splits (train, val, test) received 100% deterministic preprocessing during feature extraction.
- **Cache Tensor Verification:**
  - `data/features_vit_scratch/train_patch_features.pt`: Patches `torch.Size([486, 196, 192])`, Labels `torch.Size([486])`.
  - `data/features_vit_scratch/val_patch_features.pt`: Patches `torch.Size([108, 196, 192])`, Labels `torch.Size([108])`.
  - `data/features_vit_scratch/test_patch_features.pt`: Patches `torch.Size([109, 196, 192])`, Labels `torch.Size([109])`.
  - Manifest Matching: `train_f['image_paths'] == train_csv['image_path'].tolist()` evaluated to **True**.
  - Label Matching: `(train_f['labels'].numpy() == train_csv['label'].values).all()` evaluated to **True**.
  - Identical 100% path and label equality verified for validation and test splits. Zero cross-split mixing detected.

---

## 4. Graph Experiment Leakage Audit

### 4.1 Data Loader Isolation (`src/train_graph_models.py`)
- `load_cached_datasets()` loads exclusively `train_patch_features.pt` for the training loader and `val_patch_features.pt` for the validation loader.
- Test features (`test_patch_features.pt`) were never opened, passed, or referenced inside `src/train_graph_models.py`.
- Balanced class weights were derived strictly from training labels:
  $$w_0 = \frac{486}{2 \times 164} = 1.4817, \quad w_1 = \frac{486}{2 \times 322} = 0.7547$$
- Model selection was governed exclusively by minimum validation loss on `val_loader`.
- Test evaluation was conducted exclusively post-training in `src/evaluate_graph_models.py`.

### 4.2 Checkpoint Independent Initialization
Each of the 6 graph models was instantiated fresh inside `train_single_experiment()` with `set_seed(42)` resetting the random generator before layer construction. Each GNN model received independent Xavier initialization with no shared or reused graph parameters.

---

## 5. Graph Construction Audit

### 5.1 Spatial Adjacency Graph (`src/graph/spatial_graph.py`)
- **Nodes:** Exactly $196$ nodes corresponding to the $14 \times 14$ patch grid.
- **Undirected Edges (without self-loops):** $2 \times 14 \times 13 = 364$ undirected edges $\implies 728$ directed edges.
- **Self-Loops:** Exactly $196$ diagonal entries ($A_{ii} = 1.0$).
- **Total Directed Edges:** $728 + 196 = \mathbf{924}$ directed edges.
- **Degree Verification:**
  - $4$ corners: degree $= 2 \text{ (neighbours)} + 1 \text{ (self-loop)} = \mathbf{3}$.
  - $48$ border patches: degree $= 3 + 1 = \mathbf{4}$.
  - $144$ interior patches: degree $= 4 + 1 = \mathbf{5}$.
  - Total: $4 \times 3 + 48 \times 4 + 144 \times 5 = 12 + 192 + 720 = \mathbf{924}$.

### 5.2 Feature-Similarity Graph (`src/graph/similarity_graph.py`)
- **Cosine Similarity:** Computed via $\tilde{X} \tilde{X}^T$ on $L_2$-normalized patch features (`F.normalize(features, p=2, dim=-1)`).
- **Self-Connection Exclusion:** Diagonal masked to $-10^9$ before `torch.topk(..., k=8)`.
- **Bidirectionality:** Symmetrized via `torch.clamp(adj + adj.transpose(1, 2), min=0.0, max=1.0)`.
- **Self-Loops:** Added explicitly via `adj.masked_fill(eye_mask, 1.0)`.
- **Degree Guarantee:** Every node has minimum degree $\ge k + 1 = 9$.

### 5.3 Hybrid Graph (`src/graph/hybrid_graph.py`)
- **Deduplicated Union:** $A_{\text{hybrid}} = \text{clamp}(A_{\text{spatial}} + A_{\text{similarity}}, 0.0, 1.0)$.
- **Integrity:** Binary clamp guarantees no duplicate edges or values $> 1.0$. All node features remain identical.

### 5.4 Batch Isolation (`src/graph/graph_builder.py`)
- In dense representation $[B, 196, 196]$, batch matrix multiplications operate independently along dimension 0.
- In sparse edge index $[2, \sum E_b]$, node indices for image $b$ are offset by $b \times 196$. Automated test `test_batched_edge_index_isolation` confirmed:
  $$\lfloor u / 196 \rfloor == \lfloor v / 196 \rfloor \quad \forall (u, v) \in \mathcal{E}$$
  Zero cross-image edges exist.

---

## 6. GAT and GATv2 Mathematical Implementation Audit

### 6.1 GAT Implementation (`src/models/graph_attention.py`)
Implements the original Veličković et al. (2018) formulation:
$$z_i = W h_i, \quad z_j = W h_j$$
$$e_{ij} = \text{LeakyReLU}\left(a_{\text{src}}^T z_i + a_{\text{dst}}^T z_j\right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
$$h'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} z_j\right)$$
- Masking: Non-neighbor positions masked with $-10^9$ prior to Softmax.
- Attention dropout and feature dropout applied at $p=0.1$.
- Residual skip connection and LayerNorm applied after concatenation.

### 6.2 GATv2 Implementation (`src/models/graph_attention_v2.py`)
Implements the Brody et al. (2021) dynamic attention formulation:
$$z_{\text{src}, i} = W_{\text{src}} h_i, \quad z_{\text{dst}, j} = W_{\text{dst}} h_j, \quad z_{\text{val}, j} = W_{\text{val}} h_j$$
$$e_{ij} = a^T \text{LeakyReLU}\left(z_{\text{src}, i} + z_{\text{dst}, j}\right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
$$h'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} z_{\text{val}, j}\right)$$

### 6.3 Mathematical Distinction
The code proves that GAT and GATv2 are fundamentally distinct:
1. **Order of Operations:** In GAT, the projection $a^T$ operates outside after the LeakyReLU on pre-summed linear projections ($a_{\text{src}}^T W h_i + a_{\text{dst}}^T W h_j$). In GATv2, the LeakyReLU non-linearity is applied to the pairwise sum **before** the inner product with attention vector $a$.
2. **Dynamic Ranking:** In GATv2, the rank order of attention coefficients depends on the query node, eliminating the static attention bottleneck.

---

## 7. Prediction & Confusion Matrix Investigation

The user specifically requested an investigation into whether the models produced identical confusion matrices and whether test prediction vectors were identical or copied.

### 7.1 Empirical Investigation of Predictions and Probabilities

A sample-by-sample inspection across all 109 test instances revealed the following ground-truth facts:

#### Investigation A: Spatial GAT vs. Similarity GAT
1. **Confusion Matrices:**
   - Spatial GAT: $\begin{bmatrix} 23 & 11 \\ 18 & 57 \end{bmatrix} \implies \text{Accuracy} = \mathbf{73.39\%}$ ($80/109$), Specificity $= 67.65\%$.
   - Similarity GAT: $\begin{bmatrix} 25 & 9 \\ 18 & 57 \end{bmatrix} \implies \text{Accuracy} = \mathbf{75.23\%}$ ($82/109$), Specificity $= 73.53\%$.
   - **Finding:** Their confusion matrices are **NOT identical**. Spatial GAT had $2$ additional false positives ($FP=11$) compared to Similarity GAT ($FP=9$).
2. **Sample-by-Sample Divergence:**
   Spatial GAT and Similarity GAT disagree on **4 distinct test samples**:
   - Sample 3 (True 0): Spatial predicts 1 ($p=0.5260$), Similarity predicts 0 ($p=0.4855$).
   - Sample 4 (True 0): Spatial predicts 1 ($p=0.5137$), Similarity predicts 0 ($p=0.4303$).
   - Sample 40 (True 1): Spatial predicts 0 ($p=0.4588$), Similarity predicts 1 ($p=0.5323$).
   - Sample 69 (True 1): Spatial predicts 1 ($p=0.5141$), Similarity predicts 0 ($p=0.4046$).
   - Continuous probabilities differ by up to $14.6\%$.

#### Investigation B: Spatial GATv2 vs. Similarity GATv2
1. **Confusion Matrices:**
   - Spatial GATv2: $\begin{bmatrix} 23 & 11 \\ 17 & 58 \end{bmatrix} \implies \text{Accuracy} = \mathbf{74.31\%}$ ($81/109$).
   - Similarity GATv2: $\begin{bmatrix} 23 & 11 \\ 17 & 58 \end{bmatrix} \implies \text{Accuracy} = \mathbf{74.31\%}$ ($81/109$).
   - **Finding:** Both models arrive at the **exact same summary confusion matrix counts** ($TN=23, FP=11, FN=17, TP=58$).
2. **Are the Underlying Predictions Identical?**
   **NO.** The prediction vectors disagree on **4 distinct test samples**:
   - Sample 3 (True 0): Spatial GATv2 predicts **0** ($p=0.4443$); Similarity GATv2 predicts **1** ($p=0.5471$).
   - Sample 27 (True 0): Spatial GATv2 predicts **1** ($p=0.5093$); Similarity GATv2 predicts **0** ($p=0.3619$).
   - Sample 40 (True 1): Spatial GATv2 predicts **0** ($p=0.4552$); Similarity GATv2 predicts **1** ($p=0.5889$).
   - Sample 61 (True 1): Spatial GATv2 predicts **1** ($p=0.5181$); Similarity GATv2 predicts **0** ($p=0.4385$).
3. **Why the Macroscopic Counts Match Exactly:**
   - On Class 0 (Non-Cancer): Spatial gained 1 TN on Sample 3 but lost 1 TN on Sample 27. Net change $= 0$ (both total $23$ TN, $11$ FP).
   - On Class 1 (Cancer): Spatial gained 1 TP on Sample 61 but lost 1 TP on Sample 40. Net change $= 0$ (both total $58$ TP, $17$ FN).
   - This exact compensatory balance across 4 borderline samples causes the macroscopic summary counts to match, even though the models learned different decision boundaries.
4. **Probability & AUC Verification:**
   Because continuous probabilities differ by up to **$27.59\%$**, their continuous ranking metrics are distinct:
   - Spatial GATv2: **ROC-AUC $= 0.8420$**, **PR-AUC $= 0.9213$**.
   - Similarity GATv2: **ROC-AUC $= 0.8173$**, **PR-AUC $= 0.9090$**.
   - This proves that predictions and metrics were not copied or estimated, but computed from independent model executions.

---

## 8. Metric Verification

Independent recomputation verified all metrics in `reports/phase4_experiment_comparison.csv`:

| Model | Acc | Sensitivity | Specificity | Precision | Cancer $F_1$ | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **ViT Baseline** | $75.23\%$ | $76.00\%$ | $73.53\%$ | $86.36\%$ | $0.8085$ | $0.8141$ | $0.8957$ |
| **GAT Spatial** | $73.39\%$ | $76.00\%$ | $67.65\%$ | $83.82\%$ | $0.7972$ | $0.8224$ | $0.9088$ |
| **GAT Similarity** | $75.23\%$ | $76.00\%$ | $73.53\%$ | $86.36\%$ | $0.8085$ | $0.8078$ | $0.8936$ |
| **GAT Hybrid** | $71.56\%$ | $76.00\%$ | $61.76\%$ | $81.43\%$ | $0.7862$ | $0.8145$ | $0.9047$ |
| **GATv2 Spatial** | $74.31\%$ | $77.33\%$ | $67.65\%$ | $84.06\%$ | $0.8056$ | $0.8420$ | $0.9213$ |
| **GATv2 Similarity**| $74.31\%$ | $77.33\%$ | $67.65\%$ | $84.06\%$ | $0.8056$ | $0.8173$ | $0.9090$ |
| **GATv2 Hybrid** | $77.06\%$ | $81.33\%$ | $67.65\%$ | $84.72\%$ | $0.8299$ | $0.8298$ | $0.9151$ |

- **ROC-AUC and PR-AUC Method:** Verified that `roc_auc_score(all_targets, all_probs)` and `average_precision_score(all_targets, all_probs)` used continuous Softmax scores `probs[:, 1]`, fully meeting scientific standards.

---

## 9. Scientific Interpretation & Research Claim Corrections

A review of `reports/phase4_gat_gatv2_report.md` identified statements that risk overclaiming. The following scientific corrections are required:

| Report Section | Overclaiming Phrasing in Draft | Required Scientific Correction / Nuance |
|---|---|---|
| **Executive Summary** | *"demonstrating the clinical and empirical benefits of universal dynamic attention for relational modeling"* | Change to: *"demonstrating an empirical improvement in cancer sensitivity on this benchmark dataset; clinical utility requires validation on multi-center prospective cohorts."* |
| **Discussion 5.1** | *"Across all three graph construction strategies, GATv2 consistently outperformed standard GAT"* | Add nuance: While GATv2 achieved higher cancer sensitivity ($77.33\%$–$81.33\%$ vs $76.00\%$) and higher ROC-AUC, its overall accuracy on Spatial ($74.31\%$) was slightly lower than Similarity GAT ($75.23\%$). Performance must be characterized as a sensitivity-oriented trade-off. |
| **Discussion 5.2** | *"proves that combining continuous 4-neighbor spatial tissue topology with long-range semantic lesion clustering provides superior relational inductive bias"* | Change to: *"suggests that the hybrid edge topology provides a complementary structural prior on this primary split, detecting 4 additional cancer cases ($TP=61$ vs $TP=57$)."* Note that on $N=109$, an increase of 4 samples is within sampling variability ($\pm 3.7\%$). |
| **Conclusion 7.2** | *"Key Clinical Insight"* | Rephrase to: *"Exploratory Machine Learning Observation"*. Clinical insights cannot be claimed from a single retrospective in-vitro/photographic dataset without clinical trial verification. |
| **Operating Trade-off**| Unmentioned drop in Specificity | Must explicitly note: GATv2 Hybrid increased sensitivity to $81.33\%$, but specificity decreased to $67.65\%$ ($FP=11$) compared to the ViT-only baseline's $73.53\%$ ($FP=9$). In a screening context, higher sensitivity reduces missed malignancies, but increases referral burden. |

---

## 10. Outstanding Issues

1. **Pre-fusion Baseline Status:** In Phase 4, the graph models operated on frozen ViT features with global mean pooling and a single linear classification head. They were intentionally evaluated as independent graph classifiers without the ViT CLS token.
2. **Phase 5 Preparation:** The ViT CLS token ($[B, 192]$) and graph representations ($[B, 192]$) are verified to be fully compatible for gated fusion and cross-attention in Phase 5.
3. **No Retraining Required:** All checkpoints, caches, and test metric files are 100% verified, valid, and permanent.

---

## 11. Final Assessment

### Verdict: **PASS WITH MINOR CORRECTIONS**

**Summary Justification:**
1. **Technical Execution (PASS):** Strict zero-pretrained weight compliance, leak-free feature caches, 21 passing unit tests, mathematically distinct GAT and GATv2 architectures, and verified continuous probability evaluation.
2. **Prediction Verification (PASS):** Compensatory confusion matrix counts between Spatial and Similarity models were thoroughly investigated and proven to stem from distinct predictions across borderline test instances, confirming zero fabrication.
3. **Scientific Claims (MINOR CORRECTIONS):** Research claims in the Phase 4 report must be qualified with appropriate scientific caution regarding sample size limitations, operating trade-offs (sensitivity vs. specificity), and the absence of prospective clinical validation.

---

**STOP POINT REACHED:** Phase 3 and Phase 4 audits are complete. No models have been retrained, no checkpoints altered, and Phase 5 has not commenced.
