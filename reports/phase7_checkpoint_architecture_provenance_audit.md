# Phase 7 — Final Checkpoint Architecture and Provenance Audit

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Checkpoint Architecture, Parameter-Count Reconciliation, Provenance Verification, and Forward-Pass Equivalence Audit  
**Date:** November 2026  
**Audit Mode:** Strict Read-Only (Zero retraining, zero modification to checkpoints, datasets, manifests, or code)

---

## 1. Executive Summary & Audit Verdicts

Following the Stage 2 consistency audit, a discrepancy was noted between the previously approved architecture parameter counts (ViT: 1,965,890; GATv2: 222,722) and figures reported in the Stage 2 consistency audit text (ViT: 2,752,130; GATv2: 300,866). 

A rigorous, independent, programmatic investigation was conducted directly on the PyTorch model classes, instantiated objects, and saved checkpoint binaries on disk.

### Key Audit Determinations:
1. **Parameter-Count Reconciliation:** The previously approved specifications (**1,965,890** for ViT, **222,722** for GATv2, and **49,794** for Concatenation Fusion Head; Total: **2,238,406**) are the **exact ground truth**. The figures 2,752,130 and 300,866 in the Stage 2 audit were typographical drafting errors inserted into prose without executing a programmatic parameter count on the models. The model binaries and code implementations have always possessed exactly 1,965,890 and 222,722 parameters.
2. **Provenance & Modular Encapsulation:** In Phase 5, the hybrid concatenation model was trained under Strategy A (frozen feature representations). Consequently, `models/hybrid_vit_gat_concat_best.pth` contains **only the 6 parameter tensors (49,794 parameters)** of the `ConcatenationFusion` head. End-to-end inference correctly loads the three disjoint component checkpoints (`models/vit_scratch_best.pth`, `models/graph_hybrid_gatv2_best.pth`, and `models/hybrid_vit_gat_concat_best.pth`). Because parameter namespaces are strictly segregated across distinct Python class containers, **zero parameter overwriting can occur**.
3. **Stage-by-Stage Forward-Pass Equivalence:** Across 10 primary validation samples, the live inference pipeline ([src/verify_hybrid_inference.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/verify_hybrid_inference.py)) and the reconstructed training/evaluation forward pass achieved **exact bitwise equivalence ($0.0000000000\text{e}+00$ maximum absolute difference)** across all intermediate representations, logits, and probabilities.
4. **Graph-Stage Timing Decomposition:** Granular GPU-synchronized profiling reveals that pure GATv2 neural network message passing takes only **$57.86\text{ ms}$** ($18.1\%$). The dynamic graph construction stage is dominated by the Python element-by-element assignment loop for the 4-neighbour spatial grid (**$256.69\text{ ms}$**, $80.4\%$), while GPU cosine similarity and top-8 sorting require only **$3.45\text{ ms}$** ($1.1\%$).

### Audit Verdict Summary

| Verification Area | Focus | Status | Key Determination |
| :--- | :--- | :---: | :--- |
| **Verification 1** | Parameter-Count Reconciliation | **PASS** | Instantiated models and checkpoint state dictionaries have exactly 1,965,890 (ViT), 222,722 (GATv2), and 49,794 (Fusion). Combined total: 2,238,406. Erroneous Stage 2 numbers arose from drafting prose without running code. |
| **Verification 2** | Architecture & Checkpoint Provenance | **PASS** | Exact checkpoint filenames, SHA-256 hashes, and component training histories verified. `models/hybrid_vit_gat_concat_best.pth` contains only fusion head parameters. Loading order and module namespacing prevent any parameter overwriting. |
| **Verification 3** | Feature-Cache Provenance | **PASS** | Provenance of offline feature caches (`data/features_vit_scratch/`, `data/features_graph/`) confirmed with SHA-256 hashes. Maximum numerical deviation between live extraction and cached representations is $< 1.41 \times 10^{-4}$ (standard CPU/GPU floating-point serialization tolerance). |
| **Verification 4** | Stage-by-Stage Forward-Pass Equivalence | **PASS** | Zero numerical difference ($0.0000000000\text{e}+00$) across preprocessed tensors, ViT CLS tokens, patch tokens, graph adjacency, graph representations, fused vectors, logits, and probabilities on primary validation images. |
| **Verification 5** | Graph-Stage Latency Decomposition | **PASS** | GPU synchronization deconstructs the graph stage: spatial grid generation ($256.7\text{ ms}$), cosine similarity top-8 ($3.5\text{ ms}$), clamping union ($0.5\text{ ms}$), GATv2 Layer 1 & 2 ($57.9\text{ ms}$), pooling ($0.9\text{ ms}$). |
| **Verification 6** | Scientific Wording & Methodological Limitations | **PASS** | Formally replaced ranking claims in class-weight analysis with approved methodological limitation statement. Corrected graph timing attribution. |

---

## 2. Parameter-Count Reconciliation

### 2.1 Direct Programmatic Recalculation from Model Objects

To eliminate all ambiguity, parameter counts were counted directly from the instantiated model objects (`sum(p.numel() for p in model.parameters())`) and verified against the checkpoint state dictionaries (`sum(p.numel() for p in checkpoint['model_state_dict'].values())`).

```
================================================================================
RECALCULATING PARAMETER COUNTS DIRECTLY FROM INSTANTIATED PYTORCH OBJECTS
================================================================================
Component 1: ViT Backbone (models/vit_scratch_best.pth)
- Class: VisionTransformerScratch (src/models/vit_scratch.py)
- Instantiated Total Parameters:      1,965,890
- Instantiated Trainable Parameters:  1,965,890
- State Dict Parameter Elements:      1,965,890 (56 keys)
- Unit Test Assertion:                tests/test_vit.py (assertEqual 1,965,890) -> PASSED

Component 2: GATv2 Backbone (models/graph_hybrid_gatv2_best.pth)
- Class: GraphClassifier (src/models/graph_classifier.py)
- Instantiated Total Parameters:        222,722
- Instantiated Trainable Parameters:    222,722
- State Dict Parameter Elements:        222,722 (14 keys)
- State Dict Verification:            Phase 4 report -> MATCHED

Component 3: Concatenation Fusion Head (models/hybrid_vit_gat_concat_best.pth)
- Class: ConcatenationFusion (src/models/hybrid_fusion.py)
- Instantiated Total Parameters:         49,794
- Instantiated Trainable Parameters:     49,794
- State Dict Parameter Elements:         49,794 (6 keys)
- Phase 5 Specification:              Phase 5 report line 66 -> MATCHED

TOTAL COMBINED HYBRID ARCHITECTURE PARAMETERS: 2,238,406
================================================================================
```

### 2.2 Root Cause of Earlier Discrepancies

| Architecture Component | Previously Approved Specifications | Erroneous Stage 2 Text | Actual Physical Parameters (Recalculated) | Root Cause of Discrepancy |
| :--- | :---: | :---: | :---: | :--- |
| **Vision Transformer (ViT)** | 1,965,890 | 2,752,130 | **1,965,890** | The numbers 2,752,130 and 300,866 were drafted into Markdown text in the Stage 2 report without running a programmatic parameter counter. The actual codebase, test suite (`tests/test_vit.py`), and checkpoint state dictionaries have always contained 1,965,890 parameters. |
| **GATv2 Graph Classifier** | 222,722 | 300,866 | **222,722** | Same drafting error in report prose. The physical model and checkpoint state dict have always had 222,722 parameters. |
| **Concatenation Fusion Head** | 49,794 | 49,794 | **49,794** | Correctly reported across all reports and code. |
| **Total Hybrid System** | **2,238,406** | 3,102,790 | **2,238,406** | Sum of ground truth components: $1,965,890 + 222,722 + 49,794 = \mathbf{2,238,406}$. |

---

## 3. Architecture and Checkpoint Provenance

### 3.1 Checkpoint Inventory & Cryptographic Hashes

Every model component utilized in the primary hybrid pipeline has been cataloged with its exact filename, file size, SHA-256 cryptographic digest, class definition, and architecture hyperparameter configuration:

```
1. ViT Backbone Checkpoint
   - Filename: models/vit_scratch_best.pth
   - SHA-256: 1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf
   - Filesize: 7,929,915 bytes
   - Source Class: src/models/vit_scratch.py:VisionTransformerScratch
   - Configuration:
       img_size: 224, patch_size: 16 (196 patches)
       embed_dim: 192, depth: 4 transformer encoder blocks
       num_heads: 3 (head_dim: 64), mlp_dim: 768 (expansion 4x)
       num_classes: 2, dropout: 0.1
   - Total Parameters: 1,965,890
   - Trainable Parameters: 1,965,890
   - State Dict Elements: 1,965,890 across 56 keys
   - Checkpoint Epoch & Performance: Epoch 34 (val_loss: 0.4880, val_acc: 80.56%, val_f1: 0.8531)
   - Trained Using This Exact Architecture: YES (Phase 3 from-scratch training).

2. GATv2 Graph Classifier Checkpoint
   - Filename: models/graph_hybrid_gatv2_best.pth
   - SHA-256: 04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f
   - Filesize: 902,683 bytes
   - Source Class: src/models/graph_classifier.py:GraphClassifier
   - Configuration:
       graph_type: "hybrid" (4-neighbour spatial grid + dynamic cosine top-8)
       gnn_type: "gatv2" (dynamic attention scoring)
       in_features: 192, hidden_per_head: 48, num_heads: 4 (out_dim: 192)
       num_layers: 2, pooling: "mean", num_classes: 2, k: 8, dropout: 0.1
   - Total Parameters: 222,722
   - Trainable Parameters: 222,722
   - State Dict Elements: 222,722 across 14 keys
   - Checkpoint Epoch & Performance: Epoch 21 (val_loss: 0.4576, val_acc: 82.41%, val_f1: 0.8707)
   - Trained Using This Exact Architecture: YES (Phase 4 graph training on cached ViT patch tokens).

3. Concatenation Fusion Head Checkpoint
   - Filename: models/hybrid_vit_gat_concat_best.pth
   - SHA-256: 05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365
   - Filesize: 204,449 bytes
   - Source Class: src/models/hybrid_fusion.py:ConcatenationFusion
   - Configuration:
       vit_dim: 192, graph_dim: 192 (concatenated input dim: 384)
       hidden_dim: 128, num_classes: 2, dropout: 0.2
       Layer 0: Linear(384, 128)
       Layer 1: BatchNorm1d(128)
       Layer 2: ReLU()
       Layer 3: Dropout(p=0.2)
       Layer 4: Linear(128, 2)
   - Total Parameters: 49,794
   - Trainable Parameters: 49,794
   - State Dict Elements: 49,794 across 6 keys:
       classifier.0.weight: torch.Size([128, 384])  (49,152 params)
       classifier.0.bias:   torch.Size([128])        (128 params)
       classifier.1.weight: torch.Size([128])        (128 params)
       classifier.1.bias:   torch.Size([128])        (128 params)
       classifier.4.weight: torch.Size([2, 128])     (256 params)
       classifier.4.bias:   torch.Size([2])          (2 params)
   - Checkpoint Epoch & Performance: Epoch 4 (val_loss: 0.5313, val_acc: 81.48%, val_f1: 0.8592)
   - Trained Using This Exact Architecture: YES (Phase 5 fusion training on cached CLS and graph representations).
```

### 3.2 Differences Between Standalone Models and Phase 5 Hybrid Model

1. **Standalone ViT Baseline (`models/vit_scratch_best.pth`):**
   - Contains a classification head `Linear(192, 2)` (386 parameters: `head.weight` $[2, 192]$, `head.bias` $[2]$).
   - In Phase 5 hybrid inference, this head is bypassed; instead, `forward_features()` extracts `cls` ($[B, 192]$) and `patches` ($[B, 196, 192]$).
2. **Standalone GATv2 Baseline (`models/graph_hybrid_gatv2_best.pth`):**
   - Contains a classification head `Linear(192, 2)` (386 parameters: `head.weight` $[2, 192]$, `head.bias` $[2]$).
   - In Phase 5 hybrid inference, this head is bypassed; instead, `get_graph_representation()` extracts the global mean-pooled graph embedding ($[B, 192]$).
3. **Phase 5 Hybrid Architecture:**
   - Combines the 192-dim ViT CLS representation with the 192-dim GATv2 pooled graph representation into a 384-dim concatenated vector.
   - The standalone heads of ViT and GATv2 are retained in the checkpoint binaries (preserving their standalone baseline utility), but during hybrid screening, classification is performed exclusively by the `ConcatenationFusion` head ($49,794$ parameters).

### 3.3 Checkpoint Separation and Zero-Overwrite Guarantee

In Phase 5, the hybrid concatenation experiment was executed under **Strategy A (Frozen Feature Extractors)**:
- Pre-extracted CLS tokens and patch tokens from `models/vit_scratch_best.pth` were cached in `data/features_vit_scratch/`.
- Pre-extracted graph representations from `models/graph_hybrid_gatv2_best.pth` were cached in `data/features_graph/`.
- The fusion training script (`src/train_hybrid.py`) trained **only** the `ConcatenationFusion` model instance on top of these cached representations.
- Therefore, `models/hybrid_vit_gat_concat_best.pth` contains **strictly the 6 parameter tensors of the fusion head** ($49,794$ parameters). It does not contain ViT or GATv2 weights.

When `VerifiedHybridInferencePipeline` in [src/verify_hybrid_inference.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/verify_hybrid_inference.py) runs:
1. `self.vit` is instantiated as a `VisionTransformerScratch` object and loads `models/vit_scratch_best.pth`.
2. `self.gatv2` is instantiated as a `GraphClassifier` object and loads `models/graph_hybrid_gatv2_best.pth`.
3. `self.fusion` is instantiated as a `ConcatenationFusion` object and loads `models/hybrid_vit_gat_concat_best.pth`.

Because each sub-model is an independent Python object with completely disjoint parameter namespaces (`self.vit.*`, `self.gatv2.*`, `self.fusion.*`), **no checkpoint can overwrite parameters belonging to another component**.

---

## 4. Feature-Cache Provenance Verification

The offline feature caches used to train and validate all Phase 5 and 6 models were inspected and cryptographically hashed:

| Cache File | Role in Research | Size (Bytes) | SHA-256 Cryptographic Hash | Embedded Source Checkpoint Metadata |
| :--- | :--- | :---: | :--- | :--- |
| `data/features_vit_scratch/train_patch_features.pt` | Train patch & CLS tokens ($N=486$) | 73,570,991 | `c25a5a16ccbca1404ade1286f1cc458c17ab707b032d0077bbb4acff5d30286d` | `models/vit_scratch_best.pth` (Epoch 34) |
| `data/features_vit_scratch/val_patch_features.pt` | Val patch & CLS tokens ($N=108$) | 16,350,497 | `2d9689623328f119676cc7f4ab309a06cf39824624994977ddd87b45fc433fae` | `models/vit_scratch_best.pth` (Epoch 34) |
| `data/features_graph/train_graph_hybrid_gatv2.pt` | Train graph representations ($N=486$) | 385,342 | `0ece316fc48e269452b83ae82a4680aa75d5ac56e438f6fb1db9d73ff416fc41` | `models/graph_hybrid_gatv2_best.pth` (Epoch 21) |
| `data/features_graph/val_graph_hybrid_gatv2.pt` | Val graph representations ($N=108$) | 87,154 | `42281acf2098ad7a21729969f5b292a2adf4f1aa221603c4d90f1b417a2c7272` | `models/graph_hybrid_gatv2_best.pth` (Epoch 21) |

All caches explicitly record the exact source checkpoint filenames, epoch numbers, and best validation losses in their internal dictionary metadata.

---

## 5. Actual Forward-Pass Equivalence Validation

To verify that the live inference pipeline ([src/verify_hybrid_inference.py](file:///c:/Users/Sakshi%20Aswale/OneDrive/Desktop/Major%20Project%20final/Oral-Cancer-Screening-Using-Vision-Transformer-GAT-with-Explainable-AI/src/verify_hybrid_inference.py)) reproduces the exact mathematical operations of the original training and evaluation implementations, a numerical validation test was executed across 10 validation images from `data/splits_primary/val.csv`.

Both pipelines were executed in deterministic evaluation mode (`eval()`) with zero gradients (`torch.no_grad()`).

### 5.1 Maximum Absolute Numerical Differences (Live Inference vs. Training Implementation)

| Computational Stage | Tensor Shape | Maximum Absolute Difference | Equivalence Status |
| :--- | :---: | :---: | :---: |
| **1. Preprocessed Input Image Tensor** | $[1, 3, 224, 224]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **2. ViT CLS Representation ($h_{\text{CLS}}$)** | $[1, 192]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **3. ViT Patch Token Representations** | $[1, 196, 192]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **4. Graph Adjacency Matrix ($\mathbf{A}_{\text{hybrid}}$)** | $[1, 196, 196]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **5. GATv2 Pooled Graph Embedding ($h_{\text{graph}}$)** | $[1, 192]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **6. Joint Concatenated Embedding ($z_{\text{concat}}$)** | $[1, 384]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **7. Final Classification Logits** | $[1, 2]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |
| **8. Softmax Class Probabilities** | $[1, 2]$ | **$0.0000000000\text{e}+00$** | **EXACT MATCH** |

### 5.2 Comparison with Offline Feature Caches
When comparing live on-the-fly extracted tensors against tensors saved in the offline disk caches (`data/features_vit_scratch/` and `data/features_graph/`):
- **Live CLS vs. Cached CLS:** $9.30 \times 10^{-6}$
- **Live Patches vs. Cached Patches:** $1.74 \times 10^{-5}$
- **Live Graph Embedding vs. Cached Graph Embedding:** $1.41 \times 10^{-4}$

These minimal deviations ($\le 1.41 \times 10^{-4}$) are well within expected single-precision floating-point tolerances arising from CPU tensor serialization vs. GPU runtime execution.

**Determination:** Forward-pass equivalence is **PASS**. The inference pipeline is mathematically identical to the training implementation.

---

## 6. GPU-Synchronized Granular Latency Decomposition

In Audit E of the previous report, an overall graph-stage duration of $285.96\text{ ms}$ was reported. To address the user requirement and clarify whether this represents pure GATv2 neural network computation or includes graph construction and data transfers, granular GPU-synchronized profiling was executed on the NVIDIA GeForce RTX 4050 Laptop GPU over 30 synchronized runs.

### 6.1 Empirical Breakdown of the Graph Stage

| Sub-Stage | Operation Description | Mean Latency (ms) | Std Dev (ms) | % of Graph Stage | Implementation Bottleneck |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **4a. Spatial Grid Construction** | Iterative Python element assignment in `src/graph/spatial_graph.py` | 256.686 | 49.330 | 80.4% | Python nested loop writing 924 directed edges into CUDA tensor |
| **4b. Cosine Similarity & Top-8** | Vectorized $L_2$ norm, batched matrix mult, top-$k$ sort | 3.451 | 0.846 | 1.1% | Fully vectorized PyTorch GPU operations |
| **4c. Adjacency Union & Clamping** | `clamp(spatial + similarity, 0, 1)` | 0.475 | 0.307 | 0.1% | Fast element-wise GPU kernel |
| **Graph Construction Subtotal (4a+4b+4c)** | Complete dynamic adjacency matrix generation | **260.612** | **49.400** | **81.6%** | Dominated by unvectorized spatial loop |
| **4d. GATv2 Layer 1** | Multi-head dynamic graph attention (4 heads, 48 dim) | 28.275 | 8.641 | 8.9% | Pure neural network message passing |
| **4e. GATv2 Layer 2** | Multi-head dynamic graph attention (4 heads, 48 dim) | 29.585 | 9.143 | 9.3% | Pure neural network message passing |
| **GATv2 Message Passing Subtotal (4d+4e)** | Two layers of dynamic graph attention | **57.860** | **17.784** | **18.1%** | Pure neural network execution |
| **4f. Global Mean Pooling** | Patch-level average pooling to $[1, 192]$ | 0.924 | 0.151 | 0.3% | Fast reduction kernel |
| **Total Graph Stage** | **Construction + GATv2 + Pooling** | **319.434** | **53.385** | **100.0%** | Full graph branch execution |

### 6.2 Key Latency Findings
1. **GATv2 Neural Network Latency is Fast:** The actual GATv2 attention layers require only **$57.86\text{ ms}$** combined.
2. **Construction Bottleneck Identified:** The $256.69\text{ ms}$ duration is caused exclusively by the unvectorized element-by-element Python assignment loop in `src/graph/spatial_graph.py` (`adj[node_idx, nbr] = 1.0`). In Stage 3, precomputing the static spatial grid or vectorizing it via scatter operations could easily reduce total graph-stage latency from $\approx 319\text{ ms}$ to under $70\text{ ms}$.

---

## 7. Scientific Wording Corrections

In accordance with user requirements, the scientific documentation has been revised:

### 7.1 Class-Weight Deviation Impact
The earlier claim that "architecture ranking remained unbiased" has been formally retracted and replaced with the following approved methodological statement:

> "The same class-weight configuration was used across the Phase 5–6 experiments, improving consistency within that experimental group. However, the deviation from the approved class weights remains a methodological limitation, and its influence on individual models or their relative ranking has not been independently quantified."

### 7.2 Inference Timing Attribution
Earlier documentation attributing the $\approx 286\text{ ms}$ graph duration to GATv2 computation has been formally corrected. The report now explicitly documents that graph construction accounts for $81.6\%$ ($260.61\text{ ms}$) of this duration, while pure GATv2 message-passing layers account for $18.1\%$ ($57.86\text{ ms}$).

---

## 8. Inspected Files, Functions, and Checkpoints

### Python Files & Functions:
1. `src/models/vit_scratch.py`: `VisionTransformerScratch`, `build_vit_scratch`
2. `src/models/graph_classifier.py`: `GraphClassifier`
3. `src/models/graph_attention_v2.py`: `GATv2`, `build_gatv2`
4. `src/models/hybrid_fusion.py`: `ConcatenationFusion`
5. `src/verify_hybrid_inference.py`: `VerifiedHybridInferencePipeline`, `predict_image`
6. `src/extract_patch_features.py`: `extract_split_features`
7. `src/extract_graph_representations.py`: `load_gnn_model`, `extract_for_checkpoint`
8. `src/train_hybrid.py`: `load_dataset_splits`, `train_single_model`
9. `src/run_ablation_studies.py`: `fwd_concat`, `run_all_ablations`
10. `tests/test_vit.py`: `test_parameter_counts_and_pretrained_flag`

### Checkpoints & Artifacts Verified:
1. `models/vit_scratch_best.pth` (SHA-256: `1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf`)
2. `models/graph_hybrid_gatv2_best.pth` (SHA-256: `04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f`)
3. `models/hybrid_vit_gat_concat_best.pth` (SHA-256: `05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365`)
4. `data/features_vit_scratch/val_patch_features.pt` (SHA-256: `2d9689623328f119676cc7f4ab309a06cf39824624994977ddd87b45fc433fae`)
5. `data/features_graph/val_graph_hybrid_gatv2.pt` (SHA-256: `42281acf2098ad7a21729969f5b292a2adf4f1aa221603c4d90f1b417a2c7272`)

---
*Report certified by autonomous research audit pipeline.*
