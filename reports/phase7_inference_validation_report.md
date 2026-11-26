# Phase 7 Stage 2: Inference Pipeline Validation & Verification Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Phase:** Phase 7 (Final Research Application, Explainable AI, User-Friendly Reporting, and Frontend)  
**Stage:** Stage 2 (Inference Validation & Verification)  
**Date:** November 24, 2026  
**Auditor / Assistant:** Antigravity AI Assistant  
**Verification Engine:** `src/verify_hybrid_inference.py`  
**Target Hardware:** NVIDIA GPU (`cuda`) / PyTorch 2.6.0  

---

## 1. Executive Summary

This report documents the rigorous technical and numerical verification of the end-to-end inference pipeline for the selected primary hybrid model:
$$\text{Vision Transformer (Scratch)} \oplus \text{GATv2 (Hybrid Adjacency)} \longrightarrow \text{Concatenation Fusion Head}$$

All 13 validation criteria specified in the Phase 7 mandate were tested using the dedicated, non-invasive verification script `src/verify_hybrid_inference.py`.

**Verification Result:** **ALL 13 VERIFICATION CHECKPOINTS PASSED.**  
The inference pipeline is completely deterministic, exhibits zero weight drift, loads checkpoints without warnings, and achieves $100\%$ prediction agreement on representative validation samples.

---

## 2. Detailed Audit of the 13 Verification Checkpoints

### Checkpoint 1: Actual Trained Checkpoint Loading
- **Check:** Loaded `models/vit_scratch_best.pth`, `models/graph_hybrid_gatv2_best.pth`, and `models/hybrid_vit_gat_concat_best.pth`.
- **Observation:** All three checkpoint files loaded cleanly via `torch.load(..., map_location=device, weights_only=False)` without missing keys or shape mismatches.
- **Status:** **PASS**

### Checkpoint 2: Model Component Integrity & Parameter Counts
- **Check:** Verified parameter counts across the decoupled hybrid pipeline:
  - ViT Scratch Backbone: **1,965,890** parameters
  - GATv2 Hybrid Backbone: **222,722** parameters
  - Concatenation Fusion Head: **49,794** parameters
  - Total Architecture Parameters: **2,238,406** parameters
- **Status:** **PASS**

### Checkpoint 3: Input Image Dimensions & Normalization
- **Check:** Evaluated input tensor dimensionality and range.
- **Observation:** Input tensor has exact shape $[1, 3, 224, 224]$, `dtype=torch.float32`. Dataset channel normalization applied:
  - Channel 0 (R): $\mu = 0.485, \sigma = 0.229$
  - Channel 1 (G): $\mu = 0.456, \sigma = 0.224$
  - Channel 2 (B): $\mu = 0.406, \sigma = 0.225$
- **Status:** **PASS**

### Checkpoint 4: RGB Conversion & Preprocessing Pipeline
- **Check:** Tested Pillow image ingestion.
- **Observation:** All images explicitly converted via `.convert("RGB")` before resizing, preventing channel mismatches from grayscale or RGBA files. Deterministic Bicubic interpolation verified.
- **Status:** **PASS**

### Checkpoint 5: Graph Construction (Spatial + Similarity)
- **Check:** Verified adjacency matrix construction from ViT patch tokens.
- **Observation:** Spatial 8-NN adjacency computed via Chebyshev distance on $14 \times 14$ grid ($196$ nodes). Cosine similarity 8-NN computed dynamically on patch feature vectors. Union adjacency matrix with added self-loops verified ($196 \times 196$, binary symmetric).
- **Status:** **PASS**

### Checkpoint 6: ViT Patch Representation Extraction
- **Check:** Monitored ViT forward pass outputs.
- **Observation:**
  - `cls_token` shape: $[1, 192]$ (Global context representation)
  - `patch_tokens` shape: $[1, 196, 192]$ (Localized patch representations)
- **Status:** **PASS**

### Checkpoint 7: GATv2 Graph Representation
- **Check:** Monitored GATv2 node attention and global pooling.
- **Observation:** GATv2 applies 2 layers of dynamic multi-head attention ($4$ heads, $d_h = 48$). Global mean pooling across updated nodes yields graph-level embedding $[1, 192]$.
- **Status:** **PASS**

### Checkpoint 8: Fusion Layer Representation
- **Check:** Monitored representation concatenation.
- **Observation:** Vector concatenation of `cls_token` and `graph_rep` yields $[1, 384]$. Forward pass through `Linear(384, 128) -> LayerNorm -> ReLU -> Dropout` verified.
- **Status:** **PASS**

### Checkpoint 9: Final Logits & Class Mapping
- **Check:** Monitored final output layer.
- **Observation:** Final `Linear(128, 2)` produces logits $[1, 2]$:
  - Logit index `0`: Negative class (`NON CANCER`)
  - Logit index `1`: Positive class (`CANCER`)
- **Status:** **PASS**

### Checkpoint 10: Continuous Score & Softmax Probability
- **Check:** Monitored risk score generation.
- **Observation:** Softmax activation:
  $$p_{\text{cancer}} = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$
  Guaranteed in range $[0.0, 1.0]$. Binary threshold frozen at $0.50$.
- **Status:** **PASS**

### Checkpoint 11: Inference on Representative Validation Split Images
- **Check:** Executed inference across 6 representative images from `data/splits_primary/val.csv` (3 Non-Cancer, 3 Cancer).
- **Results:**
  | Image File Path | True Label | Predicted Label | Cancer Probability | Logits $[z_0, z_1]$ | Screening Status |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | `Oral Cancer/.../NON CANCER/007.jpeg` | NON CANCER | NON CANCER | **12.67%** | $[+1.087, -0.843]$ | **MATCH** |
  | `Oral Cancer/.../NON CANCER/009.jpeg` | NON CANCER | NON CANCER | **1.64%** | $[+2.046, -2.045]$ | **MATCH** |
  | `Oral Cancer/.../NON CANCER/015.jpeg` | NON CANCER | NON CANCER | **8.14%** | $[+1.320, -1.103]$ | **MATCH** |
  | `Oral Cancer/.../CANCER/005.jpeg` | CANCER | CANCER | **65.64%** | $[-0.237, +0.410]$ | **MATCH** |
  | `Oral Cancer/.../CANCER/014.jpeg` | CANCER | CANCER | **86.51%** | $[-0.929, +0.929]$ | **MATCH** |
  | `Oral Cancer/.../CANCER/024.jpeg` | CANCER | CANCER | **78.17%** | $[-0.638, +0.638]$ | **MATCH** |
- **Observation:** All 6 validation cases ($100\%$) correctly classified with expected confidence margins.
- **Status:** **PASS**

### Checkpoint 12: Deterministic Repeatability
- **Check:** Executed 5 repeated inference passes on `NON CANCER/007.jpeg` under identical conditions.
- **Observation:** Output logits were identical across all 5 runs:
  $$\max_{i} |z_i^{(k)} - z_i^{(1)}| < 1.0 \times 10^{-7} \quad \forall k \in \{2, 3, 4, 5\}$$
- **Status:** **PASS**

### Checkpoint 13: Zero Weight Drift Verification
- **Check:** Computed SHA-256 digests over all parameters before and after the inference runs:
  - `ViT` Parameter Hash: `3e8f...` (Identical before and after)
  - `GATv2` Parameter Hash: `9b21...` (Identical before and after)
  - `Fusion` Parameter Hash: `7c44...` (Identical before and after)
- **Observation:** No gradients accumulated (`torch.no_grad()`); model weights remained 100% invariant.
- **Status:** **PASS**

---

## 3. Unresolved Issues & Recommendations for Stage 3

1. **Inference Latency:**
   - Full end-to-end forward pass on GPU takes approximately $120\text{ ms} - 180\text{ ms}$ per image. On CPU fallback, it takes $\sim 850\text{ ms}$. This latency is well within acceptable limits for a responsive web application.
2. **Intermediate Attention Caching for Stage 3:**
   - For Explainable AI (Stage 3), the ViT forward pass must be configured to retain all layer attention matrices ($4 \times [1, 3, 197, 197]$) and the GATv2 forward pass must retain edge attention weights ($2 \times [1, 4, 196, 196]$).
   - This will be packaged into a clean, dedicated XAI module (`src/xai/`) without modifying existing training scripts.

---

## 4. Verification Conclusion

The primary hybrid model (`hybrid_vit_gat_concat_best.pth`) is **fully certified, validated, and ready for Stage 3 (Explainable AI implementation).**
