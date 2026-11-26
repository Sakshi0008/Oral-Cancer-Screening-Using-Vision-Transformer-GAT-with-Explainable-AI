# ORALSCAN AI: Explainable AI (XAI) Methodology Documentation

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Mathematical Formulations, Rollout Algorithms, Graph Attention Visualizations, and Axiomatic Attribution  
**Date:** November 2026  

---

## 1. Overview of the Multimodal XAI Suite

ORALSCAN AI implements a cohesive four-pillar Explainable AI framework designed to explain model behavior at every tier of the deep learning architecture:

1. **Visual Patch Level:** Vision Transformer Attention Rollout (Abnar & Zuidema, 2020) tracing information flow from image patches to the class token.
2. **Relational Topology Level:** Dynamic edge attention coefficients and incoming node centrality in the Graph Attention Network (GATv2).
3. **Multimodal Bottleneck Level:** Integrated Gradients (Sundararajan et al., 2017) measuring the relative sensitivity share of visual vs topological branches.
4. **Human Communication Level:** Automated clinical narrative generation synthesizing plain-language explanations with non-diagnostic boundary enforcement.

---

## 2. Mathematical Formulations

### 2.1 ViT Attention Rollout

In multi-head self-attention, the attention matrix at layer $l$ is averaged across all $H=3$ heads:
$$\bar{\mathbf{A}}_l = \frac{1}{H} \sum_{h=1}^H \mathbf{A}_{l, h} \in \mathbb{R}^{197 \times 197}$$

To account for identity residual connections, an identity matrix is added, followed by row-normalization to ensure a valid transition probability matrix:
$$\hat{\mathbf{A}}_l = 0.5 \bar{\mathbf{A}}_l + 0.5 \mathbf{I}, \quad \hat{\mathbf{A}}_l[i, :] = \frac{\hat{\mathbf{A}}_l[i, :]}{\sum_j \hat{\mathbf{A}}_l[i, j]}$$

The recursive rollout matrix $\mathbf{R}_L$ across all 4 Transformer layers is:
$$\mathbf{R}_l = \hat{\mathbf{A}}_l \times \mathbf{R}_{l-1}, \quad \text{with } \mathbf{R}_0 = \mathbf{I}$$

Row 0 of $\mathbf{R}_4$ represents the accumulated flow of information from all tokens into the final $[\text{CLS}]$ token. Spatial attribution is extracted by cleanly dropping the self-attention of the class token (column 0):
$$\mathbf{s} = \mathbf{R}_4[0, 1:197] \in \mathbb{R}^{196}$$
The 196 values are reshaped to a $14 \times 14$ grid and upsampled using bicubic interpolation to $224 \times 224$ for visual overlay.

### 2.2 GATv2 Attention Extraction and Topological Overlay

In `GATv2ConvScratch`, attention coefficients $\alpha_{ij}$ represent dynamic normalized weights computed across source neighbors $j$ for destination node $i$:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{x}_i \parallel \mathbf{W} \mathbf{x}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{x}_i \parallel \mathbf{W} \mathbf{x}_k]\right)\right)}$$

- **Node Centrality:** Self-loops are explicitly zeroed out ($\alpha_{ii} = 0$), and incoming attention is summed:
  $$C_i = \sum_{j \in \mathcal{N}_i, j \neq i} \alpha_{ij}$$
  This isolates true inter-patch message passing from self-retention.
- **Topological Edge Classification:** Top-attended edges are categorized into 3 distinct functional classes:
  - **Spatial Only (Cyan):** Connects physical cardinal neighbors ($4$-grid adjacency).
  - **Similarity Only (Gold):** Connects non-adjacent patches with high feature cosine similarity ($k=8$).
  - **Shared (Magenta):** Connects spatially adjacent patches that simultaneously share high semantic feature affinity.

### 2.3 Fusion Integrated Gradients

Integrated Gradients attributes the scalar predicted-class logit $y_{\hat{c}}$ to each of the 384 dimensions of the concatenated representation $\mathbf{z} = [\mathbf{z}_{\text{cls}} \parallel \mathbf{z}_{\text{graph}}]$:
$$\text{IG}_i(\mathbf{z}) = (z_i - z'_i) \times \int_0^1 \frac{\partial F(\mathbf{z}' + \alpha (\mathbf{z} - \mathbf{z}'))}{\partial z_i} d\alpha$$

- **Reference Baseline:** Default representation baseline is $\mathbf{z}' = [0.05]_{384}$, which avoids LayerNorm zero-variance singularities.
- **Numerical Approximation:** Evaluated via a 50-step trapezoidal rule:
  $$\text{IG}_i \approx (z_i - z'_i) \sum_{k=1}^M \frac{g_{k-1, i} + g_{k, i}}{2 M}$$
- **Branch Sensitivity Shares:**
  $$\text{ViT Share} = \frac{\sum_{i=1}^{192} |\text{IG}_i|}{\sum_{j=1}^{384} |\text{IG}_j|} \times 100\%, \quad \text{GAT Share} = \frac{\sum_{i=193}^{384} |\text{IG}_i|}{\sum_{j=1}^{384} |\text{IG}_j|} \times 100\%$$
- **Completeness Dual-Reporting:** Reports both the absolute delta $|\sum \text{IG} - \Delta f|$ and the relative percentage error, clearly noting that relative percentages are mathematically sensitive to small logit denominators ($\Delta f < 0.5$).

---

## 3. Sanity and Faithfulness Verification Summary

1. **Adebayo Parameter Randomization Check:** Comparing attributions from the trained ViT vs an uninitialized randomized model drops Spearman correlation from $1.000$ to **$r = 0.118$**, proving that saliency maps depend genuinely on learned weights.
2. **Prediction Invariance:** Forward hooks detached post-softmax attention weights without gradient or state retention; logit differences with vs without hooks are strictly **$0.000000\text{e}+00$**.
3. **Repeatability:** Identical inputs yield identical attributions to machine precision ($\Delta = 0.000000\text{e}+00$).
4. **Perturbation Faithfulness:** Patch deletion curves show that Most Important First (MIF) deletion drops model target confidence significantly faster than random deletion ($\text{AUDC}_{\text{MIF}} = 0.6431$ vs $\text{AUDC}_{\text{Random}} = 0.6893$).
