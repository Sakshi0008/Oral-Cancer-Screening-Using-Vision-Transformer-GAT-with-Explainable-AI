# Phase 4 Scientific Report: Image Graph Construction Strategies

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Author:** Antigravity AI Research System
**Date:** November 13, 2026
**Status:** Completed and Validated

---

## 1. Motivation: Patches as Graph Nodes

In standard Vision Transformer architectures, self-attention calculates a dense, unconstrained all-to-all similarity matrix between all $N=196$ patch tokens. While this provides global context, it suffers from two major limitations in medical image analysis:
1. **Dispersion Over Background Tissue:** It permits spurious interactions between distant, unrelated anatomical structures (e.g., normal dorsal tongue vs. dental reflections).
2. **Loss of Geometric Constraints:** It ignores the continuous biological tissue boundaries where oral malignant transformations naturally propagate.

By conceptualizing each of the $196$ ViT patch tokens as a **node in a graph** ($V = \{v_1, \dots, v_{196}\}$ with node feature vectors $h_i \in \mathbb{R}^{192}$), we transform the classification task from unstructured sequence modeling into structured geometric deep learning. Graph construction allows us to explicitly enforce spatial contiguity, feature-based semantic affinities, or their combination, providing an inductive bias that guides subsequent Graph Attention Networks.

---

## 2. Graph Construction Strategies

Three independent, reproducible graph construction strategies were implemented in pure PyTorch and validated across all dataset splits:

```
+---------------------------------------------------------------------------------------+
|                               GRAPH CONSTRUCTION STRATEGIES                           |
+------------------------------------+--------------------------------------------------+
| Strategy 1: Spatial Grid Graph     | 4-neighbour spatial mesh (local tissue continuity)|
| Strategy 2: Feature-Similarity     | Cosine similarity top-k (semantic lesion clusters)|
| Strategy 3: Hybrid Graph           | Union of spatial + similarity (multi-relational)  |
+------------------------------------+--------------------------------------------------+
```

---

### 2.1 Strategy 1: Spatial Adjacency Graph

The spatial adjacency graph models physical anatomical proximity across the $14 \times 14$ patch grid.

#### Topology & Connectivity:
- **Grid Mapping:** Node index $i = r \times 14 + c$ where row $r \in [0, 13]$ and column $c \in [0, 13]$.
- **4-Neighbour Rule:** Each node $i=(r,c)$ connects to its orthogonal neighbours:
  $$\mathcal{N}_{\text{spatial}}(r, c) = \{(r-1, c), \, (r+1, c), \, (r, c-1), \, (r, c+1)\} \cap [0, 13]^2$$
- **Bidirectionality:** All edges are symmetric: if $i \to j$, then $j \to i$.
- **Explicit Self-Loops:** $(i, i) \in \mathcal{E}$ for all $i \in [0, 195]$ to allow nodes to retain their self-representations during attention aggregation.

#### Boundary Node Degree Analysis:
Because of the finite $14 \times 14$ grid boundary, the degree distribution is strictly deterministic:
- **Corners ($4$ nodes):** $(0,0), (0,13), (13,0), (13,13) \implies 2\text{ neighbours} + 1\text{ self-loop} = \mathbf{3}$ edges.
- **Borders ($48$ nodes):** Remaining edge patches $\implies 3\text{ neighbours} + 1\text{ self-loop} = \mathbf{4}$ edges.
- **Interior ($144$ nodes):** $12 \times 12$ central patches $\implies 4\text{ neighbours} + 1\text{ self-loop} = \mathbf{5}$ edges.
- **Total Directed Edges per Image:**
  $$4 \times 3 + 48 \times 4 + 144 \times 5 = 12 + 192 + 720 = \mathbf{924} \text{ directed edges}$$

#### Clinical Rationale:
Tumor progression and cellular atypia in oral squamous cell carcinoma (OSCC) follow contiguous tissue expansion. The spatial graph ensures that attention messages flow along contiguous epithelial layers.

---

### 2.2 Strategy 2: Feature-Similarity Graph

The feature-similarity graph captures non-local morphological similarities between patches regardless of spatial distance.

#### Construction Pipeline:
1. **$L_2$ Normalization:** Each 192-dimensional patch feature vector is normalized to unit length:
   $$\tilde{h}_i = \frac{h_i}{\|h_i\|_2}$$
2. **Pairwise Cosine Similarity:**
   $$S_{ij} = \langle \tilde{h}_i, \, \tilde{h}_j \rangle = \tilde{h}_i \cdot \tilde{h}_j^T \in [-1.0, 1.0]$$
3. **Self-Similarity Exclusion:** To prevent self-loops from dominating nearest-neighbour selection, the diagonal elements are masked before top-$k$ selection:
   $$S'_{ii} = -\infty$$
4. **Top-$k$ Neighbour Selection ($k=8$):**
   $$\mathcal{N}_{\text{top-k}}(i) = \text{argtopk}_{j \ne i} (S'_{ij}, \, k=8)$$
5. **Bidirectional Symmetrization:** If $j \in \mathcal{N}_{\text{top-k}}(i)$, then both directed edges $(i, j)$ and $(j, i)$ are added to the adjacency matrix:
   $$A_{\text{sim}} = \text{clamp}(A_{\text{directed}} + A_{\text{directed}}^T, \, 0.0, \, 1.0)$$
6. **Explicit Self-Loops:** $A_{\text{sim}}[i, i] = 1.0$ for all $i \in [0, 195]$.

#### Clinical Rationale:
Malignant oral lesions often exhibit multifocal keratinization, hyperchromatic nuclei, or cellular pleomorphism scattered across different non-contiguous portions of the oral cavity. Feature similarity allows disparate patches sharing malignant textural features to directly exchange information.

---

### 2.3 Strategy 3: Hybrid Spatial + Feature-Similarity Graph

The hybrid graph merges local anatomical continuity with non-local semantic affinities.

#### Mathematical Definition:
The hybrid adjacency matrix $A_{\text{hybrid}}$ is defined as the set union of the spatial and feature-similarity edge sets:
$$A_{\text{hybrid}} = \text{clamp}(A_{\text{spatial}} + A_{\text{similarity}}, \, 0.0, \, 1.0)$$

#### Properties:
1. **Deduplication:** The binary clamp operation guarantees that any edge present in both the spatial grid and top-$k$ similarity is counted exactly once, eliminating multi-edges.
2. **Symmetry:** Because both $A_{\text{spatial}}$ and $A_{\text{similarity}}$ are symmetric, $A_{\text{hybrid}}$ is mathematically guaranteed to be symmetric (bidirectional).
3. **Guaranteed Spatial Scaffold:** The hybrid graph retains all 924 spatial edges as a base topological scaffold, supplemented by non-local semantic shortcuts. Total directed edges per image typically range between $1,800$ and $2,600$, representing a dense yet sparse relational topology.

---

## 3. Graph Validation & Batching Verification

All three graph construction implementations were verified via automated unit tests in `tests/test_graph_construction.py` and `tests/test_graph_batching.py`:

| Verification Test | Criterion Tested | Result |
|---|---|---|
| **Node Count** | Exactly 196 nodes per image graph | **PASSED** |
| **Feature Dimensions** | Exactly 192 dimensions per node | **PASSED** |
| **Spatial Edge Count** | Exactly 924 directed edges per image | **PASSED** |
| **Boundary Degrees** | Corners $= 3$, Borders $= 4$, Interior $= 5$ | **PASSED** |
| **Similarity Symmetry** | $A_{\text{sim}} = A_{\text{sim}}^T$ (Bidirectional) | **PASSED** |
| **Self-Loop Presence** | Diagonal $A_{ii} = 1.0$ for all nodes across all graphs | **PASSED** |
| **Hybrid Edge Union** | $A_{\text{hybrid}} \ge A_{\text{spatial}}$, no duplicate edges | **PASSED** |
| **Batch Isolation** | Zero edges between node $u$ in image $b_1$ and $v$ in image $b_2$ | **PASSED** |
| **Determinism** | Identical inputs produce identical adjacency matrices | **PASSED** |

### Batched Adjacency vs. Sparse Edge Index
Two complementary graph batching representations were implemented:
1. **Dense Adjacency Tensor $[B, 196, 196]$:** Used for high-speed GPU execution with dense tensor core matrix multiplications (`torch.matmul`). Because $196 \times 196 \times 4\text{ bytes} \approx 153\text{ KB}$ per image, batching 32 images requires only $\sim 4.9\text{ MB}$ of VRAM, providing extreme computational efficiency and zero external graph library overhead.
2. **Offset Sparse Edge Index $[2, \sum E_b]$:** Offsets node indices by $b \times 196$ for image $b$, guaranteeing complete mathematical isolation between distinct images.
