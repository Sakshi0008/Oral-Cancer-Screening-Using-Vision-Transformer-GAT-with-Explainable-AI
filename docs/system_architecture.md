# ORALSCAN AI: System Architecture Documentation

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Version:** 1.0.0 (Phase 7 Final Release)  
**Status:** Certified Academic Research Prototype  

---

## 1. High-Level System Overview

ORALSCAN AI is a multimodal, explainable deep learning screening system designed to assist clinical researchers and dental practitioners in screening clinical photographs of oral cavity lesions for suspicious tissue patterns.

The software architecture is decoupled into three primary tiers:
1. **Frontend Presentation Layer (Single-Page Web Application):** Built using vanilla HTML5, modern CSS3, and ES6 JavaScript with zero external npm build dependencies. It communicates with the backend via RESTful JSON APIs and serves interactive XAI overlays and downloadable PDF clinical reports.
2. **Backend API and Application Server Layer (FastAPI & SQLite):** An asynchronous Python service providing session-based authentication (PBKDF2-HMAC-SHA256), cryptographic checkpoint verification, image validation, user data isolation, and persistent screening records.
3. **Deep Multimodal Inference & XAI Engine Layer (PyTorch):** A singleton model execution pipeline integrating a custom Vision Transformer (ViT), a dynamic Graph Attention Network (GATv2), and a Concatenation Fusion Head with 4 Explainable AI modules (Attention Rollout, GAT Dynamic Attention, Fusion Integrated Gradients, and Automated Clinical Narrative Generation).

```mermaid
flowchart TD
    User["Clinician / User"] -->|Uploads Oral Photo (224x224 RGB)| Frontend["Frontend SPA (HTML5/CSS3/Vanilla JS)"]
    Frontend -->|POST /api/screen (Bearer Token)| BackendAPI["Backend REST API (FastAPI)"]
    
    subgraph Security_Storage ["Security & Storage"]
        BackendAPI --> DB[("SQLite Database (oralscan.db)")]
        BackendAPI --> Uploads["Upload Storage (uploads/)"]
        BackendAPI --> Outputs["Output Cache (outputs/)"]
    end
    
    subgraph Model_Pipeline ["Multimodal Inference Engine (2,238,406 Parameters)"]
        BackendAPI --> ViT["Vision Transformer Backbone (1,965,890 params)"]
        ViT -->|Patch Tokens [1, 196, 192]| GraphBuilder["Hybrid Graph Builder (Spatial + Top-8 Cosine)"]
        ViT -->|CLS Token [1, 192]| FusionHead["Concatenation Head (49,794 params)"]
        GraphBuilder -->|Adjacency [1, 196, 196]| GAT["GATv2 Backbone (222,722 params)"]
        GAT -->|Graph Rep [1, 192]| FusionHead
        FusionHead --> Logits["Logits & Softmax Scores"]
    end
    
    subgraph XAI_Suite ["Explainable AI (XAI) Suite"]
        ViT -.-> Rollout["ViT Attention Rollout (224x224 Heatmap)"]
        GAT -.-> GraphViz["GATv2 Topological Overlay (3-Color Edges)"]
        FusionHead -.-> IG["Integrated Gradients (ViT vs GAT Share)"]
        Logits -.-> TextGen["Plain-Language Narrative Generator"]
    end
    
    XAI_Suite --> PDFEngine["PDF Report Engine (Matplotlib/Vector)"]
    PDFEngine --> Download["Clinical Research PDF Report"]
```

---

## 2. Deep Learning Component Specifications

The primary hybrid screening architecture consists of exactly **2,238,406 parameters** distributed across three modular components:

### 2.1 Vision Transformer Backbone (`models/vit_scratch_best.pth`)
- **Class:** `VisionTransformerScratch` (`src/models/vit_scratch.py`)
- **Pretrained Weights:** NONE (Trained strictly from scratch on domain data; zero ImageNet contamination).
- **Parameters:** 1,965,890 parameters (56 tensor keys).
- **Architecture:**
  - Input: $224 \times 224 \times 3$ RGB.
  - Patch Extraction: $16 \times 16$ non-overlapping patches ($14 \times 14 = 196$ spatial patches).
  - Projection Dimension: $d = 192$.
  - Transformer Depth: $L = 4$ layers.
  - Multi-Head Attention: $H = 3$ heads ($d_k = 64$ per head).
  - Feed-Forward Dimension: $d_{\text{ff}} = 768$.
  - Regularization: Dropout $0.1$.
- **Outputs:**
  - Class Token $\mathbf{z}_{\text{cls}} \in \mathbb{R}^{1 \times 192}$ (global visual context).
  - Patch Tokens $\mathbf{Z}_{\text{patch}} \in \mathbb{R}^{1 \times 196 \times 192}$ (spatial patch embeddings).

### 2.2 Graph Attention Network Backbone (`models/graph_hybrid_gatv2_best.pth`)
- **Class:** `GraphClassifier` with `OralScanGATv2Scratch` (`src/models/graph_classifier.py`)
- **Parameters:** 222,722 parameters (14 tensor keys).
- **Architecture:**
  - Nodes: $N = 196$ patch tokens ($d_{\text{in}} = 192$).
  - Graph Topology: Hybrid spatial 4-neighborhood grid unioned with top-$k=8$ cosine feature-similarity edges.
  - GAT Layers: 2 GATv2 layers with dynamic attention mechanism:
    $$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{x}_i \parallel \mathbf{W} \mathbf{x}_j]\right)$$
  - Attention Heads: $H = 4$ heads ($d_h = 48$ per head).
  - Graph Pooling: Mean pooling over all patch nodes.
- **Output:** Graph representation $\mathbf{z}_{\text{graph}} \in \mathbb{R}^{1 \times 192}$.

### 2.3 Concatenation Fusion Head (`models/hybrid_vit_gat_concat_best.pth`)
- **Class:** `ConcatenationFusion` (`src/models/hybrid_fusion.py`)
- **Parameters:** 49,794 parameters (6 tensor keys).
- **Architecture:**
  - Input: Concatenation $\mathbf{z} = [\mathbf{z}_{\text{cls}} \parallel \mathbf{z}_{\text{graph}}] \in \mathbb{R}^{1 \times 384}$.
  - Layer 0: `Linear(384, 128)`.
  - Layer 1: `LayerNorm(128)` ($\epsilon = 10^{-5}$).
  - Layer 2: `ReLU()`.
  - Layer 3: `Dropout(p=0.2)`.
  - Layer 4: `Linear(128, 2)`.
- **Outputs:** Logits $\mathbf{y} \in \mathbb{R}^{1 \times 2}$ and Softmax probabilities.

---

## 3. Database Schema and Security Architecture

The application persists user credentials and screening metadata in a relational SQLite store (`data/oralscan.db`).

### 3.1 Relational Tables
1. **`users`:** Stores user credentials with PBKDF2-HMAC-SHA256 password hashing (100,000 iterations) and 16-byte cryptographically secure random salt. Plaintext passwords are never stored or logged.
2. **`sessions`:** Tracks 32-byte hexadecimal bearer tokens with 7-day expiration timestamps.
3. **`screenings`:** Tracks screening records, model predictions, softmax scores, branch sensitivity percentages, XAI visual asset file paths, and generated PDF report paths.

### 3.2 Security Controls
- **User Isolation:** All screening queries enforce `WHERE id = ? AND user_id = ?`, preventing unauthorized cross-user record access.
- **Upload Validation:** Enforces magic-byte PIL content verification, file size caps ($10\text{ MB}$), allowed MIME types (JPEG, PNG), and safe UUID-based disk storage.
- **Zero Raw Checkpoint Exposure:** Checkpoint binaries are loaded in-process into memory at startup; no API endpoint exposes model weights or checkpoint internals.
