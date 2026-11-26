# Phase 7 — Stage 4: Backend Implementation Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Backend REST API, Database Architecture, Checkpoint Verification, and Upload Security Implementation  
**Date:** November 2026  
**Status:** Certified and Tested  

---

## 1. Executive Summary

Stage 4 of Phase 7 has been successfully executed. A robust, production-grade asynchronous backend service was developed using FastAPI, SQLite, and PyTorch, adhering strictly to the frozen model configuration and security guidelines:

1. **Singleton Inference Engine (`backend/inference_engine.py`):** Encapsulates the approved 2,238,406 parameter ViT-GATv2 hybrid pipeline, loading model weights into memory once at startup.
2. **Cryptographic Checkpoint Verification:** SHA-256 hashes of all three required model checkpoints (`models/vit_scratch_best.pth`, `models/graph_hybrid_gatv2_best.pth`, and `models/hybrid_vit_gat_concat_best.pth`) are verified before initialization, blocking any execution on corrupted or modified weights.
3. **Relational Database & Auth Layer (`backend/database.py`):** Implements user authentication using PBKDF2-HMAC-SHA256 password hashing with 100,000 iterations and 16-byte random salts. Session tokens are 32-byte cryptographically secure random hexadecimal strings. User ownership is strictly enforced across all screening queries.
4. **Upload Security & Preprocessing:** Uploaded files undergo PIL content verification, MIME-type checks, and a 10MB size cap. Approved domain normalization ($\mu = [0.6432, 0.4537, 0.4275]$, $\sigma = [0.2189, 0.2118, 0.2090]$) and $224 \times 224$ bilinear resizing are applied deterministically.
5. **PDF Screening Report Engine (`backend/pdf_generator.py`):** Autonomously renders two-page vector-quality clinical research reports documenting model predictions, softmax scores, visual XAI overlays, and mandatory regulatory disclaimers.

---

## 2. Source Code Architecture

| Component File | Role & Primary Class | Key Responsibilities |
| :--- | :--- | :--- |
| `backend/database.py` | `DatabaseManager` | SQLite CRUD, PBKDF2 hashing, session management, user isolation. |
| `backend/inference_engine.py` | `OralScanEngine` | Singleton coordinator, SHA-256 verification, multimodal inference, XAI rendering. |
| `backend/pdf_generator.py` | `ScreeningPDFGenerator` | Multi-page clinical research PDF synthesis using vector Matplotlib backends. |
| `backend/app.py` | FastAPI Application | REST endpoints, CORS, request validation, static frontend SPA hosting. |

---

## 3. Verified Model Specifications

- **Vision Transformer Backbone:** 1,965,890 parameters (`models/vit_scratch_best.pth`)  
  *SHA-256:* `1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf` (MATCH)
- **GATv2 Backbone:** 222,722 parameters (`models/graph_hybrid_gatv2_best.pth`)  
  *SHA-256:* `04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f` (MATCH)
- **Concatenation Fusion Head:** 49,794 parameters (`models/hybrid_vit_gat_concat_best.pth`)  
  *SHA-256:* `05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365` (MATCH)
- **Total Parameters:** **2,238,406**

---

## 4. API Endpoints Implemented

1. `GET /api/health` — Service readiness and checkpoint timestamp.
2. `GET /api/model/status` — Operational status, parameter counts, and checkpoint hashes.
3. `POST /api/auth/register` — User registration with email/username uniqueness checks.
4. `POST /api/auth/login` — Credential verification and session token issuance.
5. `POST /api/auth/logout` — Revocation of active session token.
6. `GET /api/auth/me` — User profile and summary screening statistics.
7. `POST /api/screen` — Multipart image upload, ViT-GATv2 inference, XAI extraction, and database persistence.
8. `GET /api/screen/{id}` — Retrieval of screening record enforcing user ownership.
9. `GET /api/screen/{id}/explanation` — Narrative explanation and visualization URLs.
10. `GET /api/screen/{id}/image` — Secure retrieval of original lesion image.
11. `GET /api/screen/{id}/visual/{type}` — Secure retrieval of generated XAI overlays (rollout, graph, multipanel).
12. `GET /api/screen/{id}/pdf` — On-demand generation and download of screening PDF report.
13. `GET /api/history` — Chronological listing of user's past screenings.
14. `DELETE /api/history/{id}` — Permanent removal of screening record.
15. `GET /` — Serves Single-Page Application (`frontend/index.html`).

---
*Report certified by autonomous research verification pipeline.*
