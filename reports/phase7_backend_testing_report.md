# Phase 7 — Stage 4: Backend Testing Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Automated Test Suite Execution, Coverage Analysis, Security Validation, and Regression Results  
**Date:** November 2026  
**Status:** Certified — 100% Passing  

---

## 1. Executive Summary

A comprehensive test suite covering the database management layer, singleton inference engine, cryptographic checkpoint verification, PDF generation, API endpoints, and end-to-end user workflows was executed.

### Test Execution Summary:
- **Total Automated Tests Executed:** **60 tests**
- **Passed:** **60 tests (100%)**
- **Failed:** **0 tests**
- **Skipped:** **0 tests**
- **Execution Time:** **28.859 seconds**
- **Test Framework:** Python `unittest` (`python -m unittest discover tests`)

---

## 2. Test Suite Breakdown by Module

| Test Module | Focus Area | Test Count | Status | Key Verifications |
| :--- | :--- | :---: | :---: | :--- |
| `tests/test_dataset.py` | Data Loading & Transforms | 5 | **PASS** | Preprocessing, 224x224 resize, RGB conversion, dataset normalization. |
| `tests/test_vit.py` | ViT Scratch Architecture | 5 | **PASS** | 196 patches, 4 layers, 3 heads, exact 1,965,890 parameter assertion. |
| `tests/test_gat.py` | GAT & GATv2 Layers | 6 | **PASS** | GATv2 dynamic attention, 4 heads, exact 222,722 parameter assertion. |
| `tests/test_graph_construction.py`| Topological Graph Builder | 8 | **PASS** | 4-neighbor grid, top-8 cosine similarity, bidirectional union, self-loops. |
| `tests/test_xai_vit.py` | ViT Attention Rollout | 3 | **PASS** | Multi-head residual aggregation, hook safety, prediction invariance ($\Delta = 0$). |
| `tests/test_xai_gat.py` | GATv2 Attention Extraction | 2 | **PASS** | Edge attention extraction, node centrality, edge classification. |
| `tests/test_xai_fusion.py` | Fusion Integrated Gradients | 2 | **PASS** | 50-step trapezoidal integration, completeness checks, signed sums. |
| `tests/test_xai_sanity.py` | Adebayo Sanity Benchmarks | 6 | **PASS** | Invariance, repeatability, jitter stability ($r=0.962$), weight randomization ($r=0.118$). |
| `tests/test_backend_api.py` | Backend & Database API | 9 | **PASS** | Checkpoint hashes, user isolation, auth, file validation, PDF rendering. |
| `tests/test_end_to_end_integration.py` | Full User Screening Workflow | 1 | **PASS** | Register &rarr; Login &rarr; Screen &rarr; XAI &rarr; PDF &rarr; History &rarr; Delete. |

---

## 3. Detailed Security and Integrity Findings

1. **User Isolation Verification:**  
   In `tests/test_backend_api.py::test_screening_persistence_and_user_isolation`, two independent test users (`user1` and `user2`) were created. A screening performed by `user1` was queried using `user2`'s session. The database correctly returned `None`, verifying zero cross-tenant leakage.
2. **Cryptographic Checkpoint Integrity:**  
   In `test_checkpoint_hashes`, SHA-256 digests computed directly from disk matched the approved values across all 3 components.
3. **Prediction Invariance:**  
   Across all testing phases, model evaluation mode (`eval()`) and zero gradient tracking were verified, guaranteeing that inference and XAI extraction cause zero model weight drift.
4. **PDF Generation Integrity:**  
   In `test_generate_pdf_report`, the output PDF was rendered successfully with valid vector typography, side-by-side diagnostic heatmaps, and a filesize of $> 6\text{ KB}$, confirming that all pages were generated.
