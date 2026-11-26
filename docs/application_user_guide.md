# ORALSCAN AI: Application User & Operation Guide

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Deployment Instructions, API Usage, Web Interface Navigation, and Demonstration Procedures  
**Date:** November 2026  

---

## 1. System Requirements & Environment Setup

### 1.1 Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** Python 3.10 or 3.11
- **Hardware:**
  - GPU: CUDA-compatible GPU (e.g., NVIDIA GeForce RTX 3050/4050 or higher with $\ge 4\text{ GB}$ VRAM) recommended for $< 500\text{ ms}$ latency.
  - CPU: 4+ CPU cores with $\ge 8\text{ GB}$ RAM (CPU fallback mode supported automatically).

### 1.2 Verification of Approved Model Checkpoints
Before launching the application, confirm that the three required checkpoints exist in `models/`:
```bash
python -c "
import hashlib
checkpoints = {
    'vit': ('models/vit_scratch_best.pth', '1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf'),
    'gatv2': ('models/graph_hybrid_gatv2_best.pth', '04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f'),
    'fusion': ('models/hybrid_vit_gat_concat_best.pth', '05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365')
}
for name, (path, expected) in checkpoints.items():
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assert digest == expected, f'Checkpoint {name} hash mismatch!'
print('All 3 approved checkpoints verified successfully.')
"
```

---

## 2. Starting the Application Server

The entire frontend, backend API, database, and inference engine are unified under Uvicorn/FastAPI.

To start the server:
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
- **Web User Interface:** Open your web browser and navigate to: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation:** Available at: `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `http://127.0.0.1:8000/api/health`

---

## 3. End-to-End Demonstration Walkthrough

Follow these steps to demonstrate the full application workflow to faculty or clinical reviewers:

### Step 1: User Account Registration & Sign In
1. On the landing page, click **"Register"** or **"Sign In"** in the top navigation bar.
2. Enter a username (e.g., `dr_sharma`), email (`sharma@hospital.org`), and secure password.
3. Upon registration, a secure session token is generated, and the navigation switches to authenticated mode with access to the **Dashboard** and **Screening History**.

### Step 2: Uploading a Clinical Oral Image
1. Click **"Screening Portal"** in the navigation bar.
2. Drag and drop a clinical oral lesion image into the upload box (or click to browse).
3. Alternatively, click one of the quick test chips to load a sample validation case:
   - `Sample 1: Cancer Case (val_01)` (`Oral Cancer/Oral Cancer Dataset/CANCER/005.jpeg`)
   - `Sample 2: Benign Case (val_04)` (`Oral Cancer/Oral Cancer Dataset/NON CANCER/007.jpeg`)
4. Preview the image in real time.

### Step 3: Executing Screening Inference
1. Click **"Analyze Lesion with Hybrid Model"**.
2. An animated progress bar shows the real-time execution of the 6 pipeline stages:
   - Image Normalization
   - Vision Transformer Patch Encoding (196 nodes)
   - Dynamic Hybrid Graph Construction (Spatial + Top-8 Cosine)
   - GATv2 Relational Message Passing
   - Multimodal Concatenation
   - Integrated Gradients & Saliency Rollout
3. Within $< 500\text{ ms}$ (on GPU), the application transitions to the **Diagnostic & Interpretability View**.

### Step 4: Reviewing Results and Explainable AI Overlays
1. **Model Finding Banner:** Review the cautious finding (e.g., *"Suspicious pattern detected by the model."*) and the continuous softmax score (e.g., `65.6%`).
2. **Interactive 4-Panel Visualization:**
   - *Panel 1:* Uploaded lesion photograph.
   - *Panel 2:* ViT Attention Rollout heatmap showing spatial tissue zones with elevated visual attention.
   - *Panel 3:* GATv2 graph overlay showing inter-patch relational connectivity (Cyan = spatial, Gold = feature similarity).
   - *Panel 4:* Fusion sensitivity breakdown bar showing the relative share of visual versus relational branches (e.g., `ViT: 66.1%`, `GAT: 33.9%`).
3. **Plain-Language Narrative:** Read the synthesized plain-language clinical explanation summarizing model behavior.
4. **Clinical Referral Protocols:** Review standard follow-up guidance for in-person dental examination and biopsy.

### Step 5: Generating and Downloading the PDF Report
1. Click the **"Download PDF Report"** button in the header.
2. The server autonomously renders a vector-quality, two-page clinical research screening report containing the screening ID, timestamp, full visual overlays, branch attribution metrics, and mandatory regulatory disclaimers.
3. Save or open the downloaded file (`oralscan_report_<id>.pdf`).

### Step 6: Screening History and Record Deletion
1. Click **"History"** in the top navigation bar.
2. Verify that the newly completed screening appears with its date, thumbnail, model score, and result status.
3. Click **"PDF"** to redownload the report at any time.
4. Click **"Delete"** to permanently remove the screening record and associated disk assets, verifying strict user data management.

---

## 4. Automated Testing Procedures

Run the complete 60-test verification suite to confirm system integrity:
```bash
python -m unittest discover tests
```
All 60 tests must pass (`Ran 60 tests ... OK`).
