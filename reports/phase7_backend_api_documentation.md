# ORALSCAN AI: Backend REST API Reference Documentation

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Base URL:** `http://127.0.0.1:8000`  
**Protocol:** HTTP/1.1 REST + JSON (with Multipart Uploads)  
**Authentication:** HTTP Bearer Token (`Authorization: Bearer <token>` or `?auth=<token>` for media)  

---

## 1. Authentication & User Management Endpoints

### 1.1 Register User
- **Endpoint:** `POST /api/auth/register`
- **Request Body (JSON):**
  ```json
  {
    "username": "clinician_jane",
    "email": "jane@oralclinic.org",
    "password": "SecurePassword123!"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "success": true,
    "message": "User registered successfully.",
    "token": "4a7b9c... (64-character hex string)",
    "user": {
      "id": 1,
      "username": "clinician_jane",
      "email": "jane@oralclinic.org"
    }
  }
  ```

### 1.2 Sign In (Login)
- **Endpoint:** `POST /api/auth/login`
- **Request Body (JSON):**
  ```json
  {
    "username_or_email": "clinician_jane",
    "password": "SecurePassword123!"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "success": true,
    "message": "Login successful.",
    "token": "4a7b9c...",
    "user": { "id": 1, "username": "clinician_jane", "email": "jane@oralclinic.org" }
  }
  ```

### 1.3 User Profile & Statistics
- **Endpoint:** `GET /api/auth/me`
- **Headers:** `Authorization: Bearer <token>`
- **Response (200 OK):**
  ```json
  {
    "user": { "id": 1, "username": "clinician_jane", "email": "jane@oralclinic.org" },
    "stats": {
      "total_screenings": 12,
      "suspicious_count": 4,
      "benign_count": 8
    }
  }
  ```

---

## 2. Model Status & Screening Endpoints

### 2.1 Model Status
- **Endpoint:** `GET /api/model/status`
- **Response (200 OK):**
  ```json
  {
    "model_name": "ORALSCAN AI (ViT-GATv2 Hybrid)",
    "device": "cuda",
    "architecture": {
      "vit_parameters": 1965890,
      "gatv2_parameters": 222722,
      "fusion_parameters": 49794,
      "total_parameters": 2238406
    },
    "classes": { "0": "NON CANCER", "1": "CANCER" },
    "operational": true
  }
  ```

### 2.2 Execute Image Screening
- **Endpoint:** `POST /api/screen`
- **Headers:** `Authorization: Bearer <token>`
- **Content-Type:** `multipart/form-data`
- **Query Parameters:** `run_perturbation=false` (optional boolean)
- **Form Data:** `file`: Clinical image binary (JPEG or PNG, max 10MB)
- **Response (200 OK):**
  ```json
  {
    "screening_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "pred_label": 1,
    "pred_name": "CANCER",
    "cautious_result": "Suspicious pattern detected by the model.",
    "cancer_score": 65.64,
    "confidence": 65.64,
    "model_score_explanation": "This score represents the uncalibrated softmax confidence...",
    "clinical_guidance": "The artificial intelligence screening model detected visual and topological patterns...",
    "fusion_attribution": {
      "vit_share_pct": 66.13,
      "gat_share_pct": 33.87,
      "vit_signed_sum": -0.4893,
      "gat_signed_sum": 0.9751,
      "ig_absolute_error": 0.2691,
      "ig_relative_error_pct": 124.14,
      "baseline_type": "Computational reference vector [0.05]^384 (Representation space)",
      "interpretation_notice": "Branch shares represent absolute path-integrated gradient sensitivity..."
    },
    "narrative_explanation": "PLAIN-LANGUAGE AI EXPLANATION:\nVisual attention focused on focal lesion zones...",
    "visualizations": {
      "multipanel_path": "outputs/visualizations/..._xai_explanation.png",
      "rollout_path": "outputs/visualizations/..._vit_rollout.png",
      "graph_path": "outputs/visualizations/..._gat_graph.png"
    },
    "latency": { "total_ms": 426.9, "breakdown": { ... } },
    "disclaimer": "RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSIS..."
  }
  ```

---

## 3. PDF & Media Endpoints

### 3.1 Download Clinical PDF Report
- **Endpoint:** `GET /api/screen/{screening_id}/pdf`
- **Headers:** `Authorization: Bearer <token>` (or `?auth=<token>`)
- **Response (200 OK):** Streamed vector PDF file (`Content-Type: application/pdf`).

### 3.2 Fetch Generated XAI Visual Overlay
- **Endpoint:** `GET /api/screen/{screening_id}/visual/{viz_type}`
- **Path Parameters:** `viz_type` $\in$ `{"rollout", "graph", "multipanel"}`
- **Headers:** `Authorization: Bearer <token>` (or `?auth=<token>`)
- **Response (200 OK):** Streamed image file (`Content-Type: image/png`).

---

## 4. Screening History Endpoints

### 4.1 List Past Screenings
- **Endpoint:** `GET /api/history`
- **Query Parameters:** `limit=50` (optional)
- **Headers:** `Authorization: Bearer <token>`
- **Response (200 OK):** List of past screening summary objects.

### 4.2 Delete Screening Record
- **Endpoint:** `DELETE /api/history/{screening_id}`
- **Headers:** `Authorization: Bearer <token>`
- **Response (200 OK):** `{"success": true, "message": "Screening record deleted successfully."}`
