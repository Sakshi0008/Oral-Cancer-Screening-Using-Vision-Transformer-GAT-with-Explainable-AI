"""
ORALSCAN AI - Primary FastAPI Application Server
===============================================
Production-grade RESTful API connecting the ViT-GATv2 hybrid inference pipeline,
XAI visual engines, user authentication, and PDF report generation.
"""

import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header, status, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, os.getcwd())
from backend.database import DatabaseManager
from backend.inference_engine import OralScanEngine
from backend.pdf_generator import ScreeningPDFGenerator


# Initialize Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Database and Engines
db = DatabaseManager()
engine = OralScanEngine.get_instance()
pdf_gen = ScreeningPDFGenerator()

# FastAPI Application Definition
app = FastAPI(
    title="ORALSCAN AI API",
    description="Multimodal Oral Cancer Screening Using Vision Transformer & Graph Attention Network with Explainable AI",
    version="1.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request / Response Pydantic Schemas
# =============================================================================

class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    username_or_email: str
    password: str


# =============================================================================
# Authentication Dependency
# =============================================================================

def get_current_user(
    authorization: Optional[str] = Header(None),
    auth: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Validates session token from either Authorization header or auth query param (for <img> tags)."""
    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        else:
            token = authorization
    elif auth:
        token = auth

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing. Please log in."
        )

    user = db.get_user_by_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again."
        )
    return user


# =============================================================================
# System Health & Model Status Endpoints
# =============================================================================

@app.get("/api/health")
def health_check():
    """Health check verifying API readiness."""
    return {
        "status": "healthy",
        "service": "ORALSCAN AI Screening API",
        "timestamp": os.path.getmtime("models/vit_scratch_best.pth")
    }


@app.get("/api/model/status")
def model_status():
    """Returns architecture specs and checkpoint integrity details."""
    return engine.get_status()


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/api/auth/register")
def register(req: UserRegisterRequest):
    """Registers a new user account."""
    try:
        user = db.register_user(req.username, req.email, req.password)
        token = db.create_session(user["id"])
        return {
            "success": True,
            "message": "User registered successfully.",
            "token": token,
            "user": user
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/auth/login")
def login(req: UserLoginRequest):
    """Authenticates credentials and issues a session token."""
    user = db.authenticate_user(req.username_or_email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    token = db.create_session(user["id"])
    return {
        "success": True,
        "message": "Login successful.",
        "token": token,
        "user": user
    }


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Terminates active session."""
    if authorization:
        parts = authorization.split()
        token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else authorization
        db.delete_session(token)
    return {"success": True, "message": "Logged out successfully."}


@app.get("/api/auth/me")
def get_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns profile and dashboard summary statistics."""
    stats = db.get_user_stats(current_user["id"])
    return {
        "user": current_user,
        "stats": stats
    }


# =============================================================================
# Screening & Inference Endpoints
# =============================================================================

@app.post("/api/screen")
async def screen_oral_image(
    file: UploadFile = File(...),
    run_perturbation: bool = Query(False, description="Whether to execute optional perturbation test"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Primary screening endpoint:
    Accepts clinical lesion photo, executes ViT-GATv2 inference, extracts XAI, and persists record.
    """
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a JPEG or PNG image."
        )

    # 2. Save temporary upload safely
    screening_id = str(uuid.uuid4())
    safe_filename = f"{screening_id}{ext}"
    saved_path = UPLOAD_DIR / safe_filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save image file.")

    # 3. Validate image integrity and file size
    if saved_path.stat().st_size > 10 * 1024 * 1024:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image size exceeds 10MB limit.")

    try:
        with Image.open(saved_path) as img:
            img.verify()
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or invalid image file.")

    # 4. Execute Real Multimodal Inference + XAI Pipeline
    try:
        result = engine.run_screening(
            image_path=str(saved_path.resolve()),
            screening_id=screening_id,
            run_perturbation=run_perturbation
        )
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference execution failed: {str(e)}")

    # 5. Persist record in database
    viz = result["visualizations"]
    fusion_attr = result["fusion_attribution"]
    db.save_screening(
        screening_id=screening_id,
        user_id=current_user["id"],
        original_filename=file.filename,
        stored_image_path=str(saved_path.resolve()),
        pred_label=result["pred_label"],
        pred_name=result["pred_name"],
        cautious_result=result["cautious_result"],
        cancer_score=result["cancer_score"],
        confidence=result["confidence"],
        vit_share_pct=fusion_attr["vit_share_pct"],
        gat_share_pct=fusion_attr["gat_share_pct"],
        ig_abs_error=fusion_attr["ig_absolute_error"],
        ig_rel_error_pct=fusion_attr["ig_relative_error_pct"],
        narrative=str(result["narrative_explanation"]),
        multipanel_viz_path=viz["multipanel_path"],
        rollout_viz_path=viz["rollout_path"],
        graph_viz_path=viz["graph_path"],
        has_perturbation=run_perturbation
    )

    return result


@app.get("/api/screen/{screening_id}")
def get_screening_record(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves screening details strictly enforcing ownership."""
    record = db.get_screening(screening_id, current_user["id"])
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")
    return record


@app.get("/api/screen/{screening_id}/explanation")
def get_screening_explanation(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns narrative explanation and visualization metadata."""
    record = db.get_screening(screening_id, current_user["id"])
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")

    return {
        "screening_id": record["id"],
        "pred_name": record["pred_name"],
        "cautious_result": record["cautious_result"],
        "vit_share_pct": record["vit_share_pct"],
        "gat_share_pct": record["gat_share_pct"],
        "narrative": record["narrative"],
        "visualizations": {
            "rollout_url": f"/api/screen/{screening_id}/visual/rollout",
            "graph_url": f"/api/screen/{screening_id}/visual/graph",
            "multipanel_url": f"/api/screen/{screening_id}/visual/multipanel"
        }
    }


@app.get("/api/screen/{screening_id}/image")
def get_original_image(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Serves the uploaded patient image with ownership validation."""
    record = db.get_screening(screening_id, current_user["id"])
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")

    img_path = Path(record["stored_image_path"])
    if not img_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found on disk.")

    return FileResponse(str(img_path))


@app.get("/api/screen/{screening_id}/visual/{viz_type}")
def get_visualization_image(
    screening_id: str,
    viz_type: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Serves generated XAI visual overlays safely."""
    record = db.get_screening(screening_id, current_user["id"])
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")

    if viz_type == "rollout":
        path = record["rollout_viz_path"]
    elif viz_type == "graph":
        path = record["graph_viz_path"]
    elif viz_type == "multipanel":
        path = record["multipanel_viz_path"]
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid visualization type.")

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visualization image not found.")

    return FileResponse(path)


# =============================================================================
# PDF Report Endpoint
# =============================================================================

@app.get("/api/screen/{screening_id}/pdf")
def download_pdf_report(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generates on-demand or retrieves existing PDF screening report."""
    record = db.get_screening(screening_id, current_user["id"])
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")

    pdf_path = record.get("pdf_report_path")
    if not pdf_path or not os.path.exists(pdf_path):
        # Generate fresh PDF
        pdf_path = pdf_gen.generate_report(
            screening_data=record,
            raw_image_path=record["stored_image_path"],
            rollout_viz_path=record["rollout_viz_path"],
            graph_viz_path=record["graph_viz_path"]
        )
        db.update_pdf_path(screening_id, current_user["id"], pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"oralscan_report_{screening_id[:8]}.pdf"
    )


# =============================================================================
# User Screening History Endpoints
# =============================================================================

@app.get("/api/history")
def get_user_history(
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Returns the user's screening history."""
    return db.list_user_screenings(current_user["id"], limit=limit)


@app.delete("/api/history/{screening_id}")
def delete_screening(
    screening_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Deletes a screening record owned by the user."""
    success = db.delete_screening(screening_id, current_user["id"])
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Screening record not found.")
    return {"success": True, "message": "Screening record deleted successfully."}


# =============================================================================
# Static Files & Frontend SPA Mount
# =============================================================================

FRONTEND_DIR = Path("frontend")
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# Mount static frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Mount sample dataset images for interactive demonstration if directory exists
if Path("Oral Cancer").exists():
    app.mount("/Oral Cancer", StaticFiles(directory="Oral Cancer"), name="samples")


@app.get("/")
def serve_index():
    """Serves the primary Single-Page Application interface."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        content={
            "app": "ORALSCAN AI API Server",
            "frontend": "Frontend files being initialized under /frontend",
            "docs": "/docs"
        }
    )
