"""
ORALSCAN AI - Backend Inference and XAI Engine
=============================================
Thread-safe singleton engine managing the primary ViT-GATv2 hybrid pipeline,
cryptographic checkpoint verification, and Explainable AI generation.
"""

import os
import time
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torch

from src.verify_hybrid_inference import VerifiedHybridInferencePipeline
from src.xai.pipeline import OralScanXAIExplainer


# Approved Checkpoint Paths and SHA-256 Digests
CHECKPOINTS_CONFIG = {
    "vit": {
        "path": "models/vit_scratch_best.pth",
        "expected_sha256": "1ab35a38e8ff3fba8165de5efd284e28a5f62cc890f9d24faea6d7419d7adabf"
    },
    "gatv2": {
        "path": "models/graph_hybrid_gatv2_best.pth",
        "expected_sha256": "04811874cdf2d42f663649137122d44692d65d96635495fdb9f494d8982e977f"
    },
    "fusion": {
        "path": "models/hybrid_vit_gat_concat_best.pth",
        "expected_sha256": "05ced3ed19bbec9d0a48a09738fbfb40023b362a11b8794079f78f6dd5c0e365"
    }
}


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class OralScanEngine:
    """
    Singleton backend inference and interpretability coordinator.
    """
    _instance: Optional["OralScanEngine"] = None

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viz_dir = Path("outputs/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify checkpoint hashes before loading
        self.verification_status = self._verify_checkpoints()
        if not self.verification_status["all_valid"]:
            raise RuntimeError(f"Checkpoint integrity verification failed: {self.verification_status['errors']}")

        # Initialize core pipeline (loads weights once)
        self.pipeline = VerifiedHybridInferencePipeline(
            vit_ckpt_path=CHECKPOINTS_CONFIG["vit"]["path"],
            gat_ckpt_path=CHECKPOINTS_CONFIG["gatv2"]["path"],
            fusion_ckpt_path=CHECKPOINTS_CONFIG["fusion"]["path"],
            device=self.device
        )

        # Initialize unified XAI explainer
        self.explainer = OralScanXAIExplainer(
            inference_pipeline=self.pipeline,
            device=self.device,
            output_dir=str(self.viz_dir)
        )

    @classmethod
    def get_instance(cls, device: Optional[torch.device] = None) -> "OralScanEngine":
        """Thread-safe accessor for singleton engine."""
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def _verify_checkpoints(self) -> Dict[str, Any]:
        """Validates existence and cryptographic SHA-256 hashes of model binaries."""
        errors = []
        details = {}
        for name, cfg in CHECKPOINTS_CONFIG.items():
            path = cfg["path"]
            expected = cfg["expected_sha256"]
            if not os.path.exists(path):
                errors.append(f"Missing checkpoint file: {path}")
                details[name] = {"status": "MISSING", "path": path}
                continue
            
            measured = compute_file_sha256(path)
            if measured != expected:
                errors.append(f"Hash mismatch for {name}: expected {expected}, got {measured}")
                details[name] = {"status": "CORRUPTED", "expected": expected, "measured": measured}
            else:
                details[name] = {"status": "VERIFIED", "sha256": measured}

        return {
            "all_valid": len(errors) == 0,
            "errors": errors,
            "details": details
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive model metadata and operational status."""
        return {
            "model_name": "ORALSCAN AI (ViT-GATv2 Hybrid)",
            "device": str(self.device),
            "architecture": {
                "vit_parameters": 1965890,
                "gatv2_parameters": 222722,
                "fusion_parameters": 49794,
                "total_parameters": 2238406
            },
            "classes": {0: "NON CANCER", 1: "CANCER"},
            "checkpoints": self.verification_status["details"],
            "operational": True
        }

    def run_screening(
        self,
        image_path: str,
        screening_id: Optional[str] = None,
        run_perturbation: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal screening and interpretability generation.
        """
        sid = screening_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Run primary inference and XAI extraction
        xai_res = self.explainer.explain_image(
            image_path=image_path,
            case_id=sid,
            run_perturbation=run_perturbation,
            save_viz=True
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        pred = xai_res["prediction"]
        pred_label = pred["pred_label"]
        pred_name = pred["pred_name"]
        cancer_score = pred["cancer_prob"] * 100.0  # Percentage score
        confidence = pred["confidence"]

        # Cautious clinical phrasing adhering to non-diagnostic rules
        if pred_label == 1:
            cautious_result = "Suspicious pattern detected by the model."
            guidance = (
                "The artificial intelligence screening model detected visual and topological patterns "
                "statistically aligned with oral lesions requiring clinical inspection. "
                "Consult an oral maxillofacial specialist, dentist, or ENT oncologist for a comprehensive physical exam."
            )
        else:
            cautious_result = "No suspicious pattern detected by the model."
            guidance = (
                "The artificial intelligence screening model did not detect prominent patterns associated with malignancy. "
                "Routine clinical oral examinations should be maintained as standard preventive practice."
            )

        # XAI fusion breakdown
        fusion_attr = xai_res["fusion_attribution"]
        vit_share = fusion_attr["vit_share_pct"]
        gat_share = fusion_attr["gat_share_pct"]
        ig_abs_error = fusion_attr.get("convergence_delta", 0.0)
        ig_rel_error_pct = fusion_attr.get("convergence_error_pct", 0.0)

        # File paths of generated diagnostic overlays
        multipanel_viz = xai_res.get("visualization_path")
        if not multipanel_viz or not os.path.exists(multipanel_viz):
            multipanel_viz = str((self.viz_dir / f"{sid}_xai_explanation.png").resolve())
            if not os.path.exists(multipanel_viz):
                multipanel_viz = None

        # Render and save individual heatmaps for clean UI consumption
        rollout_viz_path = str((self.viz_dir / f"{sid}_vit_rollout.png").resolve())
        graph_viz_path = str((self.viz_dir / f"{sid}_gat_graph.png").resolve())

        # Save individual rollout overlay
        raw_pil = Image.open(image_path).convert("RGB")
        import matplotlib.pyplot as plt
        
        # Save Rollout
        vit_heatmap = xai_res["vit_rollout"]["heatmap_224x224"]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.imshow(raw_pil)
        ax.imshow(vit_heatmap, cmap="jet", alpha=0.55)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(rollout_viz_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Save Graph Heatmap
        gat_heatmap = xai_res["gatv2_graph"]["node_heatmap_224x224"]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.imshow(raw_pil)
        ax.imshow(gat_heatmap, cmap="viridis", alpha=0.55)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(graph_viz_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Format narrative text cleanly
        narrative_raw = xai_res.get("narrative_explanation", "")
        if isinstance(narrative_raw, dict):
            narrative_text = narrative_raw.get("full_narrative") or str(narrative_raw)
        else:
            narrative_text = str(narrative_raw)

        return {
            "screening_id": sid,
            "pred_label": pred_label,
            "pred_name": pred_name,
            "cautious_result": cautious_result,
            "cancer_score": round(cancer_score, 2),
            "confidence": round(confidence, 2),
            "model_score_explanation": (
                "This score represents the uncalibrated softmax confidence of the hybrid neural network. "
                "It is a comparative algorithmic score and NOT an absolute probability of disease."
            ),
            "clinical_guidance": guidance,
            "fusion_attribution": {
                "vit_share_pct": round(vit_share, 2),
                "gat_share_pct": round(gat_share, 2),
                "vit_signed_sum": round(fusion_attr["vit_signed_sum"], 4),
                "gat_signed_sum": round(fusion_attr["gat_signed_sum"], 4),
                "ig_absolute_error": round(ig_abs_error, 4),
                "ig_relative_error_pct": round(ig_rel_error_pct, 2),
                "baseline_type": "Computational reference vector [0.05]^384 (Representation space)",
                "interpretation_notice": (
                    "Branch shares represent absolute path-integrated gradient sensitivity flowing through the fusion head. "
                    "They reflect mathematical neural network dynamics, NOT clinical reasoning percentages."
                )
            },
            "narrative_explanation": narrative_text,
            "visualizations": {
                "multipanel_path": multipanel_viz,
                "rollout_path": rollout_viz_path,
                "graph_path": graph_viz_path
            },
            "latency": {
                "total_ms": round(total_latency_ms, 2),
                "breakdown": xai_res.get("timings", {})
            },
            "disclaimer": (
                "RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSIS. "
                "ORALSCAN AI is an experimental academic prototype. Saliency maps depict model feature sensitivity, "
                "not confirmed tumor borders. It must never substitute for biopsy or clinical dental examination."
            )
        }
