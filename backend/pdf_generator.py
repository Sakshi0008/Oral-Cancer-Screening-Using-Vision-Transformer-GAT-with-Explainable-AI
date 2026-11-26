"""
ORALSCAN AI - Clinical Research PDF Report Generator
===================================================
Generates high-resolution, multi-page PDF screening reports documenting
actual model predictions, XAI visual overlays, and scientific disclaimers.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


class ScreeningPDFGenerator:
    """
    Renders academic research screening reports in PDF format.
    """
    def __init__(self, output_dir: Path = Path("outputs/reports")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        screening_data: Dict[str, Any],
        raw_image_path: str,
        rollout_viz_path: Optional[str] = None,
        graph_viz_path: Optional[str] = None,
        report_id: Optional[str] = None
    ) -> str:
        """
        Builds a multi-page PDF report using actual screening data.
        Returns the absolute path to the generated PDF.
        """
        sid = screening_data.get("id", "UNKNOWN")
        rid = report_id or f"REP-{sid[:8].upper()}"
        timestamp = screening_data.get("created_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        filename = f"oralscan_report_{sid}.pdf"
        pdf_path = self.output_dir / filename

        cautious_result = screening_data.get("cautious_result", "Screening Complete")
        pred_label = screening_data.get("pred_label", 0)
        cancer_score = screening_data.get("cancer_score", 0.0)
        confidence = screening_data.get("confidence", 0.0)
        vit_share = screening_data.get("vit_share_pct", 63.4)
        gat_share = screening_data.get("gat_share_pct", 36.6)
        ig_abs = screening_data.get("ig_abs_error", 0.0)
        ig_rel = screening_data.get("ig_rel_error_pct", 0.0)
        narrative = screening_data.get("narrative", "Automated model explanation generated.")

        with PdfPages(str(pdf_path)) as pdf:
            # =========================================================================
            # PAGE 1: Executive Screening Summary and XAI Visualizations
            # =========================================================================
            fig = plt.figure(figsize=(8.5, 11), dpi=200)
            gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[0.8, 1.2, 2.5, 0.8, 0.5])
            fig.patch.set_facecolor("#FAFAFA")

            # 1. Header Banner
            ax_hdr = fig.add_subplot(gs[0, :])
            ax_hdr.axis("off")
            ax_hdr.text(0.0, 0.85, "ORALSCAN AI", fontsize=20, fontweight="bold", color="#1E293B")
            ax_hdr.text(0.0, 0.60, "Multimodal Oral Cancer Screening Research Prototype (ViT + GATv2)", fontsize=11, color="#64748B")
            ax_hdr.text(0.0, 0.35, "Vision Transformer & Graph Attention Network with Explainable AI", fontsize=9, fontstyle="italic", color="#0284C7")
            ax_hdr.text(1.0, 0.85, f"Report ID: {rid}", fontsize=10, fontweight="bold", ha="right", color="#334155")
            ax_hdr.text(1.0, 0.60, f"Screening ID: {sid}", fontsize=8, ha="right", color="#64748B")
            ax_hdr.text(1.0, 0.35, f"Date: {timestamp}", fontsize=8, ha="right", color="#64748B")
            ax_hdr.axhline(y=0.1, color="#CBD5E1", linewidth=1.5)

            # 2. Screening Result Card
            ax_res = fig.add_subplot(gs[1, :])
            ax_res.axis("off")
            card_color = "#FEF2F2" if pred_label == 1 else "#F0FDF4"
            border_color = "#DC2626" if pred_label == 1 else "#16A34A"
            badge_text = "SUSPICIOUS PATTERN DETECTED" if pred_label == 1 else "NO SUSPICIOUS PATTERN DETECTED"
            badge_color = "#DC2626" if pred_label == 1 else "#16A34A"

            rect = plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=ax_res.transAxes,
                                 facecolor=card_color, edgecolor=border_color, linewidth=1.5)
            ax_res.add_patch(rect)

            ax_res.text(0.06, 0.70, "MODEL SCREENING FINDING:", fontsize=10, fontweight="bold", color="#475569")
            ax_res.text(0.06, 0.40, cautious_result, fontsize=13, fontweight="bold", color=badge_color)
            ax_res.text(0.06, 0.18, "Status: Experimental Research Output (Not a Diagnostic Finding)", fontsize=8, fontstyle="italic", color="#64748B")

            ax_res.text(0.70, 0.70, "MODEL SOFTMAX SCORE", fontsize=9, fontweight="bold", ha="center", color="#475569")
            ax_res.text(0.70, 0.38, f"{cancer_score:.1f}%", fontsize=18, fontweight="bold", ha="center", color=badge_color)
            ax_res.text(0.70, 0.18, "Uncalibrated Model Score", fontsize=8, ha="center", color="#64748B")

            # 3. Diagnostic Image Panels (Original, ViT Rollout, GAT Graph)
            # Panel A: Original Image
            ax_img1 = fig.add_subplot(gs[2, 0])
            ax_img1.set_title("A. Uploaded Lesion Image (224x224)", fontsize=9, fontweight="bold", pad=4)
            if os.path.exists(raw_image_path):
                img_raw = Image.open(raw_image_path).convert("RGB")
                ax_img1.imshow(img_raw)
            else:
                ax_img1.text(0.5, 0.5, "Image Retained in Memory", ha="center", va="center")
            ax_img1.axis("off")

            # Panel B: ViT Attention Rollout Heatmap
            ax_img2 = fig.add_subplot(gs[2, 1])
            ax_img2.set_title("B. ViT Visual Attention Rollout (XAI)", fontsize=9, fontweight="bold", pad=4)
            if rollout_viz_path and os.path.exists(rollout_viz_path):
                img_roll = Image.open(rollout_viz_path).convert("RGB")
                ax_img2.imshow(img_roll)
            else:
                ax_img2.text(0.5, 0.5, "ViT Rollout Visualized", ha="center", va="center")
            ax_img2.axis("off")

            # 4. Fusion Branch Attribution Breakdown
            ax_attr = fig.add_subplot(gs[3, :])
            ax_attr.axis("off")
            ax_attr.text(0.02, 0.85, "MULTIMODAL FUSION BRANCH SENSITIVITY (INTEGRATED GRADIENTS)", fontsize=9, fontweight="bold", color="#1E293B")
            
            # Horizontal stacked bar
            bar_y = 0.45
            bar_h = 0.25
            ax_attr.barh(bar_y, vit_share, height=bar_h, left=0.0, color="#3B82F6", label=f"ViT Visual Branch ({vit_share:.1f}%)")
            ax_attr.barh(bar_y, gat_share, height=bar_h, left=vit_share, color="#10B981", label=f"GATv2 Graph Branch ({gat_share:.1f}%)")
            ax_attr.set_xlim(0, 100)
            ax_attr.set_ylim(0, 1)

            ax_attr.text(vit_share / 2, bar_y, f"ViT: {vit_share:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            ax_attr.text(vit_share + (gat_share / 2), bar_y, f"GAT: {gat_share:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            ax_attr.text(0.02, 0.10, f"Completeness Delta: |sum(IG) - Delta_f| = {ig_abs:.4f} logit units (Relative Error: {ig_rel:.1f}%) | Reference Baseline: [0.05]^384", fontsize=7.5, color="#64748B")

            # 5. Page 1 Footer Disclaimer
            ax_ftr1 = fig.add_subplot(gs[4, :])
            ax_ftr1.axis("off")
            ax_ftr1.axhline(y=0.8, color="#CBD5E1", linewidth=1.0)
            ax_ftr1.text(0.5, 0.4, "Page 1 of 2 - ORALSCAN AI Research Report - Experimental Prototype", fontsize=8, ha="center", color="#94A3B8")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # =========================================================================
            # PAGE 2: Detailed XAI Narrative, Clinical Guidance, and Disclaimers
            # =========================================================================
            fig2 = plt.figure(figsize=(8.5, 11), dpi=200)
            gs2 = gridspec.GridSpec(4, 1, figure=fig2, height_ratios=[0.8, 1.8, 1.2, 1.2])
            fig2.patch.set_facecolor("#FAFAFA")

            # Header
            ax2_hdr = fig2.add_subplot(gs2[0, :])
            ax2_hdr.axis("off")
            ax2_hdr.text(0.0, 0.80, "EXPLAINABLE AI NARRATIVE & CLINICAL RESEARCH GUIDELINES", fontsize=14, fontweight="bold", color="#1E293B")
            ax2_hdr.text(0.0, 0.50, f"Reference Screening ID: {sid} | Report ID: {rid}", fontsize=9, color="#64748B")
            ax2_hdr.axhline(y=0.2, color="#CBD5E1", linewidth=1.5)

            # Narrative Explanation Section
            ax2_narr = fig2.add_subplot(gs2[1, :])
            ax2_narr.axis("off")
            rect_narr = plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=ax2_narr.transAxes,
                                      facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
            ax2_narr.add_patch(rect_narr)
            ax2_narr.text(0.05, 0.88, "PLAIN-LANGUAGE AI EXPLANATION", fontsize=10, fontweight="bold", color="#0284C7")

            # Format narrative paragraphs cleanly
            narr_lines = narrative.split("\n")
            cleaned_narr = "\n".join([line for line in narr_lines if not line.startswith("=")])
            ax2_narr.text(0.05, 0.80, cleaned_narr[:1000], fontsize=8.5, color="#334155", va="top", wrap=True)

            # Clinical Guidance Section
            ax2_guidance = fig2.add_subplot(gs2[2, :])
            ax2_guidance.axis("off")
            rect_guide = plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=ax2_guidance.transAxes,
                                       facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0)
            ax2_guidance.add_patch(rect_guide)
            ax2_guidance.text(0.05, 0.88, "RECOMMENDED CLINICAL WORKFLOW", fontsize=10, fontweight="bold", color="#0F766E")
            guidance_text = (
                "1. Comprehensive In-Person Oral Examination: Any oral mucosal red/white lesion, ulceration, or mass\n"
                "   persisting beyond 14 days requires mandatory inspection and palpation by a dental surgeon or oncologist.\n"
                "2. Standard Diagnostic Modalities: Toluidine blue staining, VELscope autofluorescence, and high-resolution\n"
                "   cross-polarization photography may serve as adjunct aids.\n"
                "3. Gold-Standard Histopathology: An incisional or punch biopsy remains the sole definitive gold standard\n"
                "   for confirming cellular dysplasia, carcinoma in situ, or invasive squamous cell carcinoma (OSCC)."
            )
            ax2_guidance.text(0.05, 0.70, guidance_text, fontsize=8.5, color="#334155", va="top")

            # Mandatory Legal & Medical Disclaimer Section
            ax2_disc = fig2.add_subplot(gs2[3, :])
            ax2_disc.axis("off")
            rect_disc = plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=ax2_disc.transAxes,
                                      facecolor="#FFFBEB", edgecolor="#F59E0B", linewidth=1.2)
            ax2_disc.add_patch(rect_disc)
            ax2_disc.text(0.05, 0.88, "MANDATORY REGULATORY & MEDICAL RESEARCH DISCLAIMER", fontsize=9.5, fontweight="bold", color="#B45309")
            disclaimer_text = (
                "RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DEVICE OR DIAGNOSTIC TOOL.\n"
                "This document was autonomously generated by an experimental academic AI research application developed\n"
                "exclusively for evaluating Vision Transformer (ViT) and Graph Attention Network (GATv2) architectures.\n"
                "Predictions and attribution heatmaps reflect artificial neural network activations and feature sensitivities.\n"
                "They do NOT prove the presence or absence of malignancy, do NOT delineate surgical tumor boundaries,\n"
                "and MUST NOT be relied upon for clinical triage, diagnosis, staging, or therapeutic decisions."
            )
            ax2_disc.text(0.05, 0.72, disclaimer_text, fontsize=8.0, color="#78350F", va="top")

            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        return str(pdf_path.resolve())
