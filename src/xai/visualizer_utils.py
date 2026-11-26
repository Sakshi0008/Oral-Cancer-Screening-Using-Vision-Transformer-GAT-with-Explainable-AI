"""
Multimodal XAI Visualization Utilities
======================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Assembles standardized, publication-quality 5-panel XAI diagnostic figures:
- Panel 1: Original Clinical Image
- Panel 2: ViT Attention Rollout Heatmap Overlay
- Panel 3: GATv2 Node Centrality Heatmap Overlay
- Panel 4: GATv2 Topological Graph Overlay (Spatial vs Similarity Edges)
- Panel 5: Fusion-Level Integrated Gradients Branch Breakdown
"""

from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


class MultimodalXAIVisualizer:
    """
    Renders standardized multi-panel XAI diagnostic figures.
    """
    @staticmethod
    def render_comprehensive_figure(
        original_img: Image.Image,
        vit_heatmap_224: np.ndarray,
        gat_heatmap_224: np.ndarray,
        graph_overlay_rgb: np.ndarray,
        fusion_attr: Dict[str, Any],
        pred_label: int,
        pred_name: str,
        cancer_prob: float,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders a comprehensive 5-panel explanatory figure.

        Returns:
            RGB numpy array of the rendered multi-panel figure.
        """
        img_np = np.array(original_img.convert("RGB").resize((224, 224)))

        fig = plt.figure(figsize=(18, 5), dpi=150)
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.2])

        # Color for title based on prediction
        title_color = "#d32f2f" if pred_label == 1 else "#388e3c"

        # Panel 1: Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_np)
        ax1.set_title(f"Input Oral Image\n({pred_name}: {cancer_prob*100:.1f}%)", fontsize=10, fontweight="bold", color=title_color)
        ax1.axis("off")

        # Panel 2: ViT Attention Rollout Overlay
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_np)
        im2 = ax2.imshow(vit_heatmap_224, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        ax2.set_title("ViT Attention Rollout\n(Visual Patch Importance)", fontsize=10, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 3: GATv2 Node Contribution Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img_np)
        im3 = ax3.imshow(gat_heatmap_224, cmap="magma", alpha=0.5, vmin=0.0, vmax=1.0)
        ax3.set_title("GATv2 Node Centrality\n(Graph Relational Inflow)", fontsize=10, fontweight="bold")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Panel 4: GATv2 Graph Overlay
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(graph_overlay_rgb)
        ax4.set_title("GATv2 Graph Structure\n(Top Spatial vs Similarity Edges)", fontsize=10, fontweight="bold")
        ax4.axis("off")

        # Panel 5: Fusion Integrated Gradients Breakdown
        ax5 = fig.add_subplot(gs[0, 4])
        vit_pct = fusion_attr.get("vit_share_pct", 50.0)
        gat_pct = fusion_attr.get("gat_share_pct", 50.0)
        vit_signed = fusion_attr.get("vit_signed_sum", 0.0)
        gat_signed = fusion_attr.get("gat_signed_sum", 0.0)

        branches = ["ViT Branch\n(CLS Visual)", "GATv2 Branch\n(Graph Topology)"]
        shares = [vit_pct, gat_pct]
        bar_colors = ["#1976d2", "#7b1fa2"]

        bars = ax5.bar(branches, shares, color=bar_colors, width=0.5, edgecolor="black", linewidth=1.0)
        ax5.set_ylabel("Attribution Share (%)", fontsize=9, fontweight="bold")
        ax5.set_ylim(0, 100)
        ax5.set_title("Fusion Branch Attribution\n(Integrated Gradients)", fontsize=10, fontweight="bold")
        ax5.grid(axis="y", linestyle="--", alpha=0.6)

        # Annotate percentages and signed evidence on bars
        for b, bar, signed in zip(bars, bars, [vit_signed, gat_signed]):
            h = bar.get_height()
            sign_text = f"+{signed:.2f}" if signed >= 0 else f"{signed:.2f}"
            ax5.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%\n(net: {sign_text})",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3]
