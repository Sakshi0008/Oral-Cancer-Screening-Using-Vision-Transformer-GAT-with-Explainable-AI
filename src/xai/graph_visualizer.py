"""
Graph Overlay Visualizer Module
===============================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Renders high-quality visual overlays showing GATv2 influential relational edges
and node centrality superimposed on the oral cavity image.
Distinctly color-codes:
- Spatial Grid Edges (Cyan)
- Feature-Similarity Edges (Amber / Gold)
- Shared Hybrid Edges (Magenta)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from PIL import Image


class GraphOverlayVisualizer:
    """
    Renders topological GATv2 attention graph overlays on top of medical oral images.
    """
    def __init__(
        self,
        node_radius: float = 2.5,
        edge_alpha: float = 0.75,
        max_edge_width: float = 2.5
    ):
        self.node_radius = node_radius
        self.edge_alpha = edge_alpha
        self.max_edge_width = max_edge_width

    def render_graph_overlay(
        self,
        base_image: Image.Image,
        top_edges: List[Dict[str, Any]],
        node_centrality: np.ndarray,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders graph topology overlay on the 224x224 RGB image.

        Args:
            base_image: PIL Image of size 224x224.
            top_edges: List of edge dictionaries from GATv2GraphAttribution.
            node_centrality: 1D array of length 196 (node centrality).
            save_path: Optional filepath to save the rendered PNG figure.

        Returns:
            RGB numpy array of the rendered figure.
        """
        img_np = np.array(base_image.convert("RGB").resize((224, 224)))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(img_np)
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)  # Inverted y-axis for image coordinates
        ax.axis("off")

        # Color mapping for edge types
        edge_colors = {
            "spatial_grid": "#00e5ff",         # Vibrant Cyan
            "feature_similarity": "#ffab00",   # Vibrant Gold / Amber
            "shared_hybrid": "#d500f9",        # Neon Magenta
            "other": "#ffffff"
        }

        # Patch center coordinates in image space: (col * 16 + 8, row * 16 + 8)
        patch_coords = {}
        for idx in range(196):
            r = idx // 14
            c = idx % 14
            x = c * 16.0 + 8.0
            y = r * 16.0 + 8.0
            patch_coords[idx] = (x, y)

        # Normalize edge weights for width
        weights = [e["weight"] for e in top_edges]
        min_w, max_w = min(weights) if weights else 0.0, max(weights) if weights else 1.0

        # Draw edges
        for e in top_edges:
            if e.get("is_self_loop", False):
                continue
            src_idx = e["src_node"]
            dst_idx = e["dst_node"]
            etype = e.get("edge_type", "spatial_grid")
            color = edge_colors.get(etype, "#00e5ff")

            x0, y0 = patch_coords[src_idx]
            x1, y1 = patch_coords[dst_idx]

            # Scaled line width
            norm_w = (e["weight"] - min_w) / (max_w - min_w + 1e-9)
            lw = 0.8 + norm_w * (self.max_edge_width - 0.8)

            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    alpha=self.edge_alpha,
                    shrinkA=3,
                    shrinkB=3,
                    mutation_scale=8
                )
            )

        # Draw nodes
        ptp_val = float(node_centrality.max() - node_centrality.min())
        norm_centrality = (node_centrality - node_centrality.min()) / (ptp_val + 1e-9)
        cmap = plt.get_cmap("spring")

        for idx, (x, y) in patch_coords.items():
            c_val = norm_centrality[idx]
            # Only draw visible markers for top nodes or all with small radius
            if c_val > 0.4:
                node_color = cmap(c_val)
                circle = plt.Circle((x, y), self.node_radius * (1.0 + c_val), color=node_color, ec="black", lw=0.5, zorder=5)
                ax.add_patch(circle)

        # Custom Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#00e5ff", lw=2, label="Spatial Grid Edge (4-NN)"),
            Line2D([0], [0], color="#ffab00", lw=2, label="Similarity Edge (Cosine k=8)"),
            Line2D([0], [0], color="#d500f9", lw=2, label="Shared Hybrid Edge"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            fontsize=7,
            frameon=True,
            facecolor="white",
            edgecolor="gray"
        )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        # Convert to numpy array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3]
