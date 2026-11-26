"""
GATv2 Graph Attention and Attribution Module
============================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts multi-head dynamic edge-attention coefficients from both GATv2 layers,
computes patch-level node centrality, classifies edges into spatial vs similarity types,
and performs edge/node perturbation validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.graph_builder import build_image_graph
from src.graph.spatial_graph import build_spatial_adjacency_matrix
from src.graph.similarity_graph import build_similarity_adjacency_matrix


class GATv2GraphAttribution:
    """
    Computes graph-level interpretability for the GATv2 relational branch.
    """
    def __init__(
        self,
        gat_model: nn.Module,
        include_self_loops: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            gat_model: Instantiated GraphClassifier model.
            include_self_loops: Whether to include self-loop weights in node centrality (default False).
            device: Compute device.
        """
        self.gat = gat_model
        self.include_self_loops = include_self_loops
        self.device = device or next(self.gat.parameters()).device

    def extract_graph_attention(
        self,
        patch_tokens: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        top_k_edges: int = 30
    ) -> Dict[str, Any]:
        """
        Runs forward pass through GATv2 and extracts layer-wise, head-wise attention matrices.

        Args:
            patch_tokens: ViT patch embeddings of shape (1, 196, 192).
            adj: Optional hybrid adjacency matrix (1, 196, 196). If None, dynamically built.
            top_k_edges: Number of highest-weight directed edges to extract for visualization.

        Returns:
            Dictionary containing:
                - 'raw_attentions': [Layer1 (1, 4, 196, 196), Layer2 (1, 4, 196, 196)]
                - 'mean_layer2_attn': Attention weights averaged over 4 heads (196, 196)
                - 'node_contribution_14x14': 2D node incoming attention grid [14, 14] in [0, 1]
                - 'node_heatmap_224x224': Upsampled continuous node importance heatmap [224, 224]
                - 'top_edges': List of dicts describing top edges with spatial vs similarity type
                - 'self_loops_included': Boolean flag
        """
        self.gat.eval()
        tokens = patch_tokens.to(self.device)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        # 1. Build adjacency matrices to decompose edge types
        spatial_adj = build_spatial_adjacency_matrix(grid_size=(14, 14), add_self_loops=True, device=self.device).unsqueeze(0)
        sim_adj = build_similarity_adjacency_matrix(features=tokens, k=8, add_self_loops=True)
        if adj is None:
            adj = torch.clamp(spatial_adj + sim_adj, min=0.0, max=1.0)

        # 2. Extract forward pass outputs and attention tensors
        with torch.no_grad():
            full_out = self.gat.forward_full(tokens, adj)
            attns = full_out["attns"]  # List of [Layer1, Layer2], each (1, 4, 196, 196)
            graph_rep = full_out["graph_rep"]

        # Preserve head dimension
        layer1_attns = attns[0][0]  # (4, 196, 196)
        layer2_attns = attns[1][0]  # (4, 196, 196)

        # 3. Aggregate across attention heads (mean over H=4)
        l2_mean = layer2_attns.mean(dim=0).cpu().numpy()  # (196, 196)

        # 4. Compute Node Incoming Attention (Centrality)
        # Node i centrality: how strongly other nodes attend to node i as an information source
        if not self.include_self_loops:
            diag_mask = np.eye(196, dtype=bool)
            l2_no_self = l2_mean.copy()
            l2_no_self[diag_mask] = 0.0
            node_centrality = l2_no_self.sum(axis=0)  # incoming attention to node j
        else:
            node_centrality = l2_mean.sum(axis=0)

        # Reshape to 14x14 grid
        node_grid = node_centrality.reshape(14, 14)
        grid_min, grid_max = node_grid.min(), node_grid.max()
        norm_node_grid = (node_grid - grid_min) / (grid_max - grid_min + 1e-9)

        # Upsample to 224x224
        grid_t = torch.tensor(norm_node_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        node_heatmap = F.interpolate(grid_t, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()

        # 5. Classify and Extract Top-K Directed Edges
        # adj[b, i, j] = 1 indicates message passing from j -> i
        sp_mat = spatial_adj[0].cpu().numpy()
        sm_mat = sim_adj[0].cpu().numpy()

        edge_list = []
        for i in range(196):
            for j in range(196):
                if i == j and not self.include_self_loops:
                    continue
                w = float(l2_mean[i, j])
                if w > 0.0:
                    is_spatial = bool(sp_mat[i, j] > 0.5)
                    is_similarity = bool(sm_mat[i, j] > 0.5)

                    if is_spatial and is_similarity:
                        edge_type = "shared_hybrid"
                    elif is_spatial:
                        edge_type = "spatial_grid"
                    elif is_similarity:
                        edge_type = "feature_similarity"
                    else:
                        edge_type = "other"

                    r_src, c_src = j // 14, j % 14
                    r_dst, c_dst = i // 14, i % 14

                    edge_list.append({
                        "src_node": int(j),
                        "dst_node": int(i),
                        "src_coord": (int(r_src), int(c_src)),
                        "dst_coord": (int(r_dst), int(c_dst)),
                        "weight": w,
                        "edge_type": edge_type,
                        "is_self_loop": (i == j)
                    })

        edge_list.sort(key=lambda x: x["weight"], reverse=True)
        top_edges = edge_list[:top_k_edges]

        return {
            "raw_attentions": [a.cpu() for a in attns],
            "layer1_heads": layer1_attns.cpu(),
            "layer2_heads": layer2_attns.cpu(),
            "mean_layer2_attn": l2_mean,
            "node_centrality_vector": node_centrality,
            "node_contribution_14x14": norm_node_grid,
            "node_heatmap_224x224": node_heatmap,
            "top_edges": top_edges,
            "self_loops_included": self.include_self_loops,
            "graph_rep": graph_rep.cpu()
        }

    def validate_edge_perturbation(
        self,
        patch_tokens: torch.Tensor,
        adj: torch.Tensor,
        fusion_model: nn.Module,
        vit_cls: torch.Tensor,
        edges_to_remove: List[Dict[str, Any]],
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Validates edge attribution via targeted edge removal:
        Zeroes out the specified influential edges and measures target logit drop.
        """
        self.gat.eval()
        fusion_model.eval()

        tokens = patch_tokens.to(self.device)
        cls_t = vit_cls.to(self.device)
        adj_clean = adj.to(self.device).clone()

        with torch.no_grad():
            clean_rep = self.gat.get_graph_representation(tokens, adj_clean)
            clean_logits, _ = fusion_model(cls_t, clean_rep)
            clean_logit = float(clean_logits[0, target_class].item())

        # Perturbed adjacency: zero out specified edges
        adj_perturbed = adj_clean.clone()
        for e in edges_to_remove:
            i, j = e["dst_node"], e["src_node"]
            adj_perturbed[0, i, j] = 0.0

        with torch.no_grad():
            pert_rep = self.gat.get_graph_representation(tokens, adj_perturbed)
            pert_logits, _ = fusion_model(cls_t, pert_rep)
            pert_logit = float(pert_logits[0, target_class].item())

        logit_drop = clean_logit - pert_logit
        return {
            "clean_logit": clean_logit,
            "perturbed_logit": pert_logit,
            "logit_drop": logit_drop,
            "edges_removed_count": len(edges_to_remove)
        }
