"""
Hybrid Spatial + Feature-Similarity Graph Construction
======================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Combines:
- 4-neighbour spatial grid adjacency edges.
- Cosine feature-similarity top-k edges.

Requirements:
- Set union of both edge types.
- Automatic removal of duplicate edges (binary adjacency clamp).
- Bidirectional (symmetric).
- Explicit self-loops.
- Independent from individual spatial and similarity graph modules.
"""

from typing import Tuple, Union
import torch
from src.graph.spatial_graph import build_spatial_adjacency_matrix
from src.graph.similarity_graph import build_similarity_adjacency_matrix


def build_hybrid_adjacency_matrix(
    features: torch.Tensor,
    grid_size: Tuple[int, int] = (14, 14),
    k: int = 8,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Constructs hybrid adjacency matrix for single image or batch.

    Args:
        features: Tensor of shape (196, 192) or (B, 196, 192).
        grid_size: (H, W) tuple, default (14, 14).
        k: Top-k neighbours for similarity graph, default 8.
        add_self_loops: Explicit self-loops, default True.

    Returns:
        adj_hybrid: Binary adjacency matrix of shape (196, 196) or (B, 196, 196).
    """
    is_batched = (features.dim() == 3)
    if not is_batched:
        features = features.unsqueeze(0)

    B, N, D = features.shape
    device = features.device

    # 1. Build spatial adjacency matrix [N, N] and expand to batch
    spatial_adj = build_spatial_adjacency_matrix(
        grid_size=grid_size,
        add_self_loops=add_self_loops,
        device=device
    ).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

    # 2. Build feature-similarity adjacency matrix (B, N, N)
    sim_adj = build_similarity_adjacency_matrix(
        features=features,
        k=k,
        add_self_loops=add_self_loops
    )

    # 3. Union of edge sets: clamp(spatial + similarity, 0.0, 1.0)
    # Deduplicates any shared edges between spatial grid and top-k similarity
    hybrid_adj = torch.clamp(spatial_adj + sim_adj, min=0.0, max=1.0)

    if not is_batched:
        return hybrid_adj.squeeze(0)
    return hybrid_adj


def build_hybrid_edge_index(
    features: torch.Tensor,
    grid_size: Tuple[int, int] = (14, 14),
    k: int = 8,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Construct edge_index [2, num_edges] for a single image graph.
    """
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    assert features.dim() == 2, "Edge index is returned for single image graphs"
    adj = build_hybrid_adjacency_matrix(features, grid_size=grid_size, k=k, add_self_loops=add_self_loops)
    edge_index = adj.nonzero(as_tuple=False).t().to(torch.long)
    return edge_index


if __name__ == "__main__":
    dummy_feats = torch.randn(2, 196, 192)
    adj_hybrid = build_hybrid_adjacency_matrix(dummy_feats, k=8)
    print("Batched Hybrid Adjacency Shape:", adj_hybrid.shape)
    assert adj_hybrid.shape == (2, 196, 196)
    assert torch.allclose(adj_hybrid, adj_hybrid.transpose(1, 2)), "Hybrid adjacency is not symmetric!"
    # Hybrid must have at least as many edges as spatial alone (924)
    edges_count = (adj_hybrid[0] > 0).sum().item()
    print("Hybrid Directed Edges per image:", edges_count)
    assert edges_count >= 924, f"Hybrid edges ({edges_count}) should be >= spatial edges (924)"
    print("Hybrid graph test PASSED!")
