"""
Unified Image Graph Construction & Batching API
================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Provides a unified interface to construct image graphs using:
1. "spatial": 4-neighbour spatial grid graph.
2. "similarity": Cosine similarity top-k graph.
3. "hybrid": Union of spatial grid and similarity edges.

Guarantees:
- Strict image isolation (no edges between different images).
- Deterministic construction.
- Consistent node ordering (0 to 195).
- Compatible with pure PyTorch GAT and GATv2 architectures.
"""

from typing import Dict, Optional, Tuple, Union
import torch
from src.graph.spatial_graph import build_spatial_adjacency_matrix
from src.graph.similarity_graph import build_similarity_adjacency_matrix
from src.graph.hybrid_graph import build_hybrid_adjacency_matrix


def build_image_graph(
    strategy: str,
    features: torch.Tensor,
    k: int = 8,
    add_self_loops: bool = True,
    grid_size: Tuple[int, int] = (14, 14)
) -> torch.Tensor:
    """
    Constructs adjacency matrix for the requested strategy.

    Args:
        strategy: "spatial", "similarity", or "hybrid".
        features: Patch token tensor of shape (B, 196, 192) or (196, 192).
        k: Top-k neighbours for similarity or hybrid strategies (default 8).
        add_self_loops: Whether to explicitly include self-loops (default True).
        grid_size: (H, W) spatial patch grid, default (14, 14).

    Returns:
        adj: Binary adjacency matrix of shape (B, 196, 196) or (196, 196).
    """
    strat = strategy.lower().strip()
    is_batched = (features.dim() == 3)
    B = features.shape[0] if is_batched else 1

    if strat == "spatial":
        base_adj = build_spatial_adjacency_matrix(
            grid_size=grid_size,
            add_self_loops=add_self_loops,
            device=features.device
        )
        if is_batched:
            return base_adj.unsqueeze(0).expand(B, -1, -1)
        return base_adj

    elif strat in ("similarity", "feature_similarity"):
        return build_similarity_adjacency_matrix(
            features=features,
            k=k,
            add_self_loops=add_self_loops
        )

    elif strat == "hybrid":
        return build_hybrid_adjacency_matrix(
            features=features,
            grid_size=grid_size,
            k=k,
            add_self_loops=add_self_loops
        )
    else:
        raise ValueError(f"Unknown graph strategy '{strategy}'. Supported: 'spatial', 'similarity', 'hybrid'.")


def build_batched_edge_index(
    strategy: str,
    features: torch.Tensor,
    k: int = 8,
    add_self_loops: bool = True,
    grid_size: Tuple[int, int] = (14, 14)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct batched sparse edge_index with node offsets ensuring zero cross-image edges.

    Args:
        features: (B, 196, 192) tensor.
    Returns:
        x_flat: Flattened node features (B * 196, 192).
        edge_index: (2, total_edges) tensor with node index offsets.
        batch: (B * 196,) tensor assigning each node to its graph/image index.
    """
    assert features.dim() == 3, "Batched edge index expects (B, 196, 192)"
    B, N, D = features.shape
    device = features.device

    adj_batched = build_image_graph(strategy, features, k=k, add_self_loops=add_self_loops, grid_size=grid_size)

    x_flat = features.reshape(B * N, D)
    batch = torch.arange(B, device=device).repeat_interleave(N)

    edge_list = []
    for b in range(B):
        sub_adj = adj_batched[b]
        sub_edges = sub_adj.nonzero(as_tuple=False).t()  # (2, E_b)
        offset = b * N
        edge_list.append(sub_edges + offset)

    edge_index = torch.cat(edge_list, dim=1).to(torch.long)
    return x_flat, edge_index, batch


if __name__ == "__main__":
    dummy_feats = torch.randn(3, 196, 192)
    for strat in ["spatial", "similarity", "hybrid"]:
        adj = build_image_graph(strat, dummy_feats, k=8)
        print(f"Strategy '{strat}': Adjacency shape = {adj.shape}, Edges per graph = {(adj[0] > 0).sum().item()}")
        assert adj.shape == (3, 196, 196)

    x_flat, edge_index, batch = build_batched_edge_index("hybrid", dummy_feats)
    print("Batched x_flat shape:", x_flat.shape, "edge_index shape:", edge_index.shape, "batch shape:", batch.shape)
    assert x_flat.shape == (3 * 196, 192)
    assert batch.shape == (3 * 196,)
    print("Graph builder verification PASSED!")
