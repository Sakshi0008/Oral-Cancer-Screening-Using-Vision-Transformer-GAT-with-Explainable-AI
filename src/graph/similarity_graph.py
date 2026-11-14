"""
Feature-Similarity Graph Construction
======================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Constructs a graph based on cosine similarity between ViT patch-token features.
Requirements:
- L2-normalize patch features prior to cosine similarity.
- Calculate pairwise cosine similarities within each individual image.
- Exclude self-similarity during top-k neighbour selection.
- Connect each node to its top-k most similar nodes (default k=8, configurable).
- Make the graph bidirectional (symmetric adjacency).
- Explicitly add self-loops.
- Strictly no cross-image edges.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F


def build_similarity_adjacency_matrix(
    features: torch.Tensor,
    k: int = 8,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Constructs feature-similarity adjacency matrix for single image or batch.

    Args:
        features: Tensor of shape (196, 192) or (B, 196, 192).
        k: Number of top nearest neighbours based on cosine similarity (default: 8).
        add_self_loops: Whether to explicitly add self-loops (default: True).

    Returns:
        adj: Binary adjacency matrix of shape (196, 196) or (B, 196, 196).
    """
    is_batched = (features.dim() == 3)
    if not is_batched:
        features = features.unsqueeze(0)  # (1, 196, 192)

    B, N, D = features.shape
    device = features.device

    # 1. L2 normalize patch features along channel dimension
    norm_feats = F.normalize(features, p=2, dim=-1)  # (B, N, D)

    # 2. Pairwise cosine similarity: (B, N, D) @ (B, D, N) -> (B, N, N)
    sim_matrix = torch.bmm(norm_feats, norm_feats.transpose(1, 2))

    # 3. Exclude self-connections from top-k selection by setting diagonal to -infinity
    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    sim_matrix_no_self = sim_matrix.masked_fill(eye_mask, -1e9)

    # 4. Top-k nearest neighbours per node
    # top_indices: (B, N, k)
    _, top_indices = torch.topk(sim_matrix_no_self, k=k, dim=-1, largest=True, sorted=False)

    # 5. Build symmetric (bidirectional) adjacency matrix
    adj = torch.zeros((B, N, N), dtype=torch.float32, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(0).unsqueeze(-1).expand(B, N, k)
    adj.scatter_(dim=-1, index=top_indices, value=1.0)

    # Symmetrize: if i->j then j->i (bidirectional)
    adj = torch.clamp(adj + adj.transpose(1, 2), min=0.0, max=1.0)

    # 6. Explicitly add self-loops
    if add_self_loops:
        adj = adj.masked_fill(eye_mask, 1.0)

    if not is_batched:
        return adj.squeeze(0)
    return adj


def build_similarity_edge_index(
    features: torch.Tensor,
    k: int = 8,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Construct edge_index [2, num_edges] for a single image graph.
    """
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    assert features.dim() == 2, "Edge index is returned for single image graphs"
    adj = build_similarity_adjacency_matrix(features, k=k, add_self_loops=add_self_loops)
    edge_index = adj.nonzero(as_tuple=False).t().to(torch.long)
    return edge_index


if __name__ == "__main__":
    dummy_feats = torch.randn(2, 196, 192)
    adj = build_similarity_adjacency_matrix(dummy_feats, k=8)
    print("Batched Similarity Adjacency Shape:", adj.shape)
    assert adj.shape == (2, 196, 196)
    # Check diagonal is 1.0 (self-loops)
    diag = torch.diagonal(adj[0])
    assert torch.all(diag == 1.0), "Self-loops missing!"
    # Check symmetry
    assert torch.allclose(adj, adj.transpose(1, 2)), "Adjacency is not symmetric!"
    print("Similarity graph test PASSED!")
