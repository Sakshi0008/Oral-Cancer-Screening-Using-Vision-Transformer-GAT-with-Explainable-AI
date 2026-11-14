"""
Spatial Adjacency Graph Construction
====================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Constructs a 4-neighbour spatial grid graph for 196 Vision Transformer patches (14x14 arrangement).
- Each patch is a node (0 to 195).
- 4-neighbour connectivity: (r-1, c), (r+1, c), (r, c-1), (r, c+1).
- Bidirectional edges.
- Explicit self-loops: (i, i) for all i.
- Correct boundary handling (corners have 2+1=3 edges, borders have 3+1=4 edges, interior has 4+1=5 edges).
- Total directed edges per image: 924.
"""

from typing import Tuple
import torch


def build_spatial_adjacency_matrix(
    grid_size: Tuple[int, int] = (14, 14),
    add_self_loops: bool = True,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Construct dense binary spatial adjacency matrix [N, N] for grid_size (H, W).
    Default: (14, 14) -> 196 nodes.
    """
    H, W = grid_size
    N = H * W
    adj = torch.zeros((N, N), dtype=torch.float32, device=device)

    for r in range(H):
        for c in range(W):
            node_idx = r * W + c

            # 4-neighbours
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * W + c)  # Up
            if r < H - 1:
                neighbors.append((r + 1) * W + c)  # Down
            if c > 0:
                neighbors.append(r * W + (c - 1))  # Left
            if c < W - 1:
                neighbors.append(r * W + (c + 1))  # Right

            for nbr in neighbors:
                adj[node_idx, nbr] = 1.0
                adj[nbr, node_idx] = 1.0  # Guarantee bidirectionality

            if add_self_loops:
                adj[node_idx, node_idx] = 1.0

    return adj


def build_spatial_edge_index(
    grid_size: Tuple[int, int] = (14, 14),
    add_self_loops: bool = True,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Construct edge_index of shape [2, num_edges] for the spatial graph.
    """
    adj = build_spatial_adjacency_matrix(grid_size, add_self_loops, device)
    edge_index = adj.nonzero(as_tuple=False).t()  # [2, num_edges]
    return edge_index.to(torch.long)


if __name__ == "__main__":
    adj = build_spatial_adjacency_matrix()
    edge_index = build_spatial_edge_index()
    print("Spatial Adjacency Matrix Shape:", adj.shape)
    print("Total Directed Edges (with self-loops):", edge_index.shape[1], "(Expected: 924)")
    assert adj.shape == (196, 196)
    assert edge_index.shape[1] == 924
    print("Spatial graph test PASSED!")
