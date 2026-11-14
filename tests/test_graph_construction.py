"""
Unit Tests for Graph Construction Modules (Spatial, Similarity, Hybrid)
========================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graph.spatial_graph import build_spatial_adjacency_matrix, build_spatial_edge_index
from src.graph.similarity_graph import build_similarity_adjacency_matrix
from src.graph.hybrid_graph import build_hybrid_adjacency_matrix


class TestGraphConstruction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dummy_features = torch.randn(2, 196, 192)

    def test_spatial_graph_structure(self):
        adj = build_spatial_adjacency_matrix(grid_size=(14, 14), add_self_loops=True)
        self.assertEqual(adj.shape, (196, 196))

        # Check symmetry
        self.assertTrue(torch.allclose(adj, adj.t()))

        # Check exact edge count: 924
        edge_index = build_spatial_edge_index(grid_size=(14, 14), add_self_loops=True)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], 924)

        # Check degree of corner nodes (e.g. node 0: (0,0) -> neighbors: (0,1), (1,0) + self = 3)
        corner_degree = adj[0].sum().item()
        self.assertEqual(corner_degree, 3.0)

        # Check degree of interior nodes (e.g. node 15: (1,1) -> 4 neighbors + self = 5)
        interior_degree = adj[15].sum().item()
        self.assertEqual(interior_degree, 5.0)

        # Check degree of edge nodes (e.g. node 1: (0,1) -> 3 neighbors + self = 4)
        edge_degree = adj[1].sum().item()
        self.assertEqual(edge_degree, 4.0)

    def test_similarity_graph_structure(self):
        k = 8
        adj = build_similarity_adjacency_matrix(self.dummy_features, k=k, add_self_loops=True)
        self.assertEqual(adj.shape, (2, 196, 196))

        for b in range(2):
            sub_adj = adj[b]
            # Check symmetry (bidirectional)
            self.assertTrue(torch.allclose(sub_adj, sub_adj.t()))

            # Check self-loops exist
            diag = torch.diagonal(sub_adj)
            self.assertTrue(torch.all(diag == 1.0))

            # Each node must have degree at least k + 1 (its top-k plus self-loop, plus any incoming top-k)
            degrees = sub_adj.sum(dim=-1)
            self.assertTrue(torch.all(degrees >= (k + 1)))

    def test_hybrid_graph_structure(self):
        k = 8
        adj_spatial = build_spatial_adjacency_matrix(grid_size=(14, 14), add_self_loops=True)
        adj_hybrid = build_hybrid_adjacency_matrix(self.dummy_features, k=k, add_self_loops=True)

        self.assertEqual(adj_hybrid.shape, (2, 196, 196))

        for b in range(2):
            sub_adj = adj_hybrid[b]
            # Check symmetry
            self.assertTrue(torch.allclose(sub_adj, sub_adj.t()))

            # Check self-loops
            self.assertTrue(torch.all(torch.diagonal(sub_adj) == 1.0))

            # Hybrid must contain ALL spatial edges: wherever spatial is 1, hybrid must be 1
            spatial_mask = (adj_spatial > 0)
            self.assertTrue(torch.all(sub_adj[spatial_mask] == 1.0))

            # Total directed edges must be >= 924
            total_edges = sub_adj.sum().item()
            self.assertGreater(total_edges, 924.0)

    def test_determinism(self):
        adj1 = build_similarity_adjacency_matrix(self.dummy_features, k=8)
        adj2 = build_similarity_adjacency_matrix(self.dummy_features, k=8)
        self.assertTrue(torch.equal(adj1, adj2))


if __name__ == "__main__":
    unittest.main()
