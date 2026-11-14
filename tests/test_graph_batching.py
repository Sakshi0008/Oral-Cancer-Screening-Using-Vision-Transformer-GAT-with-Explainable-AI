"""
Unit Tests for Graph Batching and Cross-Image Isolation
======================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graph.graph_builder import build_batched_edge_index, build_image_graph


class TestGraphBatching(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 3
        self.N = 196
        self.D = 192
        self.dummy_features = torch.randn(self.B, self.N, self.D)

    def test_batched_edge_index_isolation(self):
        for strategy in ["spatial", "similarity", "hybrid"]:
            x_flat, edge_index, batch = build_batched_edge_index(
                strategy=strategy,
                features=self.dummy_features,
                k=8
            )
            # Verify shapes
            self.assertEqual(x_flat.shape, (self.B * self.N, self.D))
            self.assertEqual(batch.shape, (self.B * self.N,))

            # Verify no cross-image edges exist
            # For every edge (u, v), floor(u / 196) MUST equal floor(v / 196)
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            src_graph = torch.div(src_nodes, self.N, rounding_mode="floor")
            dst_graph = torch.div(dst_nodes, self.N, rounding_mode="floor")

            self.assertTrue(
                torch.all(src_graph == dst_graph),
                f"Cross-image edge detected in strategy '{strategy}'!"
            )

    def test_dense_adjacency_batch_dimensions(self):
        for strategy in ["spatial", "similarity", "hybrid"]:
            adj = build_image_graph(strategy, self.dummy_features, k=8)
            self.assertEqual(adj.shape, (self.B, self.N, self.N))
            for b in range(self.B):
                # Diagonal should be all 1.0 (self-loops)
                diag = torch.diagonal(adj[b])
                self.assertTrue(torch.all(diag == 1.0))


if __name__ == "__main__":
    unittest.main()
