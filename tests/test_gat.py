"""
Unit Tests for Graph Attention Network (GAT) Architecture
=========================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.graph_attention import build_gat, GAT
from src.models.graph_classifier import build_graph_classifier


class TestGAT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.N = 196
        self.D = 192
        self.dummy_x = torch.randn(self.B, self.N, self.D)
        # Sparse test adjacency: eye + one neighbor
        self.dummy_adj = torch.eye(self.N).unsqueeze(0).expand(self.B, -1, -1).clone()
        self.dummy_adj[:, 0, 1] = 1.0
        self.dummy_adj[:, 1, 0] = 1.0

    def test_gat_forward_and_attention_shapes(self):
        gat = build_gat(in_features=192, hidden_features_per_head=48, num_heads=4, num_layers=2)
        out, attns = gat(self.dummy_x, self.dummy_adj)

        self.assertEqual(out.shape, (self.B, self.N, self.D))
        self.assertEqual(len(attns), 2)
        self.assertEqual(attns[0].shape, (self.B, 4, self.N, self.N))
        self.assertEqual(attns[1].shape, (self.B, 4, self.N, self.N))

    def test_gat_attention_masking(self):
        gat = build_gat(in_features=192, hidden_features_per_head=48, num_heads=4, num_layers=1)
        _, attns = gat(self.dummy_x, self.dummy_adj)
        attn = attns[0]  # (B, H, N, N)

        # For node 0, edges only exist to node 0 and node 1.
        # Attention to node 2 must be 0.0
        for b in range(self.B):
            for h in range(4):
                self.assertAlmostEqual(attn[b, h, 0, 2].item(), 0.0, places=5)
                # Attention to self (node 0) and neighbour (node 1) must be positive
                self.assertGreater(attn[b, h, 0, 0].item(), 0.0)
                self.assertGreater(attn[b, h, 0, 1].item(), 0.0)

    def test_gat_gradient_flow(self):
        gat = build_gat()
        out, _ = gat(self.dummy_x, self.dummy_adj)
        loss = out.sum()
        loss.backward()

        for i, layer in enumerate(gat.layers):
            self.assertIsNotNone(layer.fc.weight.grad, f"Layer {i} FC grad missing")
            self.assertIsNotNone(layer.attn_src.grad, f"Layer {i} attn_src grad missing")
            self.assertIsNotNone(layer.attn_dst.grad, f"Layer {i} attn_dst grad missing")
            self.assertGreater(layer.fc.weight.grad.abs().sum().item(), 0.0)

    def test_gat_classifier_pipeline(self):
        clf = build_graph_classifier(graph_type="spatial", gnn_type="gat", pooling="mean")
        logits = clf(self.dummy_x)
        self.assertEqual(logits.shape, (self.B, 2))
        graph_rep = clf.get_graph_representation(self.dummy_x)
        self.assertEqual(graph_rep.shape, (self.B, 192))


if __name__ == "__main__":
    unittest.main()
