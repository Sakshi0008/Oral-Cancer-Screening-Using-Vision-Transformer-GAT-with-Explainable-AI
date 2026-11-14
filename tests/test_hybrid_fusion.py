"""
Unit Tests for Hybrid Fusion Architectures
=========================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.getcwd())
from src.models.hybrid_fusion import (
    ConcatenationFusion,
    GatedFusion,
    ResidualFusion,
    ViTGATHybridModel
)
from src.models.handcrafted_fusion import (
    HandcraftedProjection,
    HandcraftedOnlyClassifier,
    BiModalHandcraftedFusion,
    TriModalConcatFusion,
    TriModalGatedFusion
)


class TestHybridFusionModels(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.vit_dim = 192
        self.graph_dim = 192
        self.hc_dim = 143

        self.vit_cls = torch.randn(self.B, self.vit_dim, requires_grad=True)
        self.graph_rep = torch.randn(self.B, self.graph_dim, requires_grad=True)
        self.hc_feat = torch.randn(self.B, self.hc_dim, requires_grad=True)

    def test_concat_fusion(self):
        model = ConcatenationFusion(vit_dim=self.vit_dim, graph_dim=self.graph_dim)
        logits, fused = model(self.vit_cls, self.graph_rep)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(fused.shape, (self.B, 384))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(self.vit_cls.grad)
        self.assertIsNotNone(self.graph_rep.grad)

    def test_gated_fusion(self):
        model = GatedFusion(vit_dim=self.vit_dim, graph_dim=self.graph_dim)
        logits, fused, gate = model(self.vit_cls, self.graph_rep)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(fused.shape, (self.B, 192))
        self.assertEqual(gate.shape, (self.B, 192))
        self.assertTrue((gate >= 0.0).all() and (gate <= 1.0).all())

    def test_residual_fusion(self):
        model = ResidualFusion(vit_dim=self.vit_dim, graph_dim=self.graph_dim)
        logits, fused = model(self.vit_cls, self.graph_rep)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(fused.shape, (self.B, 192))

    def test_handcrafted_only_classifier(self):
        model = HandcraftedOnlyClassifier(in_dim=self.hc_dim)
        logits = model(self.hc_feat)
        self.assertEqual(logits.shape, (self.B, 2))

    def test_tri_modal_concat(self):
        model = TriModalConcatFusion(vit_dim=self.vit_dim, graph_dim=self.graph_dim, hc_dim=self.hc_dim)
        logits, fused = model(self.vit_cls, self.graph_rep, self.hc_feat)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(fused.shape, (self.B, 448))

    def test_tri_modal_gated(self):
        model = TriModalGatedFusion(vit_dim=self.vit_dim, graph_dim=self.graph_dim, hc_dim=self.hc_dim)
        logits, fused, weights = model(self.vit_cls, self.graph_rep, self.hc_feat)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(fused.shape, (self.B, 128))
        self.assertEqual(weights.shape, (self.B, 3))
        # Softmax weights should sum to 1.0 per sample
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(self.B))

    def test_batch_size_one(self):
        v = torch.randn(1, self.vit_dim)
        g = torch.randn(1, self.graph_dim)
        h = torch.randn(1, self.hc_dim)

        model = TriModalConcatFusion()
        model.eval()
        with torch.no_grad():
            out, _ = model(v, g, h)
        self.assertEqual(out.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
