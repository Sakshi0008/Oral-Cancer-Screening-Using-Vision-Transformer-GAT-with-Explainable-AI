"""
Unit Tests for Multi-Scale Fusion Models
========================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.getcwd())
from src.models.multiscale_fusion import (
    MultiScalePatchPooler,
    MultiScaleViTClassifier,
    MultiScaleHybridClassifier
)


class TestMultiScaleFusion(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.N = 196
        self.D = 192
        self.patch_tokens = torch.randn(self.B, self.N, self.D, requires_grad=True)
        self.vit_cls = torch.randn(self.B, self.D, requires_grad=True)
        self.graph_rep = torch.randn(self.B, self.D, requires_grad=True)

    def test_multiscale_patch_pooler(self):
        pooler = MultiScalePatchPooler(embed_dim=self.D, grid_size=14)
        h_fine, h_region = pooler(self.patch_tokens)
        self.assertEqual(h_fine.shape, (self.B, self.D))
        self.assertEqual(h_region.shape, (self.B, self.D))

    def test_multiscale_vit_classifier(self):
        model = MultiScaleViTClassifier(embed_dim=self.D)
        logits, multiscale = model(self.patch_tokens, self.vit_cls)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(multiscale.shape, (self.B, 576))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(self.patch_tokens.grad)
        self.assertIsNotNone(self.vit_cls.grad)

    def test_multiscale_hybrid_classifier(self):
        model = MultiScaleHybridClassifier(embed_dim=self.D, graph_dim=self.D)
        logits, combined = model(self.patch_tokens, self.vit_cls, self.graph_rep)
        self.assertEqual(logits.shape, (self.B, 2))
        self.assertEqual(combined.shape, (self.B, 768))


if __name__ == "__main__":
    unittest.main()
