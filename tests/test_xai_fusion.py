"""
Unit Tests for Fusion-Level Integrated Gradients Attribution
============================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.hybrid_fusion import ConcatenationFusion
from src.xai.fusion_attribution import FusionIntegratedGradients


class TestFusionXAI(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.fusion = ConcatenationFusion(vit_dim=192, graph_dim=192, hidden_dim=128, num_classes=2).to(self.device)
        self.fusion.eval()

        self.dummy_cls = torch.randn(1, 192, device=self.device)
        self.dummy_graph = torch.randn(1, 192, device=self.device)
        self.dummy_baseline = torch.randn(1, 384, device=self.device) * 0.1

    def test_integrated_gradients_dimensions_and_completeness(self):
        ig_explainer = FusionIntegratedGradients(self.fusion, steps=100, device=self.device)
        res = ig_explainer.attribute(self.dummy_cls, self.dummy_graph, target_class=1, baseline=self.dummy_baseline)

        # Vector dimensions
        self.assertEqual(len(res["ig_vector"]), 384)
        self.assertEqual(len(res["vit_ig"]), 192)
        self.assertEqual(len(res["gat_ig"]), 192)

        # Percentage shares sum to 100%
        vit_pct = res["vit_share_pct"]
        gat_pct = res["gat_share_pct"]
        self.assertAlmostEqual(vit_pct + gat_pct, 100.0, places=4)
        self.assertGreaterEqual(vit_pct, 0.0)
        self.assertGreaterEqual(gat_pct, 0.0)

        # Completeness Check (Axiom of Completeness)
        # sum(IG) should be within 2% of delta_f = F(z) - F(baseline)
        error_pct = res["convergence_error_pct"]
        self.assertLess(error_pct, 2.0, f"Integrated Gradients completeness error ({error_pct:.2f}%) exceeds 2%")

    def test_signed_attributions(self):
        ig_explainer = FusionIntegratedGradients(self.fusion, steps=50, device=self.device)
        res = ig_explainer.attribute(self.dummy_cls, self.dummy_graph, target_class=1)

        self.assertIn("vit_signed_sum", res)
        self.assertIn("gat_signed_sum", res)
        self.assertFalse(np.isnan(res["vit_signed_sum"]))
        self.assertFalse(np.isnan(res["gat_signed_sum"]))


if __name__ == "__main__":
    unittest.main()
