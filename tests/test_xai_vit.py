"""
Unit Tests for ViT Attention Rollout and Patch Attribution
==========================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.vit_scratch import build_vit_scratch
from src.xai.vit_rollout import ViTAttentionRollout
from src.xai.vit_attribution import ViTPatchAttribution


class TestViTXAI(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = build_vit_scratch(
            img_size=224, patch_size=16, embed_dim=192, depth=4, num_heads=3, mlp_dim=768, num_classes=2
        ).to(self.device)
        self.model.eval()
        self.dummy_input = torch.randn(1, 3, 224, 224, device=self.device)

    def test_rollout_shapes_and_ranges(self):
        rollout = ViTAttentionRollout(self.model)
        res = rollout.compute_rollout(self.dummy_input, device=self.device)

        self.assertEqual(len(res["raw_attentions"]), 4, "Should capture attention from all 4 blocks")
        for attn in res["raw_attentions"]:
            self.assertEqual(attn.shape, (1, 3, 197, 197), "Raw attention shape must be (B, 3, 197, 197)")

        grid = res["patch_grid_14x14"]
        self.assertEqual(grid.shape, (1, 14, 14), "Patch grid shape must be (1, 14, 14)")
        self.assertGreaterEqual(float(grid.min()), 0.0)
        self.assertLessEqual(float(grid.max()), 1.0 + 1e-6)

        heatmap = res["heatmap_224x224"]
        self.assertEqual(heatmap.shape, (1, 224, 224), "Heatmap shape must be (1, 224, 224)")
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0 + 1e-6)

        # Check top-k patches structure
        top_k = res["top_k_patches"]
        self.assertEqual(len(top_k), 10)
        self.assertIn("patch_index", top_k[0])
        self.assertIn("grid_row", top_k[0])
        self.assertIn("grid_col", top_k[0])

    def test_prediction_invariance_after_rollout(self):
        """Verifies that running rollout does NOT alter model state or predictions."""
        with torch.no_grad():
            clean_logits_before = self.model(self.dummy_input)

        rollout = ViTAttentionRollout(self.model)
        _ = rollout.compute_rollout(self.dummy_input, device=self.device)

        with torch.no_grad():
            clean_logits_after = self.model(self.dummy_input)

        diff = torch.max(torch.abs(clean_logits_before - clean_logits_after)).item()
        self.assertAlmostEqual(diff, 0.0, places=6, msg="Model logits must remain invariant after rollout")
        self.assertEqual(len(rollout.hooks), 0, "Hooks must be cleanly removed after rollout")

    def test_patch_occlusion_attribution(self):
        attr = ViTPatchAttribution(self.model, device=self.device)
        res = attr.compute_patch_occlusion(self.dummy_input, target_class=1)

        grid = res["attribution_grid_14x14"]
        self.assertEqual(grid.shape, (14, 14), "Occlusion grid shape must be (14, 14)")
        self.assertFalse(np.isnan(grid).any(), "Occlusion grid must not contain NaNs")

        heatmap = res["heatmap_224x224"]
        self.assertEqual(heatmap.shape, (224, 224), "Occlusion heatmap shape must be (224, 224)")


if __name__ == "__main__":
    unittest.main()
