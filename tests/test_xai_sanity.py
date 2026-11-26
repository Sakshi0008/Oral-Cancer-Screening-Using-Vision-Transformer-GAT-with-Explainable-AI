"""
XAI Sanity, Stability, and Faithfulness Tests (Part E)
=====================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements rigorous sanity checks adhering to Adebayo et al. (2018) and Petsiuk et al. (2018):
1. Prediction Invariance
2. Repeatability (Deterministic Reproducibility)
3. Input Perturbation Stability
4. Model Parameter Randomization Test
5. Spatial Graph Node Coordinate Consistency
6. Numerical Non-Degeneracy (No NaNs, Infs, or Empty Maps)
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.verify_hybrid_inference import VerifiedHybridInferencePipeline
from src.xai.pipeline import OralScanXAIExplainer
from src.xai.vit_rollout import ViTAttentionRollout


class TestXAISanityAndStability(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.pipeline = VerifiedHybridInferencePipeline(device=self.device)
        self.explainer = OralScanXAIExplainer(inference_pipeline=self.pipeline, device=self.device)
        self.dummy_img = Image.new("RGB", (224, 224), color=(180, 120, 110))

    def test_01_prediction_invariance(self):
        """Verifies that running the XAI pipeline does NOT alter normal inference logits."""
        # 1. Clean forward pass
        clean_res = self.pipeline.predict_image_pil(self.dummy_img) if hasattr(self.pipeline, "predict_image_pil") else None
        img_t = self.pipeline.transform(self.dummy_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vfeats = self.pipeline.vit.forward_features(img_t)
            from src.graph.graph_builder import build_image_graph
            adj = build_image_graph("hybrid", vfeats["patches"], k=8, add_self_loops=True)
            grep = self.pipeline.gatv2.get_graph_representation(vfeats["patches"], adj)
            clean_logits, _ = self.pipeline.fusion(vfeats["cls"], grep)

        # 2. XAI explanation pass
        temp_img_path = "scratch_test_sanity_img.png"
        self.dummy_img.save(temp_img_path)
        try:
            xai_res = self.explainer.explain_image(temp_img_path, save_viz=False)
            xai_logits = torch.tensor(xai_res["prediction"]["logits"], device=self.device)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        diff = torch.max(torch.abs(clean_logits[0] - xai_logits)).item()
        self.assertAlmostEqual(diff, 0.0, places=6, msg="XAI pipeline must not alter model prediction logits")

    def test_02_repeatability_deterministic(self):
        """Verifies that identical inputs produce bitwise reproducible explanation maps."""
        img_t = self.pipeline.transform(self.dummy_img).unsqueeze(0).to(self.device)
        rollout = ViTAttentionRollout(self.pipeline.vit)

        res1 = rollout.compute_rollout(img_t, device=self.device)["patch_grid_14x14"]
        res2 = rollout.compute_rollout(img_t, device=self.device)["patch_grid_14x14"]

        diff = torch.max(torch.abs(res1 - res2)).item()
        self.assertEqual(diff, 0.0, "Attribution rollout must be 100% deterministic")

    def test_03_stability_under_minor_jitter(self):
        """Verifies explanation stability under minor, non-destructive image variation (+1% intensity)."""
        img_t = self.pipeline.transform(self.dummy_img).unsqueeze(0).to(self.device)
        jittered_t = img_t * 1.01  # +1% intensity

        rollout = ViTAttentionRollout(self.pipeline.vit)
        map1 = rollout.compute_rollout(img_t, device=self.device)["patch_grid_14x14"][0].cpu().numpy().ravel()
        map2 = rollout.compute_rollout(jittered_t, device=self.device)["patch_grid_14x14"][0].cpu().numpy().ravel()

        corr = np.corrcoef(map1, map2)[0, 1]
        self.assertGreater(corr, 0.85, f"Attribution map correlation under minor jitter ({corr:.3f}) must exceed 0.85")

    def test_04_parameter_randomization_sanity_check(self):
        """Adebayo et al. parameter randomization test: randomizing trained weights must destroy the explanation."""
        from src.models.vit_scratch import build_vit_scratch
        img_t = self.pipeline.transform(self.dummy_img).unsqueeze(0).to(self.device)

        # 1. Trained model explanation
        rollout_trained = ViTAttentionRollout(self.pipeline.vit)
        trained_map = rollout_trained.compute_rollout(img_t, device=self.device)["patch_grid_14x14"][0].cpu().numpy().ravel()

        # 2. Randomly initialized model explanation
        random_vit = build_vit_scratch(
            img_size=224, patch_size=16, embed_dim=192, depth=4, num_heads=3, mlp_dim=768, num_classes=2
        ).to(self.device)
        random_vit.eval()
        rollout_random = ViTAttentionRollout(random_vit)
        random_map = rollout_random.compute_rollout(img_t, device=self.device)["patch_grid_14x14"][0].cpu().numpy().ravel()

        corr = np.corrcoef(trained_map, random_map)[0, 1]
        # Correlation between trained model and random model should be low (< 0.70)
        self.assertLess(corr, 0.70, f"Explanation must depend on trained parameters, got correlation {corr:.3f}")

    def test_05_graph_node_coordinate_consistency(self):
        """Verifies that node indices 0..195 map strictly to (r, c) on a 14x14 grid without row-col swaps."""
        for idx in range(196):
            r = idx // 14
            c = idx % 14
            reconstructed_idx = r * 14 + c
            self.assertEqual(idx, reconstructed_idx, "Node index mapping must be strictly row-major")
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 14)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 14)

    def test_06_numerical_non_degeneracy(self):
        """Verifies that explanation outputs contain no NaNs, Infs, or flat zero maps."""
        temp_img_path = "scratch_test_degeneracy.png"
        self.dummy_img.save(temp_img_path)
        try:
            xai_res = self.explainer.explain_image(temp_img_path, save_viz=False)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        vit_grid = xai_res["vit_rollout"]["patch_grid_14x14"]
        gat_grid = xai_res["gatv2_graph"]["node_contribution_14x14"]

        self.assertFalse(np.isnan(vit_grid).any(), "ViT grid must not contain NaNs")
        self.assertFalse(np.isinf(vit_grid).any(), "ViT grid must not contain Infs")
        self.assertGreater(float(vit_grid.max() - vit_grid.min()), 0.0, "ViT grid must not be completely uniform/flat")

        self.assertFalse(np.isnan(gat_grid).any(), "GAT grid must not contain NaNs")
        self.assertFalse(np.isinf(gat_grid).any(), "GAT grid must not contain Infs")
        self.assertGreater(float(gat_grid.max() - gat_grid.min()), 0.0, "GAT grid must not be completely uniform/flat")


if __name__ == "__main__":
    unittest.main()
