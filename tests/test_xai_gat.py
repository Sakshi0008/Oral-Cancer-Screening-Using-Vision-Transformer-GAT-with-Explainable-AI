"""
Unit Tests for GATv2 Graph Attention and Attribution
====================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.graph_classifier import GraphClassifier
from src.models.hybrid_fusion import ConcatenationFusion
from src.xai.gat_attribution import GATv2GraphAttribution


class TestGATv2XAI(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.gat = GraphClassifier(
            graph_type="hybrid", gnn_type="gatv2", pooling="mean", in_features=192,
            hidden_per_head=48, num_heads=4, num_layers=2, num_classes=2, k=8
        ).to(self.device)
        self.gat.eval()

        self.fusion = ConcatenationFusion(vit_dim=192, graph_dim=192, hidden_dim=128, num_classes=2).to(self.device)
        self.fusion.eval()

        self.dummy_tokens = torch.randn(1, 196, 192, device=self.device)
        self.dummy_cls = torch.randn(1, 192, device=self.device)

    def test_gatv2_attention_shapes_and_heads(self):
        attr = GATv2GraphAttribution(self.gat, include_self_loops=False, device=self.device)
        res = attr.extract_graph_attention(self.dummy_tokens, top_k_edges=20)

        # Raw attentions
        raw = res["raw_attentions"]
        self.assertEqual(len(raw), 2, "Should capture 2 GATv2 layers")
        self.assertEqual(raw[0].shape, (1, 4, 196, 196), "Layer 1 shape must be (1, 4, 196, 196)")
        self.assertEqual(raw[1].shape, (1, 4, 196, 196), "Layer 2 shape must be (1, 4, 196, 196)")

        # Node contribution grid
        node_grid = res["node_contribution_14x14"]
        self.assertEqual(node_grid.shape, (14, 14))
        self.assertFalse(np.isnan(node_grid).any())
        self.assertGreaterEqual(float(node_grid.min()), 0.0)
        self.assertLessEqual(float(node_grid.max()), 1.0 + 1e-6)

        # Node heatmap
        heatmap = res["node_heatmap_224x224"]
        self.assertEqual(heatmap.shape, (224, 224))

        # Top edges
        top_edges = res["top_edges"]
        self.assertEqual(len(top_edges), 20)
        for e in top_edges:
            self.assertIn("src_node", e)
            self.assertIn("dst_node", e)
            self.assertIn("edge_type", e)
            self.assertIn(e["edge_type"], ["spatial_grid", "feature_similarity", "shared_hybrid", "other"])

    def test_edge_perturbation_validation(self):
        from src.graph.graph_builder import build_image_graph
        adj = build_image_graph("hybrid", self.dummy_tokens, k=8, add_self_loops=True)

        attr = GATv2GraphAttribution(self.gat, include_self_loops=False, device=self.device)
        res = attr.extract_graph_attention(self.dummy_tokens, adj=adj, top_k_edges=10)

        pert_res = attr.validate_edge_perturbation(
            patch_tokens=self.dummy_tokens,
            adj=adj,
            fusion_model=self.fusion,
            vit_cls=self.dummy_cls,
            edges_to_remove=res["top_edges"][:5],
            target_class=1
        )

        self.assertIn("clean_logit", pert_res)
        self.assertIn("perturbed_logit", pert_res)
        self.assertIn("logit_drop", pert_res)
        self.assertEqual(pert_res["edges_removed_count"], 5)


if __name__ == "__main__":
    unittest.main()
