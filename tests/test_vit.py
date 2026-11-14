"""
Unit Tests for Custom Vision Transformer (From-Scratch)
=======================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.vit_scratch import VisionTransformerScratch, build_vit_scratch


class TestVisionTransformerScratch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_vit_scratch(
            img_size=224,
            patch_size=16,
            num_classes=2,
            embed_dim=192,
            depth=4,
            num_heads=3,
            mlp_dim=768,
            dropout=0.1
        )
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_parameter_counts_and_pretrained_flag(self):
        self.assertFalse(self.model.is_pretrained, "Pretrained flag must be False")
        total_params = sum(p.numel() for p in self.model.parameters())
        # Exactly 1,965,890 parameters for 4 layers, embed_dim 192, mlp_dim 768
        self.assertEqual(total_params, 1_965_890)

    def test_patch_embedding_dimensions(self):
        patch_embed = self.model.patch_embed(self.dummy_input)
        self.assertEqual(patch_embed.shape, (2, 196, 192), "Patch embedding shape must be (B, 196, 192)")

    def test_cls_and_positional_embeddings(self):
        self.assertEqual(self.model.cls_token.shape, (1, 1, 192))
        self.assertEqual(self.model.pos_embed.shape, (1, 197, 192))

    def test_forward_output_dimensions(self):
        logits = self.model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 2), "Logits shape must be (B, 2)")

    def test_forward_features_interface(self):
        feats = self.model.forward_features(self.dummy_input, return_all_tokens=True)
        self.assertIn("cls", feats)
        self.assertIn("patches", feats)
        self.assertIn("all_tokens", feats)
        self.assertEqual(feats["cls"].shape, (2, 192))
        self.assertEqual(feats["patches"].shape, (2, 196, 192))
        self.assertEqual(feats["all_tokens"].shape, (2, 197, 192))

    def test_convenience_accessors(self):
        patches = self.model.get_patch_embeddings(self.dummy_input)
        cls_rep = self.model.get_cls_token(self.dummy_input)
        self.assertEqual(patches.shape, (2, 196, 192))
        self.assertEqual(cls_rep.shape, (2, 192))

    def test_gradient_propagation_and_optimization_step(self):
        self.model.train()
        logits = self.model(self.dummy_input)
        targets = torch.tensor([1, 0], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        loss.backward()

        # Check gradients exist and are non-zero for all components
        self.assertIsNotNone(self.model.cls_token.grad)
        self.assertGreater(self.model.cls_token.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(self.model.pos_embed.grad)
        self.assertGreater(self.model.pos_embed.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(self.model.patch_embed.proj.weight.grad)
        self.assertGreater(self.model.patch_embed.proj.weight.grad.abs().sum().item(), 0.0)

        for i, block in enumerate(self.model.blocks):
            self.assertIsNotNone(block.attn.qkv.weight.grad, f"Block {i} MHSA gradient missing")
            self.assertIsNotNone(block.mlp.fc1.weight.grad, f"Block {i} MLP gradient missing")

        self.assertIsNotNone(self.model.head.weight.grad, "Head gradient missing")


if __name__ == "__main__":
    unittest.main()
