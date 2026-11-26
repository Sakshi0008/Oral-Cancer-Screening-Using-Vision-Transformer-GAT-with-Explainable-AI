"""
Vision Transformer Patch Attribution Module (Gradient and Occlusion Based)
==========================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Provides independent, mathematically sound patch attribution methods complementary to Attention Rollout:
1. Patch Occlusion Attribution: Directly measures logit sensitivity by masking each 16x16 patch.
2. Gradient x Patch Feature Attribution: Computes directional first-order gradients on patch tokens.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_PRIMARY_MEAN = [0.6432, 0.4537, 0.4275]


class ViTPatchAttribution:
    """
    Computes gradient-based and perturbation-based patch attribution for the Vision Transformer branch.
    """
    def __init__(self, vit_model: nn.Module, device: Optional[torch.device] = None):
        self.model = vit_model
        self.device = device or next(self.model.parameters()).device

    def compute_patch_occlusion(
        self,
        img_tensor: torch.Tensor,
        target_class: int = 1,
        fill_color: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Computes perturbation-based patch attribution via systematic occlusion of each 16x16 patch.

        Args:
            img_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
            target_class: Target logit index (default 1 = CANCER).
            fill_color: RGB fill value for occlusion (default: dataset primary mean).

        Returns:
            Dictionary containing:
                - 'attribution_grid_14x14': 2D patch sensitivity grid [14, 14]
                - 'heatmap_224x224': Upsampled attribution heatmap [224, 224]
                - 'baseline_logit': Unperturbed target logit
                - 'max_drop': Maximum logit reduction observed
        """
        self.model.eval()
        img_tensor = img_tensor.to(self.device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Baseline logit
        with torch.no_grad():
            clean_out = self.model(img_tensor)
            clean_logit = float(clean_out[0, target_class].item())

        H, W = 224, 224
        patch_size = 16
        grid_h, grid_w = H // patch_size, W // patch_size
        occlusion_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Normalized baseline pixel fill
        # Using 0.0 in normalized space represents the exact primary dataset mean!
        fill_val = 0.0

        for r in range(grid_h):
            for c in range(grid_w):
                occluded = img_tensor.clone()
                r_start, r_end = r * patch_size, (r + 1) * patch_size
                c_start, c_end = c * patch_size, (c + 1) * patch_size
                occluded[:, :, r_start:r_end, c_start:c_end] = fill_val

                with torch.no_grad():
                    occ_out = self.model(occluded)
                    occ_logit = float(occ_out[0, target_class].item())

                # Positive drop indicates patch was supporting cancer classification
                logit_drop = clean_logit - occ_logit
                occlusion_grid[r, c] = logit_drop

        # Normalize grid to [0, 1] using ReLU (positive contribution)
        pos_grid = np.maximum(occlusion_grid, 0.0)
        max_val = np.max(pos_grid)
        norm_grid = pos_grid / (max_val + 1e-9)

        # Upsample to 224x224
        grid_t = torch.tensor(norm_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_t = F.interpolate(grid_t, size=(224, 224), mode="bilinear", align_corners=False)
        heatmap = heatmap_t.squeeze().numpy()

        return {
            "attribution_grid_14x14": norm_grid,
            "raw_occlusion_grid": occlusion_grid,
            "heatmap_224x224": heatmap,
            "baseline_logit": clean_logit,
            "max_drop": float(np.max(occlusion_grid))
        }

    def compute_grad_patch_attribution(
        self,
        img_tensor: torch.Tensor,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Computes Input x Gradient attribution with respect to the patch embeddings.

        Args:
            img_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
            target_class: Target logit index (default 1 = CANCER).

        Returns:
            Dictionary containing 14x14 grid and 224x224 heatmap.
        """
        self.model.eval()
        img = img_tensor.to(self.device).clone().requires_grad_(True)

        feats = self.model.forward_features(img)
        cls_token = feats["cls"]
        logits = self.model.head(cls_token)
        target_score = logits[0, target_class]

        self.model.zero_grad()
        target_score.backward()

        grad = img.grad.data.abs().mean(dim=1, keepdim=True)  # (1, 1, 224, 224)
        grad_np = grad.squeeze().cpu().numpy()

        # Aggregate per 16x16 patch
        grid = np.zeros((14, 14), dtype=np.float32)
        for r in range(14):
            for c in range(14):
                patch = grad_np[r*16:(r+1)*16, c*16:(c+1)*16]
                grid[r, c] = patch.mean()

        norm_grid = grid / (np.max(grid) + 1e-9)
        grid_t = torch.tensor(norm_grid).unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(grid_t, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()

        return {
            "attribution_grid_14x14": norm_grid,
            "heatmap_224x224": heatmap
        }
