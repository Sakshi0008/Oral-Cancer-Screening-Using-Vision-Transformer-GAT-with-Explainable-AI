"""
Vision Transformer Attention Rollout Module
===========================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements residual-aware attention rollout (Abnar & Zuidema, 2020) across all 4 transformer layers
of the scratch-trained Vision Transformer without modifying model code or checkpoint weights.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViTAttentionRollout:
    """
    Computes Attention Rollout for VisionTransformerScratch.
    Uses non-invasive PyTorch forward hooks to capture raw multi-head attention weights.
    """
    def __init__(
        self,
        vit_model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        add_residual: bool = True
    ):
        """
        Args:
            vit_model: Instantiated VisionTransformerScratch model.
            head_fusion: Strategy to fuse attention heads: 'mean', 'max', or 'min'.
            discard_ratio: Top-k filtering ratio to remove weak attention connections (default 0.0).
            add_residual: Whether to account for residual identity connections (default True).
        """
        self.model = vit_model
        self.head_fusion = head_fusion.lower()
        self.discard_ratio = discard_ratio
        self.add_residual = add_residual
        self.hooks = []
        self.captured_attentions: List[torch.Tensor] = []

    def _register_hooks(self):
        """Registers non-invasive forward hooks on all MultiHeadSelfAttention modules."""
        self._remove_hooks()
        self.captured_attentions = []

        def get_hook():
            def hook(module, inp, out):
                # out is Tuple[x, attn_weights] where attn_weights is (B, H, N, N)
                attn = out[1].detach()
                self.captured_attentions.append(attn)
            return hook

        for block in self.model.blocks:
            h = block.attn.register_forward_hook(get_hook())
            self.hooks.append(h)

    def _remove_hooks(self):
        """Removes registered hooks cleanly."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_rollout(
        self,
        img_tensor: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Performs forward pass with hooks, computes attention rollout, and maps to 14x14 and 224x224.

        Args:
            img_tensor: Input image tensor of shape (B, 3, 224, 224) or (1, 3, 224, 224).
            device: Compute device.

        Returns:
            Dictionary containing:
                - 'raw_attentions': List of 4 tensors of shape (B, num_heads, 197, 197)
                - 'rollout_matrix': Full rollout matrix (B, 197, 197)
                - 'cls_patch_scores': Raw CLS-to-patch attention vector (B, 196)
                - 'patch_grid_14x14': 2D patch attention grid (B, 14, 14) in [0, 1]
                - 'heatmap_224x224': Upsampled continuous heatmap (B, 224, 224) in [0, 1]
                - 'top_k_patches': List of top patch coordinates [(r, c, score)]
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        dev = device or next(self.model.parameters()).device
        img_tensor = img_tensor.to(dev)

        self._register_hooks()
        try:
            with torch.no_grad():
                _ = self.model.forward_features(img_tensor)
        finally:
            self._remove_hooks()

        assert len(self.captured_attentions) == len(self.model.blocks), (
            f"Expected {len(self.model.blocks)} attention matrices, captured {len(self.captured_attentions)}"
        )

        B = img_tensor.shape[0]
        num_tokens = self.captured_attentions[0].shape[-1]  # 197
        eye = torch.eye(num_tokens, device=dev).unsqueeze(0).expand(B, -1, -1)

        # Rollout recursion: R_0 = I
        rollout = eye.clone()

        for layer_idx, attn in enumerate(self.captured_attentions):
            # 1. Aggregate heads: (B, H, 197, 197) -> (B, 197, 197)
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion '{self.head_fusion}'")

            # 2. Top-k filtering if discard_ratio > 0
            if self.discard_ratio > 0.0:
                flat = attn_fused.view(B, -1)
                k = int(flat.shape[-1] * (1.0 - self.discard_ratio))
                threshold = torch.topk(flat, k=k, dim=-1)[0][:, -1].unsqueeze(-1).unsqueeze(-1)
                attn_fused = torch.where(attn_fused >= threshold, attn_fused, torch.zeros_like(attn_fused))

            # 3. Residual connection weighting: A_hat = 0.5 * A + 0.5 * I
            if self.add_residual:
                attn_hat = 0.5 * attn_fused + 0.5 * eye
            else:
                attn_hat = attn_fused

            # 4. Row normalization to preserve stochastic matrix property
            attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-9)

            # 5. Matrix multiplication: R_l = A_hat @ R_{l-1}
            rollout = torch.bmm(attn_hat, rollout)

        # Extract CLS token attention to all 196 patches: [B, 196]
        # Token 0 is CLS; tokens 1..196 are spatial patches
        cls_to_patches = rollout[:, 0, 1:]  # (B, 196)

        # Reshape to 14x14 grid
        patch_grid = cls_to_patches.view(B, 14, 14)

        # Min-Max normalize per image to [0.0, 1.0] for stable visualization
        grid_min = patch_grid.amin(dim=(1, 2), keepdim=True)
        grid_max = patch_grid.amax(dim=(1, 2), keepdim=True)
        norm_grid = (patch_grid - grid_min) / (grid_max - grid_min + 1e-9)

        # Upsample to 224x224 using bilinear interpolation
        heatmap = F.interpolate(
            norm_grid.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (B, 224, 224)

        # Extract top-k patches for sample 0
        grid_np = norm_grid[0].cpu().numpy()
        top_indices = np.argsort(grid_np.ravel())[::-1]
        top_k = []
        for idx in top_indices[:10]:
            r = int(idx // 14)
            c = int(idx % 14)
            top_k.append({
                "patch_index": int(idx),
                "grid_row": r,
                "grid_col": c,
                "score": float(grid_np[r, c])
            })

        return {
            "raw_attentions": [a.cpu() for a in self.captured_attentions],
            "rollout_matrix": rollout.cpu(),
            "cls_patch_scores": cls_to_patches.cpu(),
            "patch_grid_14x14": norm_grid.cpu(),
            "heatmap_224x224": heatmap.cpu(),
            "top_k_patches": top_k
        }
