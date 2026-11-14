"""
Multi-Scale ViT Feature Representation and Fusion
=================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements multi-scale hierarchical spatial pooling over the 14x14 ViT patch tokens:
1. Fine Scale (Local Patches):
   Original 196 tokens (14x14 grid, d=192). Pooled fine representation h_fine in R^{B x 192}.
2. Intermediate Regional Scale (2x2 Region Blocks):
   Spatial 2x2 adaptive/average pooling with stride 2 over the 14x14 grid -> 7x7 = 49 region tokens.
   Regional attention/pooling yields h_region in R^{B x 192}.
3. Coarse Global Scale:
   ViT CLS token h_cls in R^{B x 192}.

Multi-Scale Fusion Models:
- MultiScaleViTClassifier: Fuses fine, regional, and global representations (d=576 -> 128 -> 2).
- MultiScaleHybridClassifier: Fuses Multi-Scale ViT representation with GAT graph-pooled representation.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MultiScalePatchPooler(nn.Module):
    """
    Transforms [B, 196, 192] patch tokens into multi-scale spatial representations:
    - Reshapes to [B, 192, 14, 14]
    - Computes 2x2 regional pooling -> [B, 192, 7, 7] -> 49 regional tokens
    - Computes fine-scale global pooling -> [B, 192]
    - Computes regional-scale global pooling -> [B, 192]
    """
    def __init__(self, embed_dim: int = 192, grid_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.region_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, 196, 192]
        Returns:
            h_fine: [B, 192] fine-scale summary
            h_region: [B, 192] intermediate regional summary
        """
        B, N, D = patch_tokens.shape
        assert N == self.grid_size * self.grid_size, f"Expected {self.grid_size*self.grid_size} patches, got {N}"

        # Transpose to [B, D, H, W] for 2D spatial pooling
        spatial = patch_tokens.transpose(1, 2).view(B, D, self.grid_size, self.grid_size)

        # 1. Fine scale summary (mean over all 196 patches)
        h_fine = torch.mean(patch_tokens, dim=1)  # [B, 192]

        # 2. Regional scale (2x2 pooling -> 7x7 grid)
        regions_2d = self.region_pool(spatial)  # [B, D, 7, 7]
        # Flatten regional tokens to [B, 49, D]
        regions_flat = regions_2d.flatten(2).transpose(1, 2)  # [B, 49, 192]
        h_region = torch.mean(regions_flat, dim=1)  # [B, 192]

        return h_fine, h_region


class MultiScaleViTClassifier(nn.Module):
    """
    Combines fine-scale, regional-scale, and global CLS representations:
    [h_cls (192) || h_region (192) || h_fine (192)] = 576 dims -> Linear(576, 128) -> LN -> ReLU -> Linear(128, 2)
    """
    def __init__(
        self,
        embed_dim: int = 192,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pooler = MultiScalePatchPooler(embed_dim=embed_dim)
        in_dim = embed_dim * 3  # 192 * 3 = 576
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, patch_tokens: torch.Tensor, vit_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_fine, h_region = self.pooler(patch_tokens)
        multiscale = torch.cat([vit_cls, h_region, h_fine], dim=-1)  # [B, 576]
        logits = self.classifier(multiscale)
        return logits, multiscale


class MultiScaleHybridClassifier(nn.Module):
    """
    Combines Multi-Scale ViT representation (576 dims) with GAT Graph representation (192 dims):
    Total in_dim = 576 + 192 = 768 dims -> Linear(768, 128) -> LN -> ReLU -> Linear(128, 2)
    """
    def __init__(
        self,
        embed_dim: int = 192,
        graph_dim: int = 192,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pooler = MultiScalePatchPooler(embed_dim=embed_dim)
        in_dim = (embed_dim * 3) + graph_dim  # 576 + 192 = 768
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        self.apply(_init_weights)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        vit_cls: torch.Tensor,
        graph_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_fine, h_region = self.pooler(patch_tokens)
        combined = torch.cat([vit_cls, h_region, h_fine, graph_rep], dim=-1)  # [B, 768]
        logits = self.classifier(combined)
        return logits, combined
