"""
Handcrafted Feature Projection and Tri-Modal Fusion Architectures
================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Integrates 143-dimensional handcrafted features with ViT and GAT representations:
1. Handcrafted Projection Head:
   Linear(143, 64) -> LayerNorm(64) -> ReLU -> Dropout(0.2) -> Linear(64, 64)
2. Handcrafted-Only Classifier (for Phase 6 baseline):
   Linear(143, 64) -> LayerNorm(64) -> ReLU -> Dropout(0.2) -> Linear(64, 2)
3. Bi-Modal Fusion:
   - ViT + Handcrafted: [h_cls (192) || h_hc (64)] = 256 dims -> MLP -> 2
   - GAT + Handcrafted: [h_graph (192) || h_hc (64)] = 256 dims -> MLP -> 2
4. Tri-Modal Fusion:
   - Concatenation: [h_cls (192) || h_graph (192) || h_hc (64)] = 448 dims -> Linear(448, 128) -> LN -> ReLU -> Linear(128, 2)
   - Modality-Gated Fusion: Projects each modality to d=128, learns softmax weights [a1, a2, a3], forms weighted combination.
"""

from typing import Dict, Literal, Optional, Tuple
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


class HandcraftedProjection(nn.Module):
    """
    Projects raw 143-dim standardized handcrafted features into a compact 64-dim embedding.
    """
    def __init__(self, in_dim: int = 143, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HandcraftedOnlyClassifier(nn.Module):
    """
    Standalone classifier for handcrafted features only.
    """
    def __init__(self, in_dim: int = 143, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiModalHandcraftedFusion(nn.Module):
    """
    Fuses deep representation (ViT CLS or GAT graph, dim=192) with handcrafted features (dim=143).
    """
    def __init__(
        self,
        deep_dim: int = 192,
        hc_dim: int = 143,
        hc_proj_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hc_proj = HandcraftedProjection(in_dim=hc_dim, out_dim=hc_proj_dim, dropout=dropout)
        fused_dim = deep_dim + hc_proj_dim  # 192 + 64 = 256
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, deep_rep: torch.Tensor, hc_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hc = self.hc_proj(hc_feat)
        fused = torch.cat([deep_rep, h_hc], dim=-1)
        logits = self.classifier(fused)
        return logits, fused


class TriModalConcatFusion(nn.Module):
    """
    Concatenates ViT CLS (192) + GAT graph (192) + Handcrafted projection (64) = 448 dims.
    """
    def __init__(
        self,
        vit_dim: int = 192,
        graph_dim: int = 192,
        hc_dim: int = 143,
        hc_proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hc_proj = HandcraftedProjection(in_dim=hc_dim, out_dim=hc_proj_dim, dropout=dropout)
        total_in = vit_dim + graph_dim + hc_proj_dim  # 448
        self.classifier = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
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
        vit_cls: torch.Tensor,
        graph_rep: torch.Tensor,
        hc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hc = self.hc_proj(hc_feat)
        fused = torch.cat([vit_cls, graph_rep, h_hc], dim=-1)
        logits = self.classifier(fused)
        return logits, fused


class TriModalGatedFusion(nn.Module):
    """
    Modality-Attention Gated Fusion:
    Projects all three modalities into a common d=128 latent space,
    computes modality importance softmax weights [a_vit, a_graph, a_hc],
    and computes the weighted representation.
    """
    def __init__(
        self,
        vit_dim: int = 192,
        graph_dim: int = 192,
        hc_dim: int = 143,
        latent_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.proj_graph = nn.Sequential(
            nn.Linear(graph_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.proj_hc = nn.Sequential(
            nn.Linear(hc_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        self.modality_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        self.apply(_init_weights)

    def forward(
        self,
        vit_cls: torch.Tensor,
        graph_rep: torch.Tensor,
        hc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_v = self.proj_vit(vit_cls)       # [B, 128]
        z_g = self.proj_graph(graph_rep)   # [B, 128]
        z_h = self.proj_hc(hc_feat)        # [B, 128]

        gate_in = torch.cat([z_v, z_g, z_h], dim=-1)  # [B, 384]
        weights = F.softmax(self.modality_gate(gate_in), dim=-1)  # [B, 3]

        w_v = weights[:, 0:1]
        w_g = weights[:, 1:2]
        w_h = weights[:, 2:3]

        fused = w_v * z_v + w_g * z_g + w_h * z_h  # [B, 128]
        logits = self.classifier(fused)
        return logits, fused, weights
