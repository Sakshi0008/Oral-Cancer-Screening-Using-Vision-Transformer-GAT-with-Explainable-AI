"""
ViT + GAT Hybrid Fusion Architecture
===================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements hybrid representation fusion combining:
1. ViT CLS token representation [B, 192] (captures global contextual attention)
2. GAT / GATv2 graph-pooled representation [B, 192] (captures topological patch relationships)

Supported Fusion Approaches:
1. Concatenation Fusion:
   [h_cls || h_graph] in R^{B x 384} -> Linear(384, 128) -> LayerNorm -> ReLU -> Dropout -> Linear(128, 2)
2. Gated Fusion:
   g = sigmoid(Linear(384, 192)([h_cls || h_graph]))
   h_fused = g * h_cls + (1 - g) * h_graph in R^{B x 192}
   Classifier: Linear(192, 64) -> LayerNorm -> ReLU -> Dropout -> Linear(64, 2)
3. Residual Fusion:
   h_res = MLP(h_graph) in R^{B x 192}
   h_fused = LayerNorm(h_cls + h_res)
   Classifier: Linear(192, 64) -> LayerNorm -> ReLU -> Dropout -> Linear(64, 2)
"""

from typing import Dict, Optional, Tuple, Literal
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


class ConcatenationFusion(nn.Module):
    """
    Concatenation Fusion: [B, 192] + [B, 192] -> [B, 384] -> [B, num_classes]
    """
    def __init__(
        self,
        vit_dim: int = 192,
        graph_dim: int = 192,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        in_dim = vit_dim + graph_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, vit_cls: torch.Tensor, graph_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([vit_cls, graph_rep], dim=-1)
        logits = self.classifier(fused)
        return logits, fused


class GatedFusion(nn.Module):
    """
    Gated Fusion: Learnable gating mechanism that balances ViT CLS and GAT graph representation:
    g = sigmoid(W_g [h_cls || h_graph] + b_g) in [0, 1]^{B x 192}
    h_fused = g * h_cls + (1 - g) * h_graph in R^{B x 192}
    """
    def __init__(
        self,
        vit_dim: int = 192,
        graph_dim: int = 192,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        if vit_dim != graph_dim:
            raise ValueError(f"Gated fusion requires vit_dim == graph_dim, got {vit_dim} and {graph_dim}")

        self.gate_proj = nn.Linear(vit_dim + graph_dim, vit_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, vit_cls: torch.Tensor, graph_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat([vit_cls, graph_rep], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))  # [B, 192]
        fused = gate * vit_cls + (1.0 - gate) * graph_rep  # [B, 192]
        logits = self.classifier(fused)
        return logits, fused, gate


class ResidualFusion(nn.Module):
    """
    Residual Fusion: ViT CLS serves as identity baseline; GAT graph representation provides residual correction:
    h_fused = LayerNorm(h_cls + MLP(h_graph))
    """
    def __init__(
        self,
        vit_dim: int = 192,
        graph_dim: int = 192,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.res_mlp = nn.Sequential(
            nn.Linear(graph_dim, vit_dim),
            nn.LayerNorm(vit_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(vit_dim, vit_dim)
        )
        self.norm = nn.LayerNorm(vit_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(_init_weights)

    def forward(self, vit_cls: torch.Tensor, graph_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.res_mlp(graph_rep)
        fused = self.norm(vit_cls + residual)
        logits = self.classifier(fused)
        return logits, fused


class ViTGATHybridModel(nn.Module):
    """
    Wrapper model for ViT + GAT Hybrid Fusion.
    Can operate in frozen feature mode (taking pre-extracted vit_cls and graph_rep tensors)
    or end-to-end mode.
    """
    def __init__(
        self,
        fusion_type: Literal["concat", "gated", "residual"] = "concat",
        vit_dim: int = 192,
        graph_dim: int = 192,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = ConcatenationFusion(vit_dim=vit_dim, graph_dim=graph_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(vit_dim=vit_dim, graph_dim=graph_dim, num_classes=num_classes, dropout=dropout)
        elif fusion_type == "residual":
            self.fusion = ResidualFusion(vit_dim=vit_dim, graph_dim=graph_dim, num_classes=num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, vit_cls: torch.Tensor, graph_rep: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.fusion_type == "gated":
            logits, fused, gate = self.fusion(vit_cls, graph_rep)
            return {"logits": logits, "fused": fused, "gate": gate}
        else:
            logits, fused = self.fusion(vit_cls, graph_rep)
            return {"logits": logits, "fused": fused}
