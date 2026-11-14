"""
Graph-Level Representation & Pooling Module
===========================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Converts node-level patch representations [B, 196, 192] into one graph-level representation [B, 192].
Supports:
1. GlobalMeanPooling: Unweighted spatial/feature averaging across all 196 patches.
2. AttentionGraphPooling: Learnable content-aware attention weighting highlighting informative patches.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMeanPooling(nn.Module):
    """
    Global Mean Pooling across nodes.
    Input:  (B, N, D)
    Output: (B, D)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            # mask: (B, N) with 1 for valid nodes, 0 for padding
            x = x * mask.unsqueeze(-1)
            sum_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return x.sum(dim=1) / sum_nodes
        return x.mean(dim=1)


class AttentionGraphPooling(nn.Module):
    """
    Attention-based Graph Pooling (Learnable Gated Attention).
    Computes an attention coefficient alpha_i for each patch node:
        score_i = w^T tanh(W h_i + b)
        alpha = Softmax(score_i)
        h_graph = sum_i alpha_i * h_i
    Input:  (B, N, D)
    Output: (B, D)
    """
    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            mask: (B, N) optional binary mask
        Returns:
            graph_rep: (B, D)
        """
        # scores: (B, N, 1)
        scores = self.proj(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (B, N, 1)
        # Weighted sum across nodes: (B, N, 1) * (B, N, D) -> sum over dim 1 -> (B, D)
        graph_rep = (attn_weights * x).sum(dim=1)
        return graph_rep


def build_graph_pooling(
    method: str = "mean",
    in_dim: int = 192,
    hidden_dim: int = 64,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function for graph pooling.
    Args:
        method: 'mean' for GlobalMeanPooling, 'attention' for AttentionGraphPooling.
    """
    m = method.lower().strip()
    if m in ("mean", "global_mean", "avg"):
        return GlobalMeanPooling()
    elif m in ("attention", "attn", "gated_attention"):
        return AttentionGraphPooling(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown pooling method '{method}'. Choose 'mean' or 'attention'.")


if __name__ == "__main__":
    dummy_nodes = torch.randn(4, 196, 192)
    mean_pool = build_graph_pooling("mean")
    attn_pool = build_graph_pooling("attention", in_dim=192, hidden_dim=64)

    out_mean = mean_pool(dummy_nodes)
    out_attn = attn_pool(dummy_nodes)

    print("Mean Pooling Output Shape:", out_mean.shape, "(Expected: [4, 192])")
    print("Attn Pooling Output Shape:", out_attn.shape, "(Expected: [4, 192])")
    assert out_mean.shape == (4, 192)
    assert out_attn.shape == (4, 192)
    print("Graph Pooling Verification PASSED!")
