"""
Graph Attention Network (GAT) — Pure PyTorch Implementation
============================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

STRICT CONSTRAINTS:
- Pure PyTorch implementation without external graph library dependencies.
- Multi-head graph self-attention with learnable node transformations.
- Numerically stable softmax normalization over adjacent neighbours.
- Handles batched dense adjacency matrices [B, 196, 196] with zero inter-image leakage.
- Residual connections, LayerNorm, and ELU activations.

Specification:
- Input feature dimension: 192
- Number of attention heads: 4
- Dimension per head: 48 (4 * 48 = 192 output dimension)
- Number of GAT layers: 2
- Dropout: 0.1
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Multi-Head Graph Attention Network (GAT) Layer (Veličković et al., 2018).
    e_ij = LeakyReLU( a_src^T W h_i + a_dst^T W h_j )
    alpha_ij = Softmax_j( e_ij * mask_ij )
    h'_i = sum_{j in N(i)} alpha_ij W h_j
    """
    def __init__(
        self,
        in_features: int = 192,
        out_features_per_head: int = 48,
        num_heads: int = 4,
        feat_drop: float = 0.1,
        attn_drop: float = 0.1,
        alpha: float = 0.2,
        residual: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.num_heads = num_heads
        self.total_out_features = out_features_per_head * num_heads  # 48 * 4 = 192
        self.alpha = alpha
        self.residual = residual

        # Linear projection for node features
        self.fc = nn.Linear(in_features, self.total_out_features, bias=False)

        # Attention parameters: a_src and a_dst per head
        self.attn_src = nn.Parameter(torch.empty(1, num_heads, 1, out_features_per_head))
        self.attn_dst = nn.Parameter(torch.empty(1, num_heads, 1, out_features_per_head))

        # Dropouts and activations
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = nn.ELU()
        self.norm = nn.LayerNorm(self.total_out_features)

        # Residual shortcut if dimensions match or linear projection if not
        if residual and in_features != self.total_out_features:
            self.res_proj = nn.Linear(in_features, self.total_out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src, gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst, gain=1.414)
        if hasattr(self.res_proj, "weight"):
            nn.init.xavier_uniform_(self.res_proj.weight, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features tensor of shape (B, N, in_features).
            adj: Binary adjacency matrix of shape (B, N, N) where adj[b, i, j] = 1 indicates edge j -> i.
        Returns:
            out: Updated node features tensor of shape (B, N, total_out_features).
            attn_weights: Attention coefficients of shape (B, num_heads, N, N).
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.out_features_per_head

        # 1. Feature dropout and linear transformation
        h = self.feat_dropout(x)
        # (B, N, H * D) -> (B, N, H, D) -> (B, H, N, D)
        z = self.fc(h).reshape(B, N, H, D).transpose(1, 2)

        # 2. Compute attention score components
        # (B, H, N, D) * (1, H, 1, D) -> sum over D -> (B, H, N, 1)
        score_src = (z * self.attn_src).sum(dim=-1, keepdim=True)
        # (B, H, N, D) * (1, H, 1, D) -> sum over D -> (B, H, 1, N)
        score_dst = (z * self.attn_dst).sum(dim=-1, keepdim=True).transpose(-2, -1)

        # Pairwise attention logits: (B, H, N, N)
        # e_ij corresponds to edge from j (dst) to i (src)
        e = self.leaky_relu(score_src + score_dst)

        # 3. Mask out non-existent edges (where adj[b, i, j] == 0)
        # adj shape: (B, N, N) -> expand to (B, 1, N, N)
        mask = (adj.unsqueeze(1) > 0)
        e = e.masked_fill(~mask, -1e9)

        # 4. Numerically stable softmax across neighbours (dim=-1 is destination nodes j)
        alpha = F.softmax(e, dim=-1)
        # Replace any all-masked rows with 0.0 to prevent NaNs
        alpha = torch.nan_to_num(alpha, nan=0.0)
        attn_weights = alpha

        # 5. Attention dropout
        alpha_drop = self.attn_dropout(alpha)

        # 6. Neighbour aggregation: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        out = torch.matmul(alpha_drop, z)

        # 7. Concatenate heads: (B, H, N, D) -> (B, N, H, D) -> (B, N, H * D)
        out = out.transpose(1, 2).reshape(B, N, H * D)

        # 8. Non-linearity, Residual Connection and LayerNorm
        out = self.activation(out)
        if self.residual:
            shortcut = self.res_proj(x)
            out = self.norm(out + shortcut)
        else:
            out = self.norm(out)

        return out, attn_weights


class GAT(nn.Module):
    """
    2-Layer Graph Attention Network.
    Input:  Node features [B, 196, 192], Adjacency [B, 196, 196]
    Output: Node representations [B, 196, 192]
    """
    def __init__(
        self,
        in_features: int = 192,
        hidden_features_per_head: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        super().__init__()
        self.is_pretrained = False
        self.num_layers = num_layers
        self.out_features = hidden_features_per_head * num_heads  # 192

        self.layers = nn.ModuleList([
            GATLayer(
                in_features=in_features if i == 0 else self.out_features,
                out_features_per_head=hidden_features_per_head,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                alpha=alpha,
                residual=True
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all GAT layers.
        Returns:
            x: Updated node features of shape (B, N, 192).
            all_attn: List of attention weight tensors from each layer.
        """
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x, adj)
            all_attn.append(attn)
        return x, all_attn


def build_gat(
    in_features: int = 192,
    hidden_features_per_head: int = 48,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> GAT:
    """Factory function for GAT."""
    model = GAT(
        in_features=in_features,
        hidden_features_per_head=hidden_features_per_head,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    dummy_x = torch.randn(2, 196, 192)
    dummy_adj = torch.eye(196).unsqueeze(0).expand(2, -1, -1)
    gat = build_gat()
    out, attns = gat(dummy_x, dummy_adj)
    print("GAT Output Shape:", out.shape, "(Expected: [2, 196, 192])")
    print(f"Number of GAT layers: {len(attns)}, Attn shape: {attns[0].shape}")
    assert out.shape == (2, 196, 192)
    assert len(attns) == 2
    assert attns[0].shape == (2, 4, 196, 196)
    print("GAT Architecture Verification PASSED!")
