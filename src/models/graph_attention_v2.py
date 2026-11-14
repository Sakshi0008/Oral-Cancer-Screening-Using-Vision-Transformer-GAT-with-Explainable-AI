"""
Graph Attention Network v2 (GATv2) — Pure PyTorch Implementation
=================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

STRICT CONSTRAINTS:
- Pure PyTorch implementation without external graph library dependencies.
- Independent implementation of GATv2 dynamic attention (Brody et al., 2021).
- Distinct from standard GAT:
    Standard GAT (Static Attention):
        e_ij = LeakyReLU( a_src^T W h_i + a_dst^T W h_j )
    GATv2 (Dynamic Attention):
        e_ij = a^T LeakyReLU( W_src h_i + W_dst h_j )
    Dynamic attention allows the ranking of attention keys to depend on the query node,
    preventing the static attention bottleneck.

Specification:
- Input feature dimension: 192
- Number of attention heads: 4
- Dimension per head: 48 (4 * 48 = 192 output dimension)
- Number of GATv2 layers: 2
- Dropout: 0.1
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """
    Single Multi-Head GATv2 Layer with Dynamic Attention (Brody et al., 2021).
    Score calculation:
        e_ij = a^T LeakyReLU( W_src h_i + W_dst h_j )
        alpha_ij = Softmax_j( e_ij * mask_ij )
        h'_i = sum_{j in N(i)} alpha_ij (W_val h_j)
    """
    def __init__(
        self,
        in_features: int = 192,
        out_features_per_head: int = 48,
        num_heads: int = 4,
        feat_drop: float = 0.1,
        attn_drop: float = 0.1,
        alpha: float = 0.2,
        residual: bool = True,
        share_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.num_heads = num_heads
        self.total_out_features = out_features_per_head * num_heads  # 48 * 4 = 192
        self.alpha = alpha
        self.residual = residual
        self.share_weights = share_weights

        # Node transformations: W_src and W_dst
        self.fc_src = nn.Linear(in_features, self.total_out_features, bias=False)
        if share_weights:
            self.fc_dst = self.fc_src
        else:
            self.fc_dst = nn.Linear(in_features, self.total_out_features, bias=False)

        # Value transformation: W_val
        self.fc_val = nn.Linear(in_features, self.total_out_features, bias=False)

        # Dynamic attention vector 'a' per head: (1, num_heads, 1, 1, out_features_per_head)
        self.attn_vec = nn.Parameter(torch.empty(1, num_heads, 1, 1, out_features_per_head))

        # Dropouts and activations
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = nn.ELU()
        self.norm = nn.LayerNorm(self.total_out_features)

        # Residual shortcut
        if residual and in_features != self.total_out_features:
            self.res_proj = nn.Linear(in_features, self.total_out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_src.weight, gain=1.414)
        if not self.share_weights:
            nn.init.xavier_uniform_(self.fc_dst.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc_val.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_vec, gain=1.414)
        if hasattr(self.res_proj, "weight"):
            nn.init.xavier_uniform_(self.res_proj.weight, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features tensor of shape (B, N, in_features).
            adj: Binary adjacency matrix of shape (B, N, N) where adj[b, i, j] = 1 indicates edge j -> i.
        Returns:
            out: Updated node features of shape (B, N, total_out_features).
            attn_weights: Attention coefficients of shape (B, num_heads, N, N).
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.out_features_per_head

        # 1. Feature dropout
        h = self.feat_dropout(x)

        # 2. Linear transformations for source, destination, and values
        # Shape: (B, H, N, D)
        z_src = self.fc_src(h).reshape(B, N, H, D).transpose(1, 2)
        z_dst = self.fc_dst(h).reshape(B, N, H, D).transpose(1, 2)
        z_val = self.fc_val(h).reshape(B, N, H, D).transpose(1, 2)

        # 3. Dynamic Attention Score Calculation
        # Pairwise broadcast sum:
        # z_src: (B, H, N, 1, D)
        # z_dst: (B, H, 1, N, D)
        # z_sum: (B, H, N, N, D)
        z_sum = z_src.unsqueeze(3) + z_dst.unsqueeze(2)

        # Apply LeakyReLU non-linearity BEFORE inner product with attention vector
        z_act = self.leaky_relu(z_sum)

        # Inner product with learnable vector a:
        # (B, H, N, N, D) * (1, H, 1, 1, D) -> sum over D -> (B, H, N, N)
        e = (z_act * self.attn_vec).sum(dim=-1)

        # 4. Mask non-existent edges
        mask = (adj.unsqueeze(1) > 0)  # (B, 1, N, N)
        e = e.masked_fill(~mask, -1e9)

        # 5. Numerically stable softmax across neighbours (dim=-1)
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        attn_weights = alpha

        # 6. Attention dropout
        alpha_drop = self.attn_dropout(alpha)

        # 7. Aggregate values: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        out = torch.matmul(alpha_drop, z_val)

        # 8. Concatenate heads: (B, H, N, D) -> (B, N, H, D) -> (B, N, H * D)
        out = out.transpose(1, 2).reshape(B, N, H * D)

        # 9. Non-linearity, Residual Connection and LayerNorm
        out = self.activation(out)
        if self.residual:
            shortcut = self.res_proj(x)
            out = self.norm(out + shortcut)
        else:
            out = self.norm(out)

        return out, attn_weights


class GATv2(nn.Module):
    """
    2-Layer Graph Attention Network v2 (Dynamic Attention).
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
            GATv2Layer(
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
        Forward pass through all GATv2 layers.
        Returns:
            x: Updated node features of shape (B, N, 192).
            all_attn: List of attention weight tensors from each layer.
        """
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x, adj)
            all_attn.append(attn)
        return x, all_attn


def build_gatv2(
    in_features: int = 192,
    hidden_features_per_head: int = 48,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> GATv2:
    """Factory function for GATv2."""
    model = GATv2(
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
    gatv2 = build_gatv2()
    out, attns = gatv2(dummy_x, dummy_adj)
    print("GATv2 Output Shape:", out.shape, "(Expected: [2, 196, 192])")
    print(f"Number of GATv2 layers: {len(attns)}, Attn shape: {attns[0].shape}")
    assert out.shape == (2, 196, 192)
    assert len(attns) == 2
    assert attns[0].shape == (2, 4, 196, 196)
    print("GATv2 Architecture Verification PASSED!")
