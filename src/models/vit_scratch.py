"""
Custom Vision Transformer (ViT) Architecture — From-Scratch Implementation
=============================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

STRICT CONSTRAINT:
- Initialized entirely from scratch with random parameters.
- NO pretrained weights, NO ImageNet checkpoints, NO timm/HuggingFace downloads.
- Clean interface exposing:
    1. Classification logits: [B, 2]
    2. CLS token representation: [B, 192]
    3. Patch-token representations: [B, 196, 192]

Specification:
- Image size: 224 x 224 x 3
- Patch size: 16 x 16 -> 196 patches (14 x 14 grid)
- Embedding dimension: 192
- Transformer encoder depth: 4 layers
- Attention heads: 3 (head dimension: 64)
- MLP hidden dimension: 768 (expansion ratio 4.0)
- Dropout: 0.1
- Classes: 2 (0: NON CANCER, 1: CANCER)
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Truncated normal initialization following standard ViT practice (DeiT/ViT).
    Values outside [mean - 2*std, mean + 2*std] are redrawn.
    """
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a * std, b=b * std)
    return tensor


class PatchEmbedding(nn.Module):
    """
    Projects 2D image into flattened 1D patch embeddings using 2D convolution.
    Input:  (B, 3, 224, 224)
    Output: (B, 196, 192)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        assert self.num_patches == 196, f"Expected 196 patches, got {self.num_patches}"

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size ({H}, {W}) != ({self.img_size}, {self.img_size})"
        # (B, C, H, W) -> (B, embed_dim, 14, 14) -> (B, embed_dim, 196) -> (B, 196, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module.
    embed_dim = 192, num_heads = 3 -> head_dim = 64
    """
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, N, C)
        Returns:
            out: Tensor of shape (B, N, C)
            attn_weights: Tensor of shape (B, num_heads, N, N)
        """
        B, N, C = x.shape
        # (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> permute to (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)

        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        # Transpose & reshape to (B, N, embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights


class MLPBlock(nn.Module):
    """
    Two-layer Feed-Forward Network with GELU activation and dropout.
    embed_dim (192) -> mlp_dim (768) -> embed_dim (192)
    """
    def __init__(
        self,
        embed_dim: int = 192,
        mlp_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block with residual connections.
    """
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        attn_drop: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self-Attention with residual connection
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x)
        x = x + attn_out

        # Pre-LN MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class VisionTransformerScratch(nn.Module):
    """
    Vision Transformer trained completely from scratch for binary oral cancer screening.
    Supports returning logits, CLS embedding, and patch representations.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        attn_drop: float = 0.1
    ):
        super().__init__()
        self.is_pretrained = False  # Strict verification flag
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depth = depth

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # 196

        # 2. Learnable CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attn_drop=attn_drop
            )
            for _ in range(depth)
        ])

        # 4. Final Normalization and Classification Head
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # 5. Initialize all weights randomly
        self._init_weights()

    def _init_weights(self):
        """
        Documented random weight initialization:
        - Linear & Conv2d weights: Truncated normal (mean=0.0, std=0.02)
        - Biases: Constant 0.0
        - LayerNorm weights: Constant 1.0, Biases: Constant 0.0
        - CLS token & Positional embeddings: Truncated normal (std=0.02)
        """
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                # Patch embedding conv layer
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor, return_all_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract token representations through patch embedding and transformer encoder.
        Returns:
            dict containing:
                "cls": [B, 192] (CLS token after encoder and final norm)
                "patches": [B, 196, 192] (196 patch tokens after encoder and final norm)
                "all_tokens": [B, 197, 192] (optional, all tokens including CLS)
        """
        B = x.shape[0]

        # Patch embedding: (B, 196, 192)
        x = self.patch_embed(x)

        # Expand CLS token to batch size: (B, 1, 192)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS token with patch tokens: (B, 197, 192)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer encoder blocks
        for block in self.blocks:
            x, _ = block(x)

        # Apply final LayerNorm
        x = self.norm(x)

        cls_rep = x[:, 0]        # (B, 192)
        patch_reps = x[:, 1:]    # (B, 196, 192)

        result = {
            "cls": cls_rep,
            "patches": patch_reps
        }
        if return_all_tokens:
            result["all_tokens"] = x
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for classification.
        Args:
            x: Input images of shape (B, 3, 224, 224)
        Returns:
            logits: Output classification logits of shape (B, 2)
        """
        features = self.forward_features(x)
        logits = self.head(features["cls"])
        return logits

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to extract patch tokens for graph construction.
        Returns:
            patches: Tensor of shape (B, 196, 192)
        """
        return self.forward_features(x)["patches"]

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to extract the CLS token representation.
        Returns:
            cls: Tensor of shape (B, 192)
        """
        return self.forward_features(x)["cls"]


def build_vit_scratch(
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 2,
    embed_dim: int = 192,
    depth: int = 4,
    num_heads: int = 3,
    mlp_dim: int = 768,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> VisionTransformerScratch:
    """
    Factory function for VisionTransformerScratch.
    """
    model = VisionTransformerScratch(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    print("=" * 65)
    print("Verifying Vision Transformer (From-Scratch) Architecture...")
    print("=" * 65)

    model = build_vit_scratch()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Is Pretrained:        {model.is_pretrained}")

    dummy_img = torch.randn(4, 3, 224, 224)
    logits = model(dummy_img)
    feats = model.forward_features(dummy_img, return_all_tokens=True)

    print(f"Input Tensor Shape:   {dummy_img.shape}")
    print(f"Logits Shape:         {logits.shape} (Expected: [4, 2])")
    print(f"CLS Token Shape:      {feats['cls'].shape} (Expected: [4, 192])")
    print(f"Patch Tokens Shape:   {feats['patches'].shape} (Expected: [4, 196, 192])")
    print(f"All Tokens Shape:     {feats['all_tokens'].shape} (Expected: [4, 197, 192])")

    assert logits.shape == (4, 2)
    assert feats["cls"].shape == (4, 192)
    assert feats["patches"].shape == (4, 196, 192)
    assert feats["all_tokens"].shape == (4, 197, 192)
    print("=" * 65)
    print("ViT Architecture Verification PASSED!")
    print("=" * 65)
