"""
EfficientNet-B0 Baseline Model Architecture (From-Scratch Binary Classifier)
============================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

STRICT CONSTRAINT:
- Initialized strictly with weights=None (NO pretrained weights, NO ImageNet checkpoints).
- All convolutional, batch-norm, and linear layers are initialized randomly.
- Binary classification head producing 2 output logits:
    Index 0: NON CANCER
    Index 1: CANCER
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0Scratch(nn.Module):
    """
    EfficientNet-B0 configured for from-scratch training on binary oral cancer screening.
    """
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.2,
        init_head: str = "kaiming"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.is_pretrained = False  # Explicit verification flag

        # STRICT: weights=None explicitly guarantees no pretrained parameters are loaded
        self.backbone = efficientnet_b0(weights=None)

        # In EfficientNet-B0, backbone.classifier is:
        # Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        in_features = self.backbone.classifier[1].in_features
        assert in_features == 1280, f"Expected 1280 in_features, got {in_features}"

        # Replace 1000-class head with binary classification head
        new_classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        )

        # Initialize the newly created linear layer
        linear_layer = new_classifier[1]
        if init_head == "kaiming":
            nn.init.kaiming_normal_(linear_layer.weight, mode="fan_out", nonlinearity="relu")
        elif init_head == "xavier":
            nn.init.xavier_uniform_(linear_layer.weight)
        else:
            nn.init.normal_(linear_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(linear_layer.bias, 0.0)

        self.backbone.classifier = new_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, 224, 224).
        Returns:
            logits: Output tensor of shape (B, 2).
        """
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature representation (B, 1280) before classification head.
        """
        feat = self.backbone.features(x)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat


def build_efficientnet_b0(
    num_classes: int = 2,
    dropout: float = 0.2,
    device: Optional[torch.device] = None
) -> EfficientNetB0Scratch:
    """
    Factory function ensuring strict from-scratch initialization.
    """
    model = EfficientNetB0Scratch(num_classes=num_classes, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Verifying EfficientNet-B0 Scratch Architecture...")
    print("=" * 60)
    model = build_efficientnet_b0()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Is Pretrained:        {model.is_pretrained}")

    dummy_input = torch.randn(4, 3, 224, 224)
    logits = model(dummy_input)
    print(f"Input Shape:          {dummy_input.shape}")
    print(f"Output Shape:         {logits.shape} (Expected: [4, 2])")
    assert logits.shape == (4, 2), "Shape mismatch!"
    print("Architecture verification PASSED!")
    print("=" * 60)
