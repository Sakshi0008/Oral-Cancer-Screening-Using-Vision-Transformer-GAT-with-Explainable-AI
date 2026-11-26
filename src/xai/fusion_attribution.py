"""
Fusion-Branch Attribution via Integrated Gradients
==================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements axiomatic feature attribution through the Concatenation Fusion head using
Integrated Gradients (Sundararajan et al., 2017).
Decomposes joint representation z = [h_cls || h_graph] (384 dims) into:
- ViT visual representation contribution (dims 0..191)
- GATv2 relational graph representation contribution (dims 192..383)
Verifies mathematical completeness and computes signed and absolute importance shares.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn


class FusionIntegratedGradients:
    """
    Computes Integrated Gradients attribution across the two 192-dim branches of ConcatenationFusion.
    """
    def __init__(
        self,
        fusion_model: nn.Module,
        steps: int = 100,
        baseline: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            fusion_model: Instantiated ConcatenationFusion model.
            steps: Number of Riemann sum approximation steps along the straight-line path (default 100).
            baseline: Optional default baseline representation (default: neutral 0.05 vector).
            device: Compute device.
        """
        self.fusion = fusion_model
        self.steps = steps
        self.device = device or next(self.fusion.parameters()).device
        if baseline is not None:
            self.default_baseline = baseline.to(self.device)
        else:
            self.default_baseline = torch.full((1, 384), 0.05, device=self.device)

    def attribute(
        self,
        cls_rep: torch.Tensor,
        graph_rep: torch.Tensor,
        target_class: int = 1,
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Computes Integrated Gradients for the concatenated representation z = [cls_rep || graph_rep].

        Args:
            cls_rep: ViT CLS representation of shape (1, 192).
            graph_rep: GATv2 pooled graph representation of shape (1, 192).
            target_class: Class logit index for attribution (default 1 = CANCER).
            baseline: Optional baseline tensor of shape (1, 384). If None, zero baseline is used.

        Returns:
            Dictionary containing:
                - 'ig_vector': Full 384-dimensional Integrated Gradients attribution vector
                - 'vit_ig': ViT branch attribution vector (192 dims)
                - 'gat_ig': GATv2 branch attribution vector (192 dims)
                - 'vit_abs_sum': Absolute attribution sum for ViT branch
                - 'gat_abs_sum': Absolute attribution sum for GATv2 branch
                - 'vit_signed_sum': Signed attribution sum for ViT branch (net directional push)
                - 'gat_signed_sum': Signed attribution sum for GATv2 branch (net directional push)
                - 'vit_share_pct': Percentage share of total absolute attribution for ViT
                - 'gat_share_pct': Percentage share of total absolute attribution for GATv2
                - 'convergence_delta': Completeness check error |sum(IG) - (F(z) - F(baseline))|
                - 'convergence_error_pct': Relative completeness error percentage
                - 'target_class': Target class index
        """
        self.fusion.eval()
        cls_t = cls_rep.to(self.device).detach()
        graph_t = graph_rep.to(self.device).detach()
        if cls_t.dim() == 1:
            cls_t = cls_t.unsqueeze(0)
        if graph_t.dim() == 1:
            graph_t = graph_t.unsqueeze(0)

        # Joint input: z in R^(1, 384)
        z_input = torch.cat([cls_t, graph_t], dim=-1)

        # Baseline: default neutral baseline in representation space (avoids LayerNorm origin singularity)
        if baseline is None:
            z_base = self.default_baseline.clone()
        else:
            z_base = baseline.to(self.device).detach()
        if z_base.dim() == 1:
            z_base = z_base.unsqueeze(0)

        # Compute model output at input and baseline
        with torch.no_grad():
            f_input = self.fusion.classifier(z_input)[0, target_class].item()
            f_base = self.fusion.classifier(z_base)[0, target_class].item()
            delta_f = f_input - f_base

        # Scaled inputs along the straight-line path from baseline to input
        # alphas in [0, 1]
        alphas = torch.linspace(0.0, 1.0, self.steps, device=self.device)
        # Shape: (steps, 384)
        z_diff = z_input - z_base
        z_steps = z_base + alphas.unsqueeze(1) * z_diff
        z_steps.requires_grad_(True)

        # Forward pass on all steps
        logits_steps = self.fusion.classifier(z_steps)
        target_logits = logits_steps[:, target_class]

        # Backward pass to obtain gradients at each interpolation step
        self.fusion.zero_grad()
        grads = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=z_steps,
            retain_graph=False,
            create_graph=False
        )[0]  # Shape: (steps, 384)

        # Trapezoidal numerical integration along the alpha interpolation path
        avg_grads = torch.trapezoid(grads, alphas, dim=0).unsqueeze(0)  # (1, 384)

        # Integrated Gradients: (input - baseline) * integral_gradient
        ig = (z_diff * avg_grads).squeeze(0).cpu().numpy()  # (384,)

        # Completeness Check
        sum_ig = float(np.sum(ig))
        completeness_error = abs(sum_ig - delta_f)
        error_pct = (completeness_error / (abs(delta_f) + 1e-9)) * 100.0

        # Branch Partitioning
        vit_ig = ig[:192]
        gat_ig = ig[192:]

        vit_abs = float(np.sum(np.abs(vit_ig)))
        gat_abs = float(np.sum(np.abs(gat_ig)))
        total_abs = vit_abs + gat_abs + 1e-9

        vit_signed = float(np.sum(vit_ig))
        gat_signed = float(np.sum(gat_ig))

        vit_share = (vit_abs / total_abs) * 100.0
        gat_share = (gat_abs / total_abs) * 100.0

        return {
            "ig_vector": ig,
            "vit_ig": vit_ig,
            "gat_ig": gat_ig,
            "vit_abs_sum": vit_abs,
            "gat_abs_sum": gat_abs,
            "vit_signed_sum": vit_signed,
            "gat_signed_sum": gat_signed,
            "vit_share_pct": vit_share,
            "gat_share_pct": gat_share,
            "clean_logit": f_input,
            "baseline_logit": f_base,
            "delta_f": delta_f,
            "sum_ig": sum_ig,
            "convergence_delta": completeness_error,
            "convergence_error_pct": error_pct,
            "target_class": target_class
        }
