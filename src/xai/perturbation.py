"""
Quantitative Image Perturbation and Deletion Curve Module
=========================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Implements controlled patch deletion and insertion benchmark tests (Petsiuk et al., 2018)
to evaluate explanation faithfulness:
- Compares deletion of High-Attribution patches vs Random patches vs Low-Attribution patches.
- Measures the drop in cancer logit and cancer softmax probability.
- Computes Area Under the Deletion Curve (AUDC).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.graph_builder import build_image_graph


class ImagePerturbationValidator:
    """
    Evaluates attribution faithfulness by measuring end-to-end model response
    under systematic patch deletion and insertion curves.
    """
    def __init__(
        self,
        inference_pipeline: Any,
        patch_size: int = 16,
        fill_value: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            inference_pipeline: VerifiedHybridInferencePipeline or OralScanXAIExplainer.
            patch_size: Size of square patches (16x16 pixels).
            fill_value: Replacement pixel value in normalized tensor space (0.0 = dataset primary mean).
            device: Compute device.
        """
        self.pipeline = inference_pipeline
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.device = device or next(self.pipeline.vit.parameters()).device

    def run_deletion_test(
        self,
        img_tensor: torch.Tensor,
        attribution_grid_14x14: np.ndarray,
        step_counts: Optional[List[int]] = None,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Runs systematic patch deletion test comparing:
        1. High Attribution First (MIF)
        2. Random Patch Order (Random Baseline)
        3. Low Attribution First (LIF)

        Args:
            img_tensor: Clean input image tensor of shape (1, 3, 224, 224).
            attribution_grid_14x14: 2D numpy array [14, 14] of patch importance scores.
            step_counts: List of cumulative patch counts to delete. Default: [0, 10, 20, 40, 70, 100, 140, 180, 196].
            target_class: Target class logit (default 1 = CANCER).

        Returns:
            Dictionary containing deletion curves, AUDC metrics, and faithfulness ratios.
        """
        if step_counts is None:
            step_counts = [0, 10, 20, 40, 70, 100, 140, 180, 196]

        img_t = img_tensor.to(self.device)
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)

        flat_scores = attribution_grid_14x14.ravel()
        num_patches = len(flat_scores)  # 196

        # Order 1: High attribution first (descending)
        high_order = np.argsort(flat_scores)[::-1]
        # Order 2: Low attribution first (ascending)
        low_order = np.argsort(flat_scores)
        # Order 3: Fixed random order
        rng = np.random.RandomState(42)
        rand_order = rng.permutation(num_patches)

        orders = {
            "high_attribution": high_order,
            "random_baseline": rand_order,
            "low_attribution": low_order
        }

        results = {
            "steps": step_counts,
            "fractions": [k / num_patches for k in step_counts],
            "curves_logit": {"high_attribution": [], "random_baseline": [], "low_attribution": []},
            "curves_prob": {"high_attribution": [], "random_baseline": [], "low_attribution": []}
        }

        for order_name, patch_order in orders.items():
            for k in step_counts:
                # Mask top-k patches in order
                masked_img = img_t.clone()
                patches_to_mask = patch_order[:k]
                for p_idx in patches_to_mask:
                    r = p_idx // 14
                    c = p_idx % 14
                    r_start, r_end = r * self.patch_size, (r + 1) * self.patch_size
                    c_start, c_end = c * self.patch_size, (c + 1) * self.patch_size
                    masked_img[:, :, r_start:r_end, c_start:c_end] = self.fill_value

                # End-to-end forward pass
                with torch.no_grad():
                    vit_feats = self.pipeline.vit.forward_features(masked_img)
                    cls_tok = vit_feats["cls"]
                    ptoks = vit_feats["patches"]

                    adj = build_image_graph("hybrid", ptoks, k=8, add_self_loops=True)
                    grep = self.pipeline.gatv2.get_graph_representation(ptoks, adj)

                    logits, _ = self.pipeline.fusion(cls_tok, grep)
                    probs = F.softmax(logits, dim=-1)

                    logit_val = float(logits[0, target_class].item())
                    prob_val = float(probs[0, target_class].item())

                results["curves_logit"][order_name].append(logit_val)
                results["curves_prob"][order_name].append(prob_val)

        # Compute Area Under Deletion Curve (AUDC) using trapezoidal rule for probability
        fractions = results["fractions"]
        audc_prob = {}
        for name in orders.keys():
            audc_prob[name] = float(np.trapezoid(results["curves_prob"][name], fractions))

        results["audc_prob"] = audc_prob

        # Faithfulness check: high attribution deletion AUDC should be significantly LOWER than random and low
        # (meaning probability drops much faster when deleting important features)
        is_faithful = bool(audc_prob["high_attribution"] < audc_prob["random_baseline"])
        results["is_faithful"] = is_faithful
        results["faithfulness_margin"] = float(audc_prob["random_baseline"] - audc_prob["high_attribution"])

        return results
