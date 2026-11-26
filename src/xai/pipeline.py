"""
Unified OralScan XAI Pipeline
=============================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Orchestrates the complete multimodal explainability pipeline for the primary
ViT + GATv2 Concatenation Fusion model.
Produces:
1. Predictions and class probabilities.
2. ViT Attention Rollout patch heatmaps.
3. GATv2 multi-head edge attention and node centrality heatmaps.
4. Topological graph overlays distinguishing spatial from similarity edges.
5. Fusion-level Integrated Gradients attribution.
6. Structured plain-language narrative explanations.
7. Publication-quality multi-panel visualization artifacts.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import get_transforms
from src.verify_hybrid_inference import VerifiedHybridInferencePipeline
from src.xai.vit_rollout import ViTAttentionRollout
from src.xai.vit_attribution import ViTPatchAttribution
from src.xai.gat_attribution import GATv2GraphAttribution
from src.xai.graph_visualizer import GraphOverlayVisualizer
from src.xai.fusion_attribution import FusionIntegratedGradients
from src.xai.text_generator import NarrativeExplanationGenerator
from src.xai.perturbation import ImagePerturbationValidator
from src.xai.visualizer_utils import MultimodalXAIVisualizer
from src.graph.graph_builder import build_image_graph


class OralScanXAIExplainer:
    """
    Unified Explainable AI Explainer for OralScan AI.
    """
    def __init__(
        self,
        inference_pipeline: Optional[VerifiedHybridInferencePipeline] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "reports/xai_examples"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = inference_pipeline or VerifiedHybridInferencePipeline(device=self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # XAI Submodules
        self.vit_rollout = ViTAttentionRollout(self.pipeline.vit)
        self.vit_attr = ViTPatchAttribution(self.pipeline.vit, device=self.device)
        self.gat_attr = GATv2GraphAttribution(self.pipeline.gatv2, include_self_loops=False, device=self.device)
        self.graph_viz = GraphOverlayVisualizer()
        self.fusion_ig = FusionIntegratedGradients(self.pipeline.fusion, steps=50, device=self.device)
        self.narrative_gen = NarrativeExplanationGenerator()
        self.perturb_val = ImagePerturbationValidator(self.pipeline, device=self.device)

    def explain_image(
        self,
        image_path: str,
        case_id: Optional[str] = None,
        run_perturbation: bool = False,
        save_viz: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal screening and generates comprehensive XAI explanations.

        Args:
            image_path: Path to oral lesion image.
            case_id: Optional unique identifier for logging and saving.
            run_perturbation: Whether to run deletion curve validation (adds ~1 sec).
            save_viz: Whether to save multi-panel visualization PNG.

        Returns:
            Dictionary containing prediction, XAI heatmaps, graph overlays, fusion IG, narrative, and latency.
        """
        cid = case_id or Path(image_path).stem
        timings = {}

        # 1. Load and Preprocess Image
        t0 = time.perf_counter()
        raw_pil = Image.open(image_path).convert("RGB")
        img_tensor = self.pipeline.transform(raw_pil).unsqueeze(0).to(self.device)
        timings["io_and_transform_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Forward Pass through ViT + GATv2 + Fusion
        t1 = time.perf_counter()
        with torch.no_grad():
            vit_feats = self.pipeline.vit.forward_features(img_tensor)
            cls_token = vit_feats["cls"]
            patch_tokens = vit_feats["patches"]

            adj = build_image_graph("hybrid", patch_tokens, k=8, add_self_loops=True)
            graph_rep = self.pipeline.gatv2.get_graph_representation(patch_tokens, adj)

            logits, fused_rep = self.pipeline.fusion(cls_token, graph_rep)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        cancer_prob = float(probs[1])
        pred_label = 1 if cancer_prob >= 0.5 else 0
        pred_name = "CANCER" if pred_label == 1 else "NON CANCER"
        confidence = float(max(probs)) * 100.0
        timings["forward_inference_ms"] = (time.perf_counter() - t1) * 1000.0

        # 3. ViT Attention Rollout
        t2 = time.perf_counter()
        rollout_res = self.vit_rollout.compute_rollout(img_tensor, device=self.device)
        vit_grid = rollout_res["patch_grid_14x14"][0].cpu().numpy()
        vit_heatmap = rollout_res["heatmap_224x224"][0].cpu().numpy()
        timings["vit_rollout_ms"] = (time.perf_counter() - t2) * 1000.0

        # 4. GATv2 Graph Attention & Node Centrality
        t3 = time.perf_counter()
        gat_res = self.gat_attr.extract_graph_attention(patch_tokens, adj=adj, top_k_edges=30)
        gat_grid = gat_res["node_contribution_14x14"]
        gat_heatmap = gat_res["node_heatmap_224x224"]
        top_edges = gat_res["top_edges"]

        # Render Graph Overlay
        graph_overlay_rgb = self.graph_viz.render_graph_overlay(
            base_image=raw_pil,
            top_edges=top_edges,
            node_centrality=gat_res["node_centrality_vector"]
        )
        timings["gat_attribution_ms"] = (time.perf_counter() - t3) * 1000.0

        # 5. Fusion Integrated Gradients Attribution
        t4 = time.perf_counter()
        fusion_res = self.fusion_ig.attribute(
            cls_rep=cls_token,
            graph_rep=graph_rep,
            target_class=pred_label
        )
        timings["fusion_ig_ms"] = (time.perf_counter() - t4) * 1000.0

        # 6. Narrative Plain-Language Explanation
        t5 = time.perf_counter()
        narrative_res = self.narrative_gen.generate_explanation(
            pred_label=pred_label,
            pred_name=pred_name,
            cancer_prob=cancer_prob,
            vit_grid_14x14=vit_grid,
            gat_grid_14x14=gat_grid,
            fusion_attr=fusion_res
        )
        timings["narrative_gen_ms"] = (time.perf_counter() - t5) * 1000.0

        # 7. Optional Perturbation Deletion Curve
        perturbation_res = None
        if run_perturbation:
            t6 = time.perf_counter()
            perturbation_res = self.perturb_val.run_deletion_test(
                img_tensor=img_tensor,
                attribution_grid_14x14=vit_grid,
                target_class=pred_label
            )
            timings["perturbation_test_ms"] = (time.perf_counter() - t6) * 1000.0

        # 8. Render Multi-Panel Visualization
        viz_path = None
        if save_viz:
            t7 = time.perf_counter()
            viz_filename = f"{cid}_xai_explanation.png"
            viz_path = str(self.output_dir / viz_filename)
            MultimodalXAIVisualizer.render_comprehensive_figure(
                original_img=raw_pil,
                vit_heatmap_224=vit_heatmap,
                gat_heatmap_224=gat_heatmap,
                graph_overlay_rgb=graph_overlay_rgb,
                fusion_attr=fusion_res,
                pred_label=pred_label,
                pred_name=pred_name,
                cancer_prob=cancer_prob,
                save_path=viz_path
            )
            timings["viz_render_ms"] = (time.perf_counter() - t7) * 1000.0

        timings["total_xai_pipeline_ms"] = sum(timings.values())

        # Compile edge summary statistics
        edge_counts = {"spatial_grid": 0, "feature_similarity": 0, "shared_hybrid": 0, "other": 0}
        for e in top_edges:
            etype = e.get("edge_type", "other")
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "case_id": cid,
            "image_path": image_path,
            "prediction": {
                "pred_label": pred_label,
                "pred_name": pred_name,
                "cancer_prob": cancer_prob,
                "confidence": confidence,
                "logits": logits[0].cpu().numpy().tolist(),
                "probs": probs.tolist()
            },
            "vit_rollout": {
                "patch_grid_14x14": vit_grid,
                "heatmap_224x224": vit_heatmap,
                "top_k_patches": rollout_res["top_k_patches"]
            },
            "gatv2_graph": {
                "node_contribution_14x14": gat_grid,
                "node_heatmap_224x224": gat_heatmap,
                "top_edges": top_edges,
                "edge_type_distribution": edge_counts,
                "self_loops_included": gat_res["self_loops_included"]
            },
            "fusion_attribution": {
                "vit_share_pct": fusion_res["vit_share_pct"],
                "gat_share_pct": fusion_res["gat_share_pct"],
                "vit_signed_sum": fusion_res["vit_signed_sum"],
                "gat_signed_sum": fusion_res["gat_signed_sum"],
                "convergence_error_pct": fusion_res["convergence_error_pct"],
                "clean_logit": fusion_res["clean_logit"]
            },
            "narrative_explanation": narrative_res,
            "perturbation_validation": perturbation_res,
            "visualization_path": viz_path,
            "timings_ms": timings
        }
