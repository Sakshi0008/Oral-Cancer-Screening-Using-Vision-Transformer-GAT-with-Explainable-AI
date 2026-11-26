"""
Explainable AI (XAI) Module for Oral Cancer Screening
=====================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Provides unified interpretability across:
1. ViT attention rollout and patch attribution.
2. GATv2 graph attention and topological edge contribution.
3. Fusion-level Integrated Gradients attribution.
4. Quantitative image perturbation and deletion validation.
5. Plain-language clinical explanation generation.
"""

from src.xai.vit_rollout import ViTAttentionRollout
from src.xai.vit_attribution import ViTPatchAttribution
from src.xai.gat_attribution import GATv2GraphAttribution
from src.xai.fusion_attribution import FusionIntegratedGradients
from src.xai.perturbation import ImagePerturbationValidator
from src.xai.text_generator import NarrativeExplanationGenerator
from src.xai.pipeline import OralScanXAIExplainer

__all__ = [
    "ViTAttentionRollout",
    "ViTPatchAttribution",
    "GATv2GraphAttribution",
    "FusionIntegratedGradients",
    "ImagePerturbationValidator",
    "NarrativeExplanationGenerator",
    "OralScanXAIExplainer",
]
