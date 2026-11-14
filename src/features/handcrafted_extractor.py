"""
Unified Handcrafted Feature Extractor
=====================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Combines four distinct handcrafted feature extraction modules:
1. LBP (Local Binary Patterns): 59 features
2. GLCM (Gray-Level Co-occurrence Matrix): 24 features
3. Color (HSV + CIELAB moments and histograms): 46 features
4. Edge (Sobel gradient statistics and orientation histogram): 14 features
Total: 143 features.

Provides modular per-family extraction and structured dictionary outputs.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.features.lbp_features import extract_lbp_features
from src.features.glcm_features import extract_glcm_features
from src.features.color_features import extract_color_features
from src.features.edge_features import extract_edge_features

FEATURE_DIMS = {
    "lbp": 59,
    "glcm": 24,
    "color": 46,
    "edge": 14,
}
TOTAL_HANDCRAFTED_DIM = sum(FEATURE_DIMS.values())  # 143


class HandcraftedFeatureExtractor:
    """
    Unified extractor combining LBP, GLCM, Color, and Edge features.
    """
    def __init__(self, families: Optional[List[str]] = None):
        if families is None:
            self.families = ["lbp", "glcm", "color", "edge"]
        else:
            for f in families:
                if f not in FEATURE_DIMS:
                    raise ValueError(f"Unknown feature family '{f}'. Choose from {list(FEATURE_DIMS.keys())}")
            self.families = families

    @property
    def output_dim(self) -> int:
        return sum(FEATURE_DIMS[f] for f in self.families)

    def extract_dict(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Extracts features and returns a dictionary with keys 'lbp', 'glcm', 'color', 'edge'.
        """
        result = {}
        if "lbp" in self.families:
            result["lbp"] = extract_lbp_features(image)
        if "glcm" in self.families:
            result["glcm"] = extract_glcm_features(image)
        if "color" in self.families:
            result["color"] = extract_color_features(image)
        if "edge" in self.families:
            result["edge"] = extract_edge_features(image)
        return result

    def extract_vector(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extracts features and concatenates them in canonical order into a 1D float32 array.
        """
        res_dict = self.extract_dict(image)
        ordered_parts = [res_dict[f] for f in self.families]
        vec = np.concatenate(ordered_parts).astype(np.float32)
        assert len(vec) == self.output_dim, f"Expected {self.output_dim} dims, got {len(vec)}"
        return vec

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        return self.extract_vector(image)
