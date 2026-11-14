"""
Color Feature Extractor (HSV and CIELAB Color Spaces)
===================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts color features from HSV and CIELAB color spaces:
- HSV:
  - 1st, 2nd, and 3rd moments (mean, std, skewness) per channel (H, S, V) = 9 features.
  - Normalized 1D histograms: H (8 bins), S (4 bins), V (4 bins) = 16 features.
  - Subtotal: 25 features.
- CIELAB:
  - 1st, 2nd, and 3rd moments (mean, std, skewness) per channel (L*, a*, b*) = 9 features.
  - Normalized 1D histograms: L* (4 bins), a* (4 bins), b* (4 bins) = 12 features.
  - Subtotal: 21 features.
Total: 25 + 21 = 46 features.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def _compute_channel_moments(channel: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes mean, standard deviation, and sample skewness for a single 2D channel.
    Safe against division by zero.
    """
    vals = channel.ravel().astype(np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std < 1e-6:
        skewness = 0.0
    else:
        # Fisher-Pearson skewness: E[((x - mu)/sigma)^3]
        normed = (vals - mean) / std
        skewness = float(np.mean(normed ** 3))
    return mean, std, skewness


def extract_color_features(
    image: Union[np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Extracts 46-dimensional color features (moments and compact histograms) in HSV and CIELAB spaces.
    Input image must be RGB.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Color feature extraction requires a 3-channel RGB image, got shape {image.shape}")

    # Ensure uint8 RGB
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            rgb_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        rgb_uint8 = image

    features = []

    # ---------------- 1. HSV Color Space ----------------
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    h_chan = hsv[:, :, 0]  # [0..179]
    s_chan = hsv[:, :, 1]  # [0..255]
    v_chan = hsv[:, :, 2]  # [0..255]

    # Moments (H, S, V) -> 9 features
    for ch in [h_chan, s_chan, v_chan]:
        m, s, sk = _compute_channel_moments(ch)
        features.extend([m, s, sk])

    # Histograms (H: 8 bins, S: 4 bins, V: 4 bins) -> 16 features
    h_hist, _ = np.histogram(h_chan, bins=8, range=(0, 180), density=True)
    h_hist_norm = h_hist / (np.sum(h_hist) + 1e-9)
    features.extend(h_hist_norm.tolist())

    s_hist, _ = np.histogram(s_chan, bins=4, range=(0, 256), density=True)
    s_hist_norm = s_hist / (np.sum(s_hist) + 1e-9)
    features.extend(s_hist_norm.tolist())

    v_hist, _ = np.histogram(v_chan, bins=4, range=(0, 256), density=True)
    v_hist_norm = v_hist / (np.sum(v_hist) + 1e-9)
    features.extend(v_hist_norm.tolist())

    # ---------------- 2. CIELAB Color Space ----------------
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]  # [0..255]
    a_chan = lab[:, :, 1]  # [0..255]
    b_chan = lab[:, :, 2]  # [0..255]

    # Moments (L, a, b) -> 9 features
    for ch in [l_chan, a_chan, b_chan]:
        m, s, sk = _compute_channel_moments(ch)
        features.extend([m, s, sk])

    # Histograms (L: 4 bins, a: 4 bins, b: 4 bins) -> 12 features
    for ch in [l_chan, a_chan, b_chan]:
        hist, _ = np.histogram(ch, bins=4, range=(0, 256), density=True)
        hist_norm = hist / (np.sum(hist) + 1e-9)
        features.extend(hist_norm.tolist())

    out = np.array(features, dtype=np.float32)
    assert len(out) == 46, f"Expected 46 color features, got {len(out)}"
    return out
