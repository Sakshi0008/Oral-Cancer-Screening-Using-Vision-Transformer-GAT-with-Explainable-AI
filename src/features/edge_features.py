"""
Edge and Gradient Feature Extractor
==================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts 14 edge and gradient features using Sobel operators:
1. Magnitude mean (1 feature)
2. Magnitude std (1 feature)
3. Magnitude skewness (1 feature)
4. Magnitude kurtosis (1 feature)
5. Edge density at threshold T1 (pixels with magnitude > mean(magnitude)) (1 feature)
6. Edge density at threshold T2 (pixels with magnitude > mean(magnitude) + std(magnitude)) (1 feature)
7. Magnitude-weighted orientation histogram with 8 bins over [0 deg, 180 deg) (8 features)
Total: 14 features.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def extract_edge_features(
    image: Union[np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Extracts 14-dimensional edge and gradient statistics and orientation histogram.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    # 1. Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 2. Magnitude and orientation
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Orientation in radians [-pi, pi] -> degrees [0, 180)
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.rad2deg(angle_rad) % 180.0

    mag_flat = mag.ravel()
    mean_mag = float(np.mean(mag_flat))
    std_mag = float(np.std(mag_flat))

    if std_mag < 1e-6:
        skew_mag = 0.0
        kurt_mag = 0.0
    else:
        normed = (mag_flat - mean_mag) / std_mag
        skew_mag = float(np.mean(normed ** 3))
        kurt_mag = float(np.mean(normed ** 4) - 3.0)  # Excess kurtosis

    # Edge densities
    t1 = mean_mag
    t2 = mean_mag + std_mag
    density_t1 = float(np.mean(mag_flat > t1))
    density_t2 = float(np.mean(mag_flat > t2))

    # Magnitude-weighted orientation histogram (8 bins over 0..180)
    hist, _ = np.histogram(
        angle_deg.ravel(),
        bins=8,
        range=(0.0, 180.0),
        weights=mag_flat
    )
    hist_sum = np.sum(hist)
    if hist_sum > 1e-9:
        hist_norm = hist / hist_sum
    else:
        hist_norm = np.ones(8, dtype=np.float32) / 8.0

    features = [
        mean_mag,
        std_mag,
        skew_mag,
        kurt_mag,
        density_t1,
        density_t2,
    ] + hist_norm.tolist()

    out = np.array(features, dtype=np.float32)
    assert len(out) == 14, f"Expected 14 edge features, got {len(out)}"
    return out
