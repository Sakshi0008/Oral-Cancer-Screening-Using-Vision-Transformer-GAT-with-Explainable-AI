"""
Gray-Level Co-occurrence Matrix (GLCM) Texture Feature Extractor
==============================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts Gray-Level Co-occurrence Matrix (GLCM) texture features:
- Grayscale quantization to 16 discrete levels [0..15].
- Offsets: Distances d in {1, 2}, 4 angles {0 deg, 45 deg, 90 deg, 135 deg}.
- Symmetric and normalized co-occurrence matrices.
- 6 Haralick texture properties:
  1. Contrast: sum_{i,j} |i - j|^2 * P(i,j)
  2. Dissimilarity: sum_{i,j} |i - j| * P(i,j)
  3. Homogeneity (Inverse Difference Moment): sum_{i,j} P(i,j) / (1 + (i - j)^2)
  4. Energy: sqrt(sum_{i,j} P(i,j)^2)
  5. Correlation: sum_{i,j} ((i - mu_i)*(j - mu_j)*P(i,j)) / (sigma_i * sigma_j + eps)
  6. Angular Second Moment (ASM): sum_{i,j} P(i,j)^2
- For each distance d in {1, 2}, computes mean and standard deviation across the 4 angles.
  Total features: 2 distances * 6 properties * 2 stats (mean, std) = 24 features.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image


def quantize_grayscale(gray: np.ndarray, num_levels: int = 16) -> np.ndarray:
    """
    Quantizes a floating-point or uint8 grayscale image into `num_levels` discrete bins [0..num_levels-1].
    """
    gray = gray.astype(np.float32)
    min_val, max_val = float(gray.min()), float(gray.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(gray, dtype=np.int32)
    normalized = (gray - min_val) / (max_val - min_val)
    quantized = np.clip(np.floor(normalized * num_levels).astype(np.int32), 0, num_levels - 1)
    return quantized


def compute_single_glcm(
    quantized_img: np.ndarray,
    dr: int,
    dc: int,
    num_levels: int = 16
) -> np.ndarray:
    """
    Computes a single normalized, symmetric GLCM for a given displacement (dr, dc).
    """
    H, W = quantized_img.shape
    r_start = max(0, -dr)
    r_end = min(H, H - dr)
    c_start = max(0, -dc)
    c_end = min(W, W - dc)

    if r_end <= r_start or c_end <= c_start:
        glcm = np.zeros((num_levels, num_levels), dtype=np.float64)
        return glcm

    i_coords = quantized_img[r_start:r_end, c_start:c_end]
    j_coords = quantized_img[r_start + dr : r_end + dr, c_start + dc : c_end + dc]

    linear_indices = i_coords.ravel() * num_levels + j_coords.ravel()
    bincount = np.bincount(linear_indices, minlength=num_levels * num_levels)
    glcm = bincount.reshape((num_levels, num_levels)).astype(np.float64)

    # Symmetrize: pairs (i, j) and (j, i) both counted
    glcm = glcm + glcm.T

    # Normalize to probability distribution
    total = glcm.sum()
    if total > 0:
        glcm /= total
    else:
        glcm = np.ones((num_levels, num_levels), dtype=np.float64) / (num_levels * num_levels)

    return glcm


def calculate_glcm_properties(glcm: np.ndarray) -> np.ndarray:
    """
    Calculates 6 Haralick properties from a normalized symmetric GLCM (num_levels x num_levels):
    [contrast, dissimilarity, homogeneity, energy, correlation, asm]
    """
    num_levels = glcm.shape[0]
    i_indices, j_indices = np.indices((num_levels, num_levels), dtype=np.float64)
    diff = i_indices - j_indices
    abs_diff = np.abs(diff)

    # 1. Contrast
    contrast = np.sum((diff ** 2) * glcm)

    # 2. Dissimilarity
    dissimilarity = np.sum(abs_diff * glcm)

    # 3. Homogeneity (Inverse Difference Moment)
    homogeneity = np.sum(glcm / (1.0 + (diff ** 2)))

    # 4. Angular Second Moment (ASM)
    asm = np.sum(glcm ** 2)

    # 5. Energy
    energy = np.sqrt(asm)

    # 6. Correlation
    mu_i = np.sum(i_indices * glcm)
    mu_j = np.sum(j_indices * glcm)
    var_i = np.sum(((i_indices - mu_i) ** 2) * glcm)
    var_j = np.sum(((j_indices - mu_j) ** 2) * glcm)
    std_i = np.sqrt(max(0.0, var_i))
    std_j = np.sqrt(max(0.0, var_j))

    if std_i * std_j > 1e-9:
        cov = np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm)
        correlation = cov / (std_i * std_j)
    else:
        correlation = 1.0

    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm], dtype=np.float32)


def extract_glcm_features(
    image: Union[np.ndarray, Image.Image],
    num_levels: int = 16,
    distances: Tuple[int, ...] = (1, 2)
) -> np.ndarray:
    """
    Extracts 24-dimensional rotational-invariant GLCM texture features from an RGB or Grayscale image.
    For each distance d in {1, 2}:
      - Computes GLCM for 4 directions: 0 deg (0, d), 45 deg (-d, d), 90 deg (-d, 0), 135 deg (-d, -d).
      - Computes 6 Haralick properties for each direction.
      - Calculates mean and std across the 4 directions -> 12 features per distance.
    Total: 2 * 12 = 24 features.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)

    quantized = quantize_grayscale(gray, num_levels=num_levels)

    angle_offsets = {
        0: (0, 1),
        45: (-1, 1),
        90: (-1, 0),
        135: (-1, -1),
    }

    feature_list = []
    for d in distances:
        props_by_angle = []
        for angle, (dr_base, dc_base) in angle_offsets.items():
            dr = dr_base * d
            dc = dc_base * d
            glcm = compute_single_glcm(quantized, dr, dc, num_levels=num_levels)
            props = calculate_glcm_properties(glcm)
            props_by_angle.append(props)

        # shape: (4, 6)
        props_mat = np.array(props_by_angle, dtype=np.float32)
        mean_props = np.mean(props_mat, axis=0)  # (6,)
        std_props = np.std(props_mat, axis=0)   # (6,)

        feature_list.extend(mean_props.tolist())
        feature_list.extend(std_props.tolist())

    out = np.array(feature_list, dtype=np.float32)
    assert len(out) == 24, f"Expected 24 GLCM features, got {len(out)}"
    return out
