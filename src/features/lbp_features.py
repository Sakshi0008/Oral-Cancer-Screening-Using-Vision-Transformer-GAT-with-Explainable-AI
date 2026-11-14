"""
Local Binary Pattern (LBP) Texture Feature Extractor
===================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts uniform Local Binary Patterns (LBP) from oral mucosal images.
Parameters:
- P = 8 neighbours, R = 1.0 radius.
- Uniform LBP: Patterns with at most 2 bit transitions (0->1 or 1->0).
- 58 uniform patterns + 1 non-uniform pattern = 59 histogram bins.
- Normalized histogram (sums to 1.0) providing illumination-invariant texture representation.
"""

from typing import Dict, Tuple, Union
import numpy as np
from PIL import Image


def _get_uniform_lbp_lookup() -> np.ndarray:
    """
    Constructs a lookup table mapping all 256 8-bit patterns to 59 uniform LBP bins:
    - 58 uniform patterns mapped to indices 0..57.
    - All non-uniform patterns (>2 transitions) mapped to index 58.
    """
    lookup = np.zeros(256, dtype=np.int32)
    current_uniform_idx = 0

    for byte_val in range(256):
        # Count bit transitions in circular byte
        binary_str = format(byte_val, "08b")
        transitions = sum(binary_str[i] != binary_str[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lookup[byte_val] = current_uniform_idx
            current_uniform_idx += 1
        else:
            lookup[byte_val] = 58  # Non-uniform bin

    assert current_uniform_idx == 58, f"Expected 58 uniform patterns, found {current_uniform_idx}"
    return lookup


_UNIFORM_LBP_LOOKUP = _get_uniform_lbp_lookup()


def compute_lbp_image(gray_img: np.ndarray) -> np.ndarray:
    """
    Computes P=8, R=1 LBP code for each internal pixel of a 2D grayscale image.
    Uses 8-neighbour offsets:
        (-1, -1), (-1, 0), (-1, 1),
        ( 0,  1), ( 1, 1), ( 1, 0),
        ( 1, -1), ( 0, -1)
    """
    H, W = gray_img.shape
    assert H >= 3 and W >= 3, "Image dimensions must be at least 3x3"

    center = gray_img[1:-1, 1:-1]
    lbp_codes = np.zeros((H - 2, W - 2), dtype=np.uint8)

    # 8 neighbor offsets and bit weights
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0,  1), ( 1,  1), ( 1, 0),
        ( 1, -1), ( 0, -1)
    ]

    for bit_idx, (dr, dc) in enumerate(neighbor_offsets):
        nbr = gray_img[1 + dr : H - 1 + dr, 1 + dc : W - 1 + dc]
        bit_val = (nbr >= center).astype(np.uint8)
        lbp_codes |= (bit_val << bit_idx)

    return lbp_codes


def extract_lbp_features(
    image: Union[np.ndarray, Image.Image],
    num_bins: int = 59
) -> np.ndarray:
    """
    Extracts normalized 59-bin uniform LBP histogram from an RGB or grayscale image.

    Args:
        image: RGB or Grayscale image (numpy array or PIL Image).
        num_bins: Expected number of bins (59 for P=8 uniform LBP).

    Returns:
        hist: 59-dimensional float32 feature vector normalized to unit sum.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if image.ndim == 3:
        # Convert RGB to grayscale using standard luminance formula
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)

    gray = gray.astype(np.float32)

    # 1. Compute 8-bit LBP code per pixel
    lbp_codes = compute_lbp_image(gray)

    # 2. Map 256 codes to 59 uniform bins via lookup table
    mapped_codes = _UNIFORM_LBP_LOOKUP[lbp_codes]

    # 3. Compute histogram across the 59 bins
    hist, _ = np.histogram(mapped_codes, bins=np.arange(num_bins + 1))
    hist = hist.astype(np.float32)

    # 4. L1 Normalize to sum to 1.0 (illumination & size invariance)
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist /= total_pixels

    assert hist.shape == (59,), f"Expected 59 features, got {hist.shape}"
    return hist


if __name__ == "__main__":
    dummy_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    lbp_feats = extract_lbp_features(dummy_rgb)
    print("LBP Features shape:", lbp_feats.shape)
    print("LBP sum:", lbp_feats.sum())
    assert lbp_feats.shape == (59,)
    assert np.isclose(lbp_feats.sum(), 1.0)
    print("LBP Feature Extractor Verification PASSED!")
