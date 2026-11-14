"""
Unit Tests for Handcrafted Feature Extraction and Scaler
========================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from src.features.lbp_features import extract_lbp_features, compute_lbp_image
from src.features.glcm_features import extract_glcm_features, quantize_grayscale
from src.features.color_features import extract_color_features
from src.features.edge_features import extract_edge_features
from src.features.handcrafted_extractor import (
    HandcraftedFeatureExtractor,
    FEATURE_DIMS,
    TOTAL_HANDCRAFTED_DIM
)
from src.features.feature_scaler import TrainingFeatureScaler


class TestHandcraftedFeatures(unittest.TestCase):
    def setUp(self):
        # Create deterministic synthetic test image (224 x 224 x 3)
        np.random.seed(42)
        self.test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        self.test_pil = Image.fromarray(self.test_img)

    def test_lbp_extractor_dimensions(self):
        lbp_vec = extract_lbp_features(self.test_img)
        self.assertEqual(len(lbp_vec), 59)
        self.assertEqual(lbp_vec.dtype, np.float32)
        # Histogram should sum to 1.0
        self.assertAlmostEqual(float(np.sum(lbp_vec)), 1.0, places=4)
        self.assertFalse(np.isnan(lbp_vec).any())

    def test_glcm_extractor_dimensions(self):
        glcm_vec = extract_glcm_features(self.test_img)
        self.assertEqual(len(glcm_vec), 24)
        self.assertEqual(glcm_vec.dtype, np.float32)
        self.assertFalse(np.isnan(glcm_vec).any())
        self.assertFalse(np.isinf(glcm_vec).any())

    def test_color_extractor_dimensions(self):
        color_vec = extract_color_features(self.test_img)
        self.assertEqual(len(color_vec), 46)
        self.assertEqual(color_vec.dtype, np.float32)
        self.assertFalse(np.isnan(color_vec).any())

    def test_edge_extractor_dimensions(self):
        edge_vec = extract_edge_features(self.test_img)
        self.assertEqual(len(edge_vec), 14)
        self.assertEqual(edge_vec.dtype, np.float32)
        self.assertFalse(np.isnan(edge_vec).any())

    def test_unified_extractor_dimensions_and_consistency(self):
        extractor = HandcraftedFeatureExtractor()
        self.assertEqual(extractor.output_dim, 143)

        vec1 = extractor.extract_vector(self.test_img)
        vec2 = extractor.extract_vector(self.test_pil)
        self.assertEqual(len(vec1), 143)
        self.assertEqual(len(vec2), 143)
        # Output should be identical whether passing numpy or PIL
        np.testing.assert_allclose(vec1, vec2, atol=1e-5)

    def test_feature_scaler_training_only_enforcement(self):
        scaler = TrainingFeatureScaler()
        dummy_data = np.random.randn(10, 143)

        # Attempting to fit on 'val' or 'test' must raise ValueError
        with self.assertRaises(ValueError):
            scaler.fit(dummy_data, split_name="val")

        with self.assertRaises(ValueError):
            scaler.fit(dummy_data, split_name="test")

        # Fitting on 'train' must succeed
        scaler.fit(dummy_data, split_name="train")
        self.assertTrue(scaler.is_fitted)

        transformed = scaler.transform(dummy_data)
        self.assertEqual(transformed.shape, (10, 143))
        # Mean of transformed training data should be ~0, std should be ~1
        np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(transformed, axis=0), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
