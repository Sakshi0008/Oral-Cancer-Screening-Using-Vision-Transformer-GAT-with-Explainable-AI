"""
Extract and Scale Handcrafted Image Features
============================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts 143-dimensional handcrafted features (LBP, GLCM, Color, Edge) across the
primary dataset splits:
- data/splits_primary/train.csv (N=486)
- data/splits_primary/val.csv   (N=108)
- data/splits_primary/test.csv  (N=109)

CRITICAL RIGOR REQUIREMENTS:
1. All feature scaling (StandardScaler) is strictly fitted on TRAIN data only.
2. Validation and Test splits are transformed using the fitted training scaler parameters.
3. Perfect alignment verified between image IDs / filenames and existing ViT feature caches.
4. Output saved to:
   - data/features_handcrafted/train_handcrafted_features.pt
   - data/features_handcrafted/val_handcrafted_features.pt
   - data/features_handcrafted/test_handcrafted_features.pt
   - models/handcrafted_scaler.pt
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, os.getcwd())
from src.features.handcrafted_extractor import (
    HandcraftedFeatureExtractor,
    FEATURE_DIMS,
    TOTAL_HANDCRAFTED_DIM
)
from src.features.feature_scaler import TrainingFeatureScaler


def process_split(
    manifest_csv: str,
    extractor: HandcraftedFeatureExtractor,
    split_name: str
) -> Dict[str, Any]:
    """
    Loads images in exact manifest order and extracts raw handcrafted features.
    """
    df = pd.read_csv(manifest_csv)
    print(f"\n[Split: {split_name}] Extracting features for {len(df)} images from '{manifest_csv}'...")

    raw_features_list = []
    lbp_list = []
    glcm_list = []
    color_list = []
    edge_list = []
    labels_list = []
    filenames_list = []
    paths_list = []

    start_time = time.time()
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        label = int(row["label"])
        filename = row["filename"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Load RGB image deterministically
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            # Resize deterministically to 224x224 matching ViT input size
            rgb_img = rgb_img.resize((224, 224), Image.Resampling.BILINEAR)
            img_np = np.array(rgb_img)

        f_dict = extractor.extract_dict(img_np)
        f_vec = np.concatenate([f_dict["lbp"], f_dict["glcm"], f_dict["color"], f_dict["edge"]]).astype(np.float32)

        raw_features_list.append(f_vec)
        lbp_list.append(f_dict["lbp"])
        glcm_list.append(f_dict["glcm"])
        color_list.append(f_dict["color"])
        edge_list.append(f_dict["edge"])
        labels_list.append(label)
        filenames_list.append(filename)
        paths_list.append(img_path)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} images ({time.time() - start_time:.1f}s)")

    raw_features = np.stack(raw_features_list, axis=0)  # [N, 143]
    assert raw_features.shape == (len(df), TOTAL_HANDCRAFTED_DIM)
    assert not np.isnan(raw_features).any(), "NaN values found in extracted raw features!"
    assert not np.isinf(raw_features).any(), "Inf values found in extracted raw features!"

    return {
        "split": split_name,
        "manifest_path": manifest_csv,
        "sample_count": len(df),
        "raw_features": torch.from_numpy(raw_features),
        "lbp_features": torch.from_numpy(np.stack(lbp_list, axis=0)),
        "glcm_features": torch.from_numpy(np.stack(glcm_list, axis=0)),
        "color_features": torch.from_numpy(np.stack(color_list, axis=0)),
        "edge_features": torch.from_numpy(np.stack(edge_list, axis=0)),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "filenames": filenames_list,
        "image_paths": paths_list,
        "feature_dims": FEATURE_DIMS,
        "total_dim": TOTAL_HANDCRAFTED_DIM,
    }


def main():
    print("=" * 70)
    print("Extracting and Scaling Handcrafted Image Features (Phases 5 & 6)")
    print("=" * 70)

    extractor = HandcraftedFeatureExtractor()
    out_dir = Path("data/features_handcrafted")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "data/splits_primary/train.csv",
        "val": "data/splits_primary/val.csv",
        "test": "data/splits_primary/test.csv"
    }

    results = {}
    for split_name, manifest in splits.items():
        results[split_name] = process_split(manifest, extractor, split_name)

    # 1. Fit scaler strictly on training data
    print("\n[Leakage Control] Fitting StandardScaler strictly on 'train' split (N=486)...")
    scaler = TrainingFeatureScaler()
    train_raw = results["train"]["raw_features"].numpy()
    scaler.fit(train_raw, split_name="train")

    # Save fitted scaler
    scaler_path = "models/handcrafted_scaler.pt"
    scaler.save(scaler_path, metadata={
        "feature_dims": FEATURE_DIMS,
        "total_dim": TOTAL_HANDCRAFTED_DIM,
        "train_samples": len(train_raw),
        "fitted_on": "data/splits_primary/train.csv"
    })
    print(f"Saved fitted scaler to '{scaler_path}'.")

    # 2. Transform all splits using training-fitted scaler
    for split_name in ["train", "val", "test"]:
        raw_feat = results[split_name]["raw_features"].numpy()
        scaled_feat = scaler.transform(raw_feat)
        results[split_name]["scaled_features"] = torch.from_numpy(scaled_feat)

        # Verify against ViT cache alignment
        vit_cache_path = f"data/features_vit_scratch/{split_name}_patch_features.pt"
        if os.path.exists(vit_cache_path):
            vit_data = torch.load(vit_cache_path, weights_only=False)
            assert results[split_name]["filenames"] == vit_data["filenames"], (
                f"Filename alignment failure in {split_name} split between handcrafted and ViT cache!"
            )
            vit_labels = torch.as_tensor(vit_data["labels"]).view(-1)
            our_labels = torch.as_tensor(results[split_name]["labels"]).view(-1)
            assert bool(torch.equal(our_labels, vit_labels)), (
                f"Label alignment failure in {split_name} split between handcrafted and ViT cache!"
            )
            print(f"Verified perfect alignment with ViT cache for split '{split_name}'.")

        # Save cache
        save_path = out_dir / f"{split_name}_handcrafted_features.pt"
        torch.save(results[split_name], str(save_path))
        print(f"Saved {split_name} features -> '{save_path}' (Shape: {scaled_feat.shape})")

    print("\nHandcrafted feature extraction and scaling completed successfully.")


if __name__ == "__main__":
    main()
