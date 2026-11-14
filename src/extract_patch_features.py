"""
Feature Extraction Module: Vision Transformer Patch Tokens
===========================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts 196 patch-token representations (dim=192) per image from the frozen ViT scratch checkpoint.
Stores extracted features separately for train, val, and test splits in:
    data/features_vit_scratch/train_patch_features.pt
    data/features_vit_scratch/val_patch_features.pt
    data/features_vit_scratch/test_patch_features.pt

STRICT REQUIREMENTS:
- ViT weights remain completely frozen.
- Fully deterministic transforms (Resize 224x224, ToTensor, Primary Normalization).
- Complete isolation: no cross-split leakage.
- Rich metadata stored alongside tensors (checkpoint path, seed, epoch, labels, paths).
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from src.models.vit_scratch import build_vit_scratch
from src.dataset import OralCancerDataset, get_transforms


def extract_split_features(
    model: torch.nn.Module,
    split_csv: str,
    split_name: str,
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Extracts patch tokens [N, 196, 192] and CLS tokens [N, 192] for one dataset split.
    """
    # Deterministic transforms for ALL splits during feature extraction
    transform = get_transforms("test", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")
    dataset = OralCancerDataset(split_csv, transform=transform, return_metadata=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_patches = []
    all_cls = []
    all_labels = []
    all_paths = []
    all_filenames = []

    print(f"Extracting features for '{split_name}' ({len(dataset)} samples)...")
    start = time.time()

    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device)
            feats = model.forward_features(images)

            all_patches.append(feats["patches"].cpu())  # (B, 196, 192)
            all_cls.append(feats["cls"].cpu())          # (B, 192)
            all_labels.extend(labels.numpy().tolist())

            for i in range(len(labels)):
                all_paths.append(meta["image_path"][i])
                all_filenames.append(meta["filename"][i])

    patches_tensor = torch.cat(all_patches, dim=0)  # (N, 196, 192)
    cls_tensor = torch.cat(all_cls, dim=0)          # (N, 192)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    elapsed = time.time() - start
    print(f"[{split_name.upper()}] Done in {elapsed:.1f}s. Patches shape: {patches_tensor.shape}, Labels shape: {labels_tensor.shape}")

    return {
        "split": split_name,
        "manifest_path": split_csv,
        "sample_count": len(dataset),
        "patch_features": patches_tensor,      # (N, 196, 192)
        "cls_features": cls_tensor,            # (N, 192)
        "labels": labels_tensor,                # (N,)
        "image_paths": all_paths,
        "filenames": all_filenames
    }


def extract_all_patch_features(
    checkpoint_path: str = "models/vit_scratch_best.pth",
    output_dir: str = "data/features_vit_scratch",
    splits_dir: str = "data/splits_primary",
    device: Optional[torch.device] = None
):
    print("=" * 65)
    print("FEATURE EXTRACTION: ViT PATCH TOKENS FOR GRAPH EXPERIMENTS")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load ViT Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded ViT checkpoint: Epoch {checkpoint['epoch']} (Val Loss: {checkpoint['best_val_loss']:.4f})")

    model = build_vit_scratch(
        img_size=224,
        patch_size=16,
        num_classes=2,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_dim=768,
        dropout=0.1,
        device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    extracted_data = {}
    for split in ["train", "val", "test"]:
        split_csv = os.path.join(splits_dir, f"{split}.csv")
        split_data = extract_split_features(
            model=model,
            split_csv=split_csv,
            split_name=split,
            device=device,
            batch_size=16
        )

        # Store metadata
        split_data["checkpoint_path"] = checkpoint_path
        split_data["best_epoch"] = checkpoint["epoch"]
        split_data["best_val_loss"] = checkpoint["best_val_loss"]
        split_data["patch_count"] = 196
        split_data["feature_dim"] = 192
        split_data["seed"] = checkpoint["config"]["seed"]

        out_file = os.path.join(output_dir, f"{split}_patch_features.pt")
        torch.save(split_data, out_file)
        print(f"Saved: {out_file} ({os.path.getsize(out_file) / (1024**2):.2f} MB)")
        extracted_data[split] = split_data

    print("=" * 65)
    print("ALL PATCH FEATURES SUCCESSFULLY EXTRACTED & CACHED!")
    print("=" * 65)
    return extracted_data


if __name__ == "__main__":
    extract_all_patch_features()
