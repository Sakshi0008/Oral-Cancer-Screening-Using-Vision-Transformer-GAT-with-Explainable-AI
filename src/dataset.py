"""
Oral Cancer Dataset Pipeline & Preprocessing Module
===================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

This module implements a leak-free, scientifically validated PyTorch Dataset
and DataLoader pipeline adhering to Phase 1 data preparation requirements.

Dataset Configuration:
---------------------
- Primary Dataset (Duplicate-Collapsed, 703 Images):
  Located in 'data/splits_primary/'
  - Train: 486 images (322 Cancer, 164 Non-Cancer)
  - Val:   108 images (73 Cancer, 35 Non-Cancer)
  - Test:  109 images (75 Cancer, 34 Non-Cancer)
  - Every clinical scene appears exactly ONCE. Zero duplicate copies.

- Secondary Dataset (Duplicate-Aware, 748 Images):
  Located in 'data/splits/'
  - Retained for sensitivity analysis with multi-copy duplicate scenes kept grouped.

- Normalization (Computed on Primary Training Set, N=486):
  DATASET_PRIMARY_MEAN = [0.6432, 0.4537, 0.4275]
  DATASET_PRIMARY_STD  = [0.2189, 0.2118, 0.2090]
"""

import os
from typing import Callable, Optional, Tuple, Dict, Any, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

# Primary dataset normalization (computed strictly on 486 primary training images, zero leakage)
DATASET_PRIMARY_MEAN = [0.6432, 0.4537, 0.4275]
DATASET_PRIMARY_STD  = [0.2189, 0.2118, 0.2090]

# Secondary 748-image dataset normalization (computed on 523 training images)
DATASET_TRAIN_MEAN = [0.6453, 0.4540, 0.4285]
DATASET_TRAIN_STD  = [0.2189, 0.2124, 0.2096]

# Standard ImageNet normalization (external baseline)
IMAGENET_MEAN = [0.4850, 0.4560, 0.4060]
IMAGENET_STD  = [0.2290, 0.2240, 0.2250]


class LetterboxPad:
    """
    Resize image preserving aspect ratio and pad symmetrically to target_size.
    Optionally generates a binary mask indicating valid anatomical regions vs padding.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        fill: int = 0
    ):
        self.target_size = target_size  # (height, width)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        target_h, target_w = self.target_size
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top

        return F.pad(resized_img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

    def get_mask(self, img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate a binary mask (1 for valid tissue, 0 for padding) of shape (target_h, target_w).
        """
        target_h, target_w = self.target_size
        w, h = img_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        mask = torch.zeros((target_h, target_w), dtype=torch.float32)
        mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1.0
        return mask


def get_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (224, 224),
    resize_mode: str = "direct",
    norm_mode: str = "dataset"
) -> transforms.Compose:
    """
    Construct torchvision transform pipeline for train, val, or test.

    Args:
        split (str): 'train', 'val', or 'test'.
        image_size (tuple): Target (height, width), default (224, 224).
        resize_mode (str): 'direct' (bilinear resize) or 'letterbox' (aspect-ratio padding).
        norm_mode (str): 'dataset' (computed on primary training set, recommended for training from scratch)
                         or 'imagenet' (standard ImageNet baseline).

    Returns:
        transforms.Compose: Composed transform pipeline.
    """
    split = split.lower().strip()

    # 1. Select normalization parameters
    if norm_mode == "dataset":
        mean, std = DATASET_PRIMARY_MEAN, DATASET_PRIMARY_STD
    elif norm_mode == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(f"Unknown norm_mode '{norm_mode}'. Choose 'dataset' or 'imagenet'.")

    # 2. Select resize method
    if resize_mode == "letterbox":
        resizer = LetterboxPad(target_size=image_size, fill=0)
    elif resize_mode == "direct":
        resizer = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unknown resize_mode '{resize_mode}'. Choose 'direct' or 'letterbox'.")

    # 3. Build split-specific pipeline
    if split == "train":
        # Conservative clinical augmentations
        return transforms.Compose([
            resizer,
            transforms.RandomHorizontalFlip(p=0.5), # Preserves oral bilateral symmetry
            transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR), # Natural head/camera tilt
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0), # Conservative lighting variance; NO color distortion
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif split in ("val", "validation", "test"):
        # Strictly deterministic preprocessing for validation and test (NO augmentation)
        return transforms.Compose([
            resizer,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")


class OralCancerDataset(Dataset):
    """
    PyTorch Dataset for Oral Cancer screening manifests.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Split CSV manifest not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir or ""
        self.transform = transform
        self.return_metadata = return_metadata

        required_cols = {"image_path", "label"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV manifest is missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["image_path"]

        if self.root_dir and not os.path.isabs(rel_path):
            full_path = os.path.join(self.root_dir, rel_path)
        else:
            full_path = rel_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        # Open and strictly convert to 3-channel RGB
        with Image.open(full_path) as raw_img:
            img = raw_img.convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_PRIMARY_MEAN, std=DATASET_PRIMARY_STD)
            ])(img)

        label = torch.tensor(row["label"], dtype=torch.long)

        if self.return_metadata:
            meta = {
                "image_path": rel_path,
                "filename": row.get("filename", os.path.basename(rel_path)),
                "label_name": row.get("label_name", ""),
                "group_id": row.get("group_id", -1),
                "sha256": row.get("sha256", ""),
                "phash": row.get("phash", "")
            }
            return image_tensor, label, meta

        return image_tensor, label


def get_dataloader(
    csv_file: str,
    split: str = "train",
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    resize_mode: str = "direct",
    norm_mode: str = "dataset",
    return_metadata: bool = False
) -> DataLoader:
    """
    Construct a PyTorch DataLoader for a specific split.
    """
    split = split.lower().strip()
    if shuffle is None:
        shuffle = (split == "train")

    transform = get_transforms(
        split=split,
        image_size=image_size,
        resize_mode=resize_mode,
        norm_mode=norm_mode
    )
    dataset = OralCancerDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
        return_metadata=return_metadata
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )


def get_all_dataloaders(
    splits_dir: str = "data/splits_primary",
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (224, 224),
    resize_mode: str = "direct",
    norm_mode: str = "dataset"
) -> Dict[str, DataLoader]:
    """
    Convenience function to load train, val, and test DataLoaders simultaneously.
    Defaults to primary duplicate-collapsed splits ('data/splits_primary').
    """
    loaders = {}
    for s in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{s}.csv")
        loaders[s] = get_dataloader(
            csv_file=csv_path,
            split=s,
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            resize_mode=resize_mode,
            norm_mode=norm_mode
        )
    return loaders


if __name__ == "__main__":
    print("=" * 65)
    print("Testing Primary Duplicate-Collapsed Dataset Pipeline...")
    print("=" * 65)

    # 1. Test DataLoaders instantiation on primary splits (703 images)
    loaders_p = get_all_dataloaders(splits_dir="data/splits_primary", batch_size=16)
    train_n = len(loaders_p["train"].dataset)
    val_n = len(loaders_p["val"].dataset)
    test_n = len(loaders_p["test"].dataset)
    print(f"[1] Primary DataLoaders: Train={train_n}, Val={val_n}, Test={test_n}, Total={train_n+val_n+test_n}")
    assert train_n + val_n + test_n == 703, f"Expected 703 primary images, got {train_n+val_n+test_n}"

    # 2. Test batch extraction & tensor shape verification
    for split_name, loader in loaders_p.items():
        images, labels = next(iter(loader))
        print(f"[2] {split_name.capitalize()} Batch: images shape={images.shape}, dtype={images.dtype}")
        assert images.shape[1:] == (3, 224, 224), f"Shape mismatch: {images.shape}"
        assert images.dtype == torch.float32, f"Dtype mismatch: {images.dtype}"

    # 3. Test secondary 748-image DataLoaders
    loaders_s = get_all_dataloaders(splits_dir="data/splits", batch_size=16)
    s_total = len(loaders_s["train"].dataset) + len(loaders_s["val"].dataset) + len(loaders_s["test"].dataset)
    print(f"[3] Secondary DataLoaders: Total={s_total} (Expected: 748)")
    assert s_total == 748

    print("=" * 65)
    print("ALL PIPELINE VERIFICATIONS PASSED!")
    print("=" * 65)
