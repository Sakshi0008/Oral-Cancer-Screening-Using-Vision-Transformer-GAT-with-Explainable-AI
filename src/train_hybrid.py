"""
Training Module for Hybrid Fusion Models and Handcrafted Ablations
=================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Provides a unified, reproducible training framework for:
- ViT + GAT Concatenation Fusion
- ViT + GAT Gated Fusion
- ViT + GAT Residual Fusion
- Handcrafted Feature Only Classifiers (Full, LBP, GLCM, Color, Edge)
- Bi-Modal Handcrafted Fusion (ViT + Handcrafted, GAT + Handcrafted)
- Tri-Modal Fusion (ViT + GAT + Handcrafted Concat & Gated)
- Multi-Scale ViT and Multi-Scale Hybrid Models

Experimental Controls:
- Seed: 42 (fixed across all experiments)
- Training-only class weights: w0=1.51875, w1=0.7454
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
- Scheduler: CosineAnnealingLR
- Checkpoint selection: Strictly minimum validation loss
- Early stopping: patience = 15 epochs, max epochs = 60
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.getcwd())


class FusionDataset(Dataset):
    """
    Multimodal Dataset loading aligned feature tensors:
    - vit_cls: [192]
    - patch_tokens: [196, 192]
    - graph_rep: [192]
    - hc_scaled: [143] (or sub-slice for sub-family ablations)
    - label: int (0 or 1)
    """
    def __init__(
        self,
        vit_cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        graph_rep: torch.Tensor,
        hc_scaled: torch.Tensor,
        labels: torch.Tensor,
        hc_slice: Optional[slice] = None
    ):
        self.vit_cls = vit_cls.float()
        self.patch_tokens = patch_tokens.float()
        self.graph_rep = graph_rep.float()
        if hc_slice is not None:
            self.hc_scaled = hc_scaled[:, hc_slice].float()
        else:
            self.hc_scaled = hc_scaled.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.vit_cls[idx],
            self.patch_tokens[idx],
            self.graph_rep[idx],
            self.hc_scaled[idx],
            self.labels[idx]
        )


def load_dataset_splits(
    gnn_name: str = "hybrid_gatv2",
    hc_slice: Optional[slice] = None
) -> Tuple[FusionDataset, FusionDataset, FusionDataset]:
    """
    Loads all pre-extracted and verified feature caches for train, val, test.
    """
    splits_data = {}
    for split in ["train", "val", "test"]:
        vit_path = f"data/features_vit_scratch/{split}_patch_features.pt"
        graph_path = f"data/features_graph/{split}_graph_{gnn_name}.pt"
        hc_path = f"data/features_handcrafted/{split}_handcrafted_features.pt"

        vit_d = torch.load(vit_path, map_location="cpu", weights_only=False)
        graph_d = torch.load(graph_path, map_location="cpu", weights_only=False)
        hc_d = torch.load(hc_path, map_location="cpu", weights_only=False)

        splits_data[split] = FusionDataset(
            vit_cls=vit_d["cls_features"],
            patch_tokens=vit_d["patch_features"],
            graph_rep=graph_d["graph_representations"],
            hc_scaled=hc_d["scaled_features"],
            labels=torch.as_tensor(vit_d["labels"]).long(),
            hc_slice=hc_slice
        )

    return splits_data["train"], splits_data["val"], splits_data["test"]


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    forward_fn: Any,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 15,
    save_path: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Trains a fusion or ablation model with early stopping on validation loss.
    """
    # Class weights from primary train split (326 cancer, 160 non-cancer)
    w0 = 486.0 / (2.0 * 160.0)
    w1 = 486.0 / (2.0 * 326.0)
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_val_metrics = {}
    best_epoch = 0
    patience_counter = 0
    history = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        # 1. Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch in train_loader:
            labels = batch[4].to(device)
            optimizer.zero_grad()
            logits = forward_fn(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            total_train += len(labels)

        scheduler.step()
        train_loss /= total_train
        train_acc = train_correct / total_train

        # 2. Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch[4].to(device)
                logits = forward_fn(model, batch, device)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)

                val_correct += (preds == labels).sum().item()
                total_val += len(labels)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_loss /= total_val
        val_acc = val_correct / total_val

        # Compute validation sensitivity, specificity, F1
        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds)
        tp = int(np.sum((preds_np == 1) & (targets_np == 1)))
        tn = int(np.sum((preds_np == 0) & (targets_np == 0)))
        fp = int(np.sum((preds_np == 1) & (targets_np == 0)))
        fn = int(np.sum((preds_np == 0) & (targets_np == 1)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        epoch_record = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_sens": sens,
            "val_spec": spec,
            "val_prec": prec,
            "val_f1": f1
        }
        history.append(epoch_record)

        # Check early stopping / best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_val_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_sens": sens,
                "val_spec": spec,
                "val_prec": prec,
                "val_f1": f1,
                "best_epoch": epoch
            }
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "epoch": epoch,
                    "best_val_metrics": best_val_metrics,
                    "is_pretrained": False,
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_metrics, history
