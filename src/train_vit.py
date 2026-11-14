"""
Training Pipeline for Custom Vision Transformer (From-Scratch Baseline)
=======================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

STRICT CONSTRAINTS:
- No pretrained weights (is_pretrained == False).
- Training exclusively on primary dataset (data/splits_primary/train.csv & val.csv).
- Class-weighted CrossEntropyLoss based on training labels.
- Early stopping based on validation loss.
- Save best checkpoint to models/vit_scratch_best.pth.
- Test set is NOT evaluated during training.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "scratch", "matplotlib")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.getcwd())
from src.models.vit_scratch import build_vit_scratch
from src.dataset import (
    OralCancerDataset,
    get_transforms,
    DATASET_PRIMARY_MEAN,
    DATASET_PRIMARY_STD
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_class_weights(train_csv: str, device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights exclusively from training labels:
        w_c = N_total / (num_classes * N_c)
    """
    df = pd.read_csv(train_csv)
    total_samples = len(df)
    n_non_cancer = (df["label"] == 0).sum()
    n_cancer = (df["label"] == 1).sum()

    w0 = total_samples / (2.0 * n_non_cancer)
    w1 = total_samples / (2.0 * n_cancer)

    weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"Training Class Distribution: Non-Cancer (0)={n_non_cancer}, Cancer (1)={n_cancer}, Total={total_samples}")
    print(f"Computed Class Weights:      [Non-Cancer={w0:.4f}, Cancer={w1:.4f}]")
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to stabilize Transformer scratch training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    return val_loss, val_acc, val_prec, val_rec, val_f1


def plot_learning_curves(history: list, output_plot_path: str):
    """Generate and save loss and accuracy learning curves."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_accs = [h["train_acc"] * 100 for h in history]
    val_accs = [h["val_acc"] * 100 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, val_losses, "r-s", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss (Cross-Entropy)", fontsize=11)
    axes[0].set_title("ViT Scratch Baseline: Loss Curves", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy curve
    axes[1].plot(epochs, train_accs, "b-o", label="Train Accuracy", markersize=4)
    axes[1].plot(epochs, val_accs, "r-s", label="Val Accuracy", markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("ViT Scratch Baseline: Accuracy Curves", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Saved Learning Curves: {output_plot_path}")


def run_training(
    train_csv: str = "data/splits_primary/train.csv",
    val_csv: str = "data/splits_primary/val.csv",
    output_model_path: str = "models/vit_scratch_best.pth",
    history_csv_path: str = "reports/phase3_vit_training_history.csv",
    curves_plot_path: str = "reports/phase3_learning_curves.png",
    seed: int = 42,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 50,
    patience: int = 12
):
    print("=" * 65)
    print("STARTING CUSTOM VISION TRANSFORMER (SCRATCH) TRAINING")
    print("=" * 65)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Transforms and DataLoaders
    train_transform = get_transforms("train", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")
    val_transform = get_transforms("val", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")

    train_dataset = OralCancerDataset(train_csv, transform=train_transform)
    val_dataset = OralCancerDataset(val_csv, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    print(f"Train samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
    print(f"Val samples:   {len(val_dataset)}, Batches per epoch: {len(val_loader)}")

    # 2. Build ViT Strictly From Scratch
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
    assert not model.is_pretrained, "Pretrained weights violation!"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized ViT Model Parameters: {total_params:,}")

    # 3. Class-weighted Loss & Optimizer
    class_weights = calculate_class_weights(train_csv, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    # 4. Training loop
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics = {}
    epochs_no_improve = 0
    history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_validation(
            model, val_loader, criterion, device
        )
        epoch_dur = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "train_f1": round(train_f1, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_prec": round(val_prec, 5),
            "val_rec": round(val_rec, 5),
            "val_f1": round(val_f1, 5),
            "lr": current_lr,
            "duration_sec": round(epoch_dur, 2)
        })

        improved = val_loss < best_val_loss
        flag = "(* BEST *)" if improved else ""

        print(f"Epoch {epoch:2d}/{max_epochs:2d} [{epoch_dur:.1f}s] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}%, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, F1: {val_f1:.4f} | "
              f"LR: {current_lr:.1e} {flag}")

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1
            }
            epochs_no_improve = 0

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_metrics": best_val_metrics,
                "config": {
                    "model": "VisionTransformer-Scratch",
                    "img_size": 224,
                    "patch_size": 16,
                    "num_patches": 196,
                    "embed_dim": 192,
                    "depth": 4,
                    "num_heads": 3,
                    "mlp_dim": 768,
                    "dropout": 0.1,
                    "is_pretrained": False,
                    "weights": None,
                    "seed": seed,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "class_weights": class_weights.cpu().tolist(),
                    "norm_mean": DATASET_PRIMARY_MEAN,
                    "norm_std": DATASET_PRIMARY_STD,
                    "train_manifest": train_csv,
                    "val_manifest": val_csv,
                    "class_mapping": {0: "NON CANCER", 1: "CANCER"}
                },
                "env": {
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                }
            }
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(checkpoint, output_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"ViT Training Complete! Total Time: {total_training_time:.1f}s")
    print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_metrics['val_acc']*100:.1f}%, Val F1: {best_val_metrics['val_f1']:.4f}")
    print(f"Saved Checkpoint: {output_model_path}")
    print("=" * 65)

    # Save history to CSV and plot curves
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved Training History: {history_csv_path}")

    plot_learning_curves(history, curves_plot_path)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "total_training_time": total_training_time,
        "history": history
    }


if __name__ == "__main__":
    results = run_training()
