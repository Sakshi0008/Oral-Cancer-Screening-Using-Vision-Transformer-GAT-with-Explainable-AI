"""
Training Pipeline for Graph Classification Models (GAT & GATv2)
================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Executes the 6 required graph classification experiments:
1. Spatial + GAT
2. Similarity + GAT
3. Hybrid + GAT
4. Spatial + GATv2
5. Similarity + GATv2
6. Hybrid + GATv2

STRICT REQUIREMENTS:
- Trained on cached ViT patch representations (train & val splits).
- Class-weighted CrossEntropyLoss based on training labels.
- Early stopping based on validation loss.
- Save best checkpoints to models/graph_{name}_best.pth.
- Test set is NOT evaluated during training.
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.getcwd())
from src.models.graph_classifier import build_graph_classifier, GraphClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cached_datasets(
    features_dir: str = "data/features_vit_scratch",
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Load cached patch feature datasets and calculate training class weights.
    """
    train_file = os.path.join(features_dir, "train_patch_features.pt")
    val_file = os.path.join(features_dir, "val_patch_features.pt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Feature caches not found in {features_dir}. Run extract_patch_features.py first.")

    train_data = torch.load(train_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)

    train_x = train_data["patch_features"]  # (486, 196, 192)
    train_y = train_data["labels"]          # (486,)
    val_x = val_data["patch_features"]      # (108, 196, 192)
    val_y = val_data["labels"]              # (108,)

    # Compute balanced class weights strictly on training labels
    n_total = len(train_y)
    n_nc = (train_y == 0).sum().item()
    n_c = (train_y == 1).sum().item()
    w0 = n_total / (2.0 * n_nc)
    w1 = n_total / (2.0 * n_c)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, class_weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().numpy())

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

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        running_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    return val_loss, val_acc, val_prec, val_rec, val_f1


def train_single_experiment(
    exp_name: str,
    graph_type: str,
    gnn_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    pooling: str = "mean",
    k: int = 8,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    print("-" * 60)
    print(f"RUNNING EXPERIMENT: {exp_name.upper()} (Graph: {graph_type}, GNN: {gnn_type}, Pooling: {pooling})")
    print("-" * 60)

    set_seed(seed)
    model = build_graph_classifier(
        graph_type=graph_type,
        gnn_type=gnn_type,
        pooling=pooling,
        in_features=192,
        hidden_per_head=48,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        k=k,
        dropout=0.1,
        device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics = {}
    epochs_no_improve = 0
    history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_validation(model, val_loader, criterion, device)
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
            "lr": current_lr
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1
            }
            epochs_no_improve = 0

            # Save model checkpoint
            ckpt_path = f"models/graph_{exp_name}_best.pth"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_metrics": best_metrics,
                "config": {
                    "experiment_name": exp_name,
                    "graph_type": graph_type,
                    "gnn_type": gnn_type,
                    "pooling": pooling,
                    "k": k,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "is_pretrained": False
                }
            }
            torch.save(checkpoint, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] Terminated at epoch {epoch} (best epoch: {best_epoch})")
                break

    dur = time.time() - start_time
    print(f"[{exp_name.upper()} Finished in {dur:.1f}s] Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Val Acc: {best_metrics['val_acc']*100:.1f}% | Val F1: {best_metrics['val_f1']:.4f}")

    # Save history
    hist_path = f"reports/phase4_training_{exp_name}.csv"
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)

    return {
        "experiment_name": exp_name,
        "graph_type": graph_type,
        "gnn_type": gnn_type,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_metrics,
        "duration_sec": dur,
        "checkpoint_path": f"models/graph_{exp_name}_best.pth"
    }


def run_all_graph_experiments(
    features_dir: str = "data/features_vit_scratch",
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    print("=" * 65)
    print("PHASE 4: TRAINING ALL 6 GRAPH CLASSIFICATION EXPERIMENTS")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_weights = load_cached_datasets(features_dir, batch_size=batch_size)
    print(f"Loaded Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")
    print(f"Class Weights: {class_weights.tolist()}")

    experiments = [
        {"exp_name": "spatial_gat",     "graph_type": "spatial",    "gnn_type": "gat"},
        {"exp_name": "similarity_gat",  "graph_type": "similarity", "gnn_type": "gat"},
        {"exp_name": "hybrid_gat",      "graph_type": "hybrid",     "gnn_type": "gat"},
        {"exp_name": "spatial_gatv2",   "graph_type": "spatial",    "gnn_type": "gatv2"},
        {"exp_name": "similarity_gatv2","graph_type": "similarity", "gnn_type": "gatv2"},
        {"exp_name": "hybrid_gatv2",    "graph_type": "hybrid",     "gnn_type": "gatv2"}
    ]

    results = []
    for exp in experiments:
        res = train_single_experiment(
            exp_name=exp["exp_name"],
            graph_type=exp["graph_type"],
            gnn_type=exp["gnn_type"],
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device
        )
        results.append(res)

    print("\n" * 1 + "=" * 65)
    print("ALL 6 GRAPH EXPERIMENTS TRAINED SUCCESSFULLY!")
    print("=" * 65)
    return results


if __name__ == "__main__":
    run_all_graph_experiments()
