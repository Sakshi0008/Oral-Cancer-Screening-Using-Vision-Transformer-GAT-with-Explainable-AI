"""
Evaluation Pipeline for Graph Classification Models & Comparative Reporting
=============================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Evaluates the 6 trained graph classification checkpoints exclusively on the primary test set:
    data/features_vit_scratch/test_patch_features.pt (109 samples)

Computes for each model:
- Accuracy, Cancer Sensitivity, Specificity, Cancer Precision, Cancer F1, Macro F1, Weighted F1
- ROC-AUC and PR-AUC using continuous Softmax probabilities
- Confusion Matrix (TN, FP, FN, TP)

Integrates ViT-only baseline results and compiles the comprehensive comparison table into:
    reports/phase4_experiment_comparison.csv
"""

import os
import sys
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

sys.path.insert(0, os.getcwd())
from src.models.graph_classifier import build_graph_classifier


def evaluate_single_graph_model(
    checkpoint_path: str,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates one saved graph checkpoint on the test set.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = build_graph_classifier(
        graph_type=cfg["graph_type"],
        gnn_type=cfg["gnn_type"],
        pooling=cfg.get("pooling", "mean"),
        in_features=192,
        hidden_per_head=48,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        k=cfg.get("k", 8),
        device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(TensorDataset(test_x, test_y), batch_size=16, shuffle=False)

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

            all_targets.extend(batch_y.numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    prec_cancer = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    rec_cancer = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)  # Sensitivity
    f1_cancer = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

    prec_nc = precision_score(all_targets, all_preds, pos_label=0, zero_division=0)
    rec_nc = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)      # Specificity
    f1_nc = f1_score(all_targets, all_preds, pos_label=0, zero_division=0)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = roc_auc_score(all_targets, all_probs)
    pr_auc = average_precision_score(all_targets, all_probs)

    return {
        "experiment_name": cfg["experiment_name"],
        "graph_type": cfg["graph_type"],
        "gnn_type": cfg["gnn_type"],
        "pooling": cfg.get("pooling", "mean"),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(round(checkpoint["best_val_loss"], 5)),
        "accuracy": float(round(acc, 5)),
        "sensitivity": float(round(rec_cancer, 5)),
        "specificity": float(round(specificity, 5)),
        "precision_cancer": float(round(prec_cancer, 5)),
        "f1_cancer": float(round(f1_cancer, 5)),
        "macro_f1": float(round(macro_f1, 5)),
        "weighted_f1": float(round(weighted_f1, 5)),
        "roc_auc": float(round(roc_auc, 5)),
        "pr_auc": float(round(pr_auc, 5)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def evaluate_all_and_compare(
    features_dir: str = "data/features_vit_scratch",
    vit_metrics_json: str = "reports/phase3_vit_metrics.json",
    output_comparison_csv: str = "reports/phase4_experiment_comparison.csv"
) -> pd.DataFrame:
    print("=" * 65)
    print("PHASE 4: EVALUATING ALL GRAPH EXPERIMENTS & COMPILING COMPARISON")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load cached test features
    test_file = os.path.join(features_dir, "test_patch_features.pt")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test features not found: {test_file}")

    test_data = torch.load(test_file, weights_only=False)
    test_x = test_data["patch_features"]  # (109, 196, 192)
    test_y = test_data["labels"]          # (109,)
    print(f"Loaded Test Samples: {len(test_y)} (Cancer: {(test_y==1).sum().item()}, Non-Cancer: {(test_y==0).sum().item()})")

    # 2. Evaluate the 6 graph experiments
    exp_names = [
        "spatial_gat",
        "similarity_gat",
        "hybrid_gat",
        "spatial_gatv2",
        "similarity_gatv2",
        "hybrid_gatv2"
    ]

    all_rows = []

    # 3. Add ViT-only baseline row if available
    if os.path.exists(vit_metrics_json):
        with open(vit_metrics_json, "r") as f:
            vit_m = json.load(f)
        all_rows.append({
            "model_architecture": "ViT-Only (Scratch Baseline)",
            "graph_strategy": "N/A",
            "gnn_type": "N/A",
            "best_epoch": vit_m.get("best_epoch", "-"),
            "best_val_loss": vit_m.get("best_val_loss", "-"),
            "accuracy": vit_m["overall_metrics"]["accuracy"],
            "sensitivity": vit_m["overall_metrics"]["sensitivity_recall"],
            "specificity": vit_m["overall_metrics"]["specificity"],
            "precision_cancer": vit_m["overall_metrics"]["precision_cancer"],
            "f1_cancer": vit_m["overall_metrics"]["f1_cancer"],
            "macro_f1": vit_m["overall_metrics"]["macro_f1"],
            "weighted_f1": vit_m["overall_metrics"]["weighted_f1"],
            "roc_auc": vit_m["overall_metrics"]["roc_auc"],
            "pr_auc": vit_m["overall_metrics"]["pr_auc"],
            "confusion_matrix": f"TN={vit_m['confusion_matrix']['true_negative']}, FP={vit_m['confusion_matrix']['false_positive']}, FN={vit_m['confusion_matrix']['false_negative']}, TP={vit_m['confusion_matrix']['true_positive']}"
        })

    for exp in exp_names:
        ckpt_path = f"models/graph_{exp}_best.pth"
        print(f"Evaluating {exp}...")
        res = evaluate_single_graph_model(ckpt_path, test_x, test_y, device)
        all_rows.append({
            "model_architecture": f"{res['gnn_type'].upper()} ({res['graph_type'].capitalize()})",
            "graph_strategy": res["graph_type"],
            "gnn_type": res["gnn_type"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "accuracy": res["accuracy"],
            "sensitivity": res["sensitivity"],
            "specificity": res["specificity"],
            "precision_cancer": res["precision_cancer"],
            "f1_cancer": res["f1_cancer"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "confusion_matrix": f"TN={res['tn']}, FP={res['fp']}, FN={res['fn']}, TP={res['tp']}"
        })

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_comparison_csv), exist_ok=True)
    df.to_csv(output_comparison_csv, index=False)
    print(f"\nSaved Comparative Results: {output_comparison_csv}")
    print("\n" + df.to_string(index=False))

    return df


if __name__ == "__main__":
    evaluate_all_and_compare()
