"""
Evaluation Pipeline for Custom Vision Transformer (From-Scratch Baseline)
==========================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Evaluates the best ViT checkpoint exclusively on data/splits_primary/test.csv.
Computes:
- Accuracy, Precision, Recall/Sensitivity (Cancer), Specificity (Non-Cancer)
- F1-score (Binary, Macro, Weighted)
- ROC-AUC and PR-AUC using continuous Softmax probabilities
- Confusion Matrix (TP, FP, TN, FN)
- Saves confusion matrix plot to reports/phase3_confusion_matrix.png
- Saves full metrics dictionary to reports/phase3_vit_metrics.json
"""

import os
import sys

# Configure writable matplotlib cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "scratch", "matplotlib")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, os.getcwd())
from src.models.vit_scratch import build_vit_scratch
from src.dataset import OralCancerDataset, get_transforms


def evaluate_test_set(
    checkpoint_path: str = "models/vit_scratch_best.pth",
    test_csv: str = "data/splits_primary/test.csv",
    output_metrics_json: str = "reports/phase3_vit_metrics.json",
    output_cm_plot: str = "reports/phase3_confusion_matrix.png"
):
    print("=" * 65)
    print("PHASE 3: ViT PRIMARY TEST SET EVALUATION")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load checkpoint (weights_only=False required in PyTorch 2.6 for custom metadata dicts)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded ViT checkpoint from: {checkpoint_path}")
    print(f"Best Epoch: {checkpoint['epoch']} (Val Loss: {checkpoint['best_val_loss']:.4f})")
    assert checkpoint['config'].get('is_pretrained', False) is False, "Pretrained weights violation in checkpoint!"

    # 2. Build model and load weights
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

    # 3. Load primary test dataset (strictly deterministic preprocessing)
    test_transform = get_transforms("test", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")
    test_dataset = OralCancerDataset(test_csv, transform=test_transform, return_metadata=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")

    # 4. Run inference
    all_targets = []
    all_preds = []
    all_probs = []  # continuous probability of class 1 (Cancer)
    all_meta = []

    with torch.no_grad():
        for images, targets, meta in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            for i in range(len(targets)):
                all_meta.append({
                    "path": meta["image_path"][i],
                    "filename": meta["filename"][i],
                    "true_label": targets[i].item(),
                    "pred_label": logits.argmax(dim=1)[i].item(),
                    "cancer_prob": float(probs[i, 1].item())
                })

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5. Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_cancer = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    rec_cancer = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)  # Sensitivity
    f1_cancer = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

    prec_non_cancer = precision_score(all_targets, all_preds, pos_label=0, zero_division=0)
    rec_non_cancer = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)  # Specificity
    f1_non_cancer = f1_score(all_targets, all_preds, pos_label=0, zero_division=0)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = rec_cancer

    roc_auc = roc_auc_score(all_targets, all_probs)
    pr_auc = average_precision_score(all_targets, all_probs)

    print("\n--- ViT TEST METRICS REPORT ---")
    print(f"Total Test Samples:    {len(all_targets)} (Cancer: {(all_targets==1).sum()}, Non-Cancer: {(all_targets==0).sum()})")
    print(f"Accuracy:              {acc*100:.2f}%")
    print(f"Sensitivity (Recall):  {sensitivity*100:.2f}% (TP={tp}, FN={fn})")
    print(f"Specificity:           {specificity*100:.2f}% (TN={tn}, FP={fp})")
    print(f"Precision (Cancer):    {prec_cancer*100:.2f}%")
    print(f"F1-Score (Cancer):     {f1_cancer:.4f}")
    print(f"F1-Score (Macro):      {macro_f1:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"PR-AUC (Avg Prec):     {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"                Predicted Non-Cancer (0)    Predicted Cancer (1)")
    print(f"Actual Non-Cancer (0):      TN = {tn:2d}                    FP = {fp:2d}")
    print(f"Actual Cancer (1):          FN = {fn:2d}                    TP = {tp:2d}")

    # 6. Save Confusion Matrix Plot
    os.makedirs(os.path.dirname(output_cm_plot), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Non-Cancer (0)", "Cancer (1)"],
        yticklabels=["Non-Cancer (0)", "Cancer (1)"],
        ylabel="Actual Label",
        xlabel="Predicted Label",
        title=f"ViT Scratch Baseline\nConfusion Matrix (Acc: {acc*100:.1f}%)"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(output_cm_plot, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix Plot: {output_cm_plot}")

    # 7. Save Metrics JSON
    metrics_summary = {
        "model": "VisionTransformer (Scratch)",
        "weights": "None (Random Initialization)",
        "test_manifest": test_csv,
        "sample_counts": {
            "total": int(len(all_targets)),
            "cancer": int((all_targets == 1).sum()),
            "non_cancer": int((all_targets == 0).sum())
        },
        "overall_metrics": {
            "accuracy": float(round(acc, 5)),
            "sensitivity_recall": float(round(sensitivity, 5)),
            "specificity": float(round(specificity, 5)),
            "precision_cancer": float(round(prec_cancer, 5)),
            "f1_cancer": float(round(f1_cancer, 5)),
            "macro_f1": float(round(macro_f1, 5)),
            "weighted_f1": float(round(weighted_f1, 5)),
            "roc_auc": float(round(roc_auc, 5)),
            "pr_auc": float(round(pr_auc, 5))
        },
        "class_wise_metrics": {
            "cancer": {
                "precision": float(round(prec_cancer, 5)),
                "recall": float(round(rec_cancer, 5)),
                "f1_score": float(round(f1_cancer, 5)),
                "support": int((all_targets == 1).sum())
            },
            "non_cancer": {
                "precision": float(round(prec_non_cancer, 5)),
                "recall": float(round(rec_non_cancer, 5)),
                "f1_score": float(round(f1_non_cancer, 5)),
                "support": int((all_targets == 0).sum())
            }
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(round(checkpoint["best_val_loss"], 5))
    }

    os.makedirs(os.path.dirname(output_metrics_json), exist_ok=True)
    with open(output_metrics_json, "w") as fp_json:
        json.dump(metrics_summary, fp_json, indent=2)
    print(f"Saved Metrics JSON:          {output_metrics_json}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_test_set()
