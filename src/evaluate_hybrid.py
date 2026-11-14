"""
Evaluation and Metrics Engine for Hybrid Models and Ablations
============================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Evaluates models on the frozen primary test split (N=109) and computes:
- Accuracy
- Cancer Sensitivity (Recall)
- Specificity
- Cancer Precision
- Cancer F1-score
- Macro F1-score
- Weighted F1-score
- ROC-AUC (continuous probabilities)
- PR-AUC (Average Precision score)
- Confusion Matrix: TN, FP, FN, TP
- 95% Confidence Intervals (Wilson score for proportions, bootstrap for AUC)
- Trainable Parameter Counts
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score interval for a binomial proportion k/n.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.959964  # for 95% CI
    p = k / n
    denominator = 1.0 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std_error = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - adjusted_std_error) / denominator)
    upper = min(1.0, (centre_adjusted_probability + adjusted_std_error) / denominator)
    return lower, upper


def bootstrap_auc_ci(
    targets: np.ndarray,
    probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes 95% bootstrap confidence interval for ROC-AUC.
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    n_samples = len(targets)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(targets[indices])) < 2:
            continue
        score = roc_auc_score(targets[indices], probs[indices])
        bootstrapped_scores.append(score)

    if len(bootstrapped_scores) == 0:
        base_auc = roc_auc_score(targets, probs)
        return base_auc, base_auc

    lower = float(np.percentile(bootstrapped_scores, 2.5))
    upper = float(np.percentile(bootstrapped_scores, 97.5))
    return lower, upper


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes all standard research evaluation metrics from true binary labels and continuous probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cancer_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.5

    sens_ci = wilson_score_interval(tp, tp + fn)
    spec_ci = wilson_score_interval(tn, tn + fp)
    auc_ci = bootstrap_auc_ci(y_true, y_probs)

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "sensitivity_ci": [float(sens_ci[0]), float(sens_ci[1])],
        "specificity": float(spec),
        "specificity_ci": [float(spec_ci[0]), float(spec_ci[1])],
        "precision": float(prec),
        "cancer_f1": float(cancer_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(roc_auc),
        "roc_auc_ci": [float(auc_ci[0]), float(auc_ci[1])],
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold)
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
