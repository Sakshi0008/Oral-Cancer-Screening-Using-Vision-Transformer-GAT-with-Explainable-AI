"""
Controlled Ablation Studies and Master Evaluation Runner (Phases 5 & 6)
======================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Executes all 10+ required ablation configurations under strict scientific control:
1. ViT-only baseline (Phase 3 Scratch ViT, Epoch 34)
2. Selected GATv2-only baseline (Phase 4 Hybrid GATv2, Epoch 21)
3. Comparative GAT-only baseline (Phase 4 Hybrid GAT, Epoch 26)
4. ViT + GAT Concatenation Fusion (Primary Hybrid Approach 1)
5. ViT + GAT Gated Fusion (Primary Hybrid Approach 2)
6. ViT + GAT Residual Fusion (Primary Hybrid Approach 3)
7. Handcrafted Features Only (143-dim Combined)
   - 7a. LBP Texture Only (59 dims)
   - 7b. GLCM Texture Only (24 dims)
   - 7c. Color Moments & Histograms Only (46 dims)
   - 7d. Edge & Gradient Statistics Only (14 dims)
8. ViT + Handcrafted Features (Bi-modal)
9. GAT + Handcrafted Features (Bi-modal)
10. ViT + GAT + Handcrafted Features: Concatenation Fusion (Full Tri-modal)
11. ViT + GAT + Handcrafted Features: Modality-Gated Fusion (Full Tri-modal)
12. Multi-Scale ViT Representation (Fine + Regional + Global CLS, 576 dims)
13. Multi-Scale Hybrid Representation (Multi-scale ViT + GAT Graph, 768 dims)
14. Strategy B: Controlled End-to-End Fine-Tuning of Primary Hybrid Model

Strict Experimental Protocol:
- Model selection strictly driven by minimum validation loss.
- Decision threshold (0.5) frozen prior to test evaluation.
- Test evaluation conducted once on frozen checkpoints.
- Saves reports, manifests, comparisons, and training histories.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from src.dataset import OralCancerDataset, get_transforms
from src.models.vit_scratch import build_vit_scratch
from src.models.graph_classifier import GraphClassifier
from src.models.hybrid_fusion import ConcatenationFusion, GatedFusion, ResidualFusion
from src.models.handcrafted_fusion import (
    HandcraftedOnlyClassifier,
    BiModalHandcraftedFusion,
    TriModalConcatFusion,
    TriModalGatedFusion
)
from src.models.multiscale_fusion import MultiScaleViTClassifier, MultiScaleHybridClassifier
from src.train_hybrid import FusionDataset, load_dataset_splits, train_single_model
from src.evaluate_hybrid import evaluate_predictions, count_trainable_parameters


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    forward_fn: Any,
    threshold: float = 0.5
) -> Dict[str, Any]:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch[4].numpy()
            logits = forward_fn(model, batch, device)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            all_targets.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    metrics = evaluate_predictions(np.array(all_targets), np.array(all_probs), threshold=threshold)
    return metrics


# Forward function adapters
def fwd_concat(m, b, dev):
    return m(b[0].to(dev), b[2].to(dev))[0]

def fwd_gated(m, b, dev):
    return m(b[0].to(dev), b[2].to(dev))[0]

def fwd_residual(m, b, dev):
    return m(b[0].to(dev), b[2].to(dev))[0]

def fwd_hc_only(m, b, dev):
    return m(b[3].to(dev))

def fwd_vit_hc(m, b, dev):
    return m(b[0].to(dev), b[3].to(dev))[0]

def fwd_gat_hc(m, b, dev):
    return m(b[2].to(dev), b[3].to(dev))[0]

def fwd_trimodal_concat(m, b, dev):
    return m(b[0].to(dev), b[2].to(dev), b[3].to(dev))[0]

def fwd_trimodal_gated(m, b, dev):
    return m(b[0].to(dev), b[2].to(dev), b[3].to(dev))[0]

def fwd_multiscale_vit(m, b, dev):
    return m(b[1].to(dev), b[0].to(dev))[0]

def fwd_multiscale_hybrid(m, b, dev):
    return m(b[1].to(dev), b[0].to(dev), b[2].to(dev))[0]


def run_all_ablations():
    print("=" * 80)
    print("PHASES 5 & 6: CONTROLLED ABLATION STUDIES AND HYBRID EVALUATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    set_seed(42)

    # 1. Load Primary Pre-extracted Splits with Hybrid GATv2
    train_ds, val_ds, test_ds = load_dataset_splits(gnn_name="hybrid_gatv2")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Also load Hybrid GAT splits for comparative graph ablation
    train_ds_gat, val_ds_gat, test_ds_gat = load_dataset_splits(gnn_name="hybrid_gat")
    train_loader_gat = DataLoader(train_ds_gat, batch_size=16, shuffle=True)
    val_loader_gat = DataLoader(val_ds_gat, batch_size=16, shuffle=False)
    test_loader_gat = DataLoader(test_ds_gat, batch_size=16, shuffle=False)

    all_results = []
    all_histories = []

    # -------------------------------------------------------------
    # Experiment 1: ViT-Only Baseline (Epoch 34 checkpoint)
    # -------------------------------------------------------------
    print("\n[Ablation 1/14] Evaluating ViT-Only Baseline (Phase 3 Epoch 34 Checkpoint)...")
    vit_ckpt = torch.load("models/vit_scratch_best.pth", map_location=device, weights_only=False)
    vit_model = build_vit_scratch(embed_dim=192, depth=4, num_heads=3, num_classes=2).to(device)
    vit_model.load_state_dict(vit_ckpt["model_state_dict"])
    vit_model.eval()

    # ViT classifier evaluation on cached CLS features
    val_vit_cls = val_ds.vit_cls.to(device)
    test_vit_cls = test_ds.vit_cls.to(device)
    with torch.no_grad():
        val_logits = vit_model.head(val_vit_cls)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        test_logits = vit_model.head(test_vit_cls)
        test_probs = torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()

    vit_val_metrics = evaluate_predictions(val_ds.labels.numpy(), val_probs)
    vit_test_metrics = evaluate_predictions(test_ds.labels.numpy(), test_probs)

    all_results.append({
        "experiment_id": "01_vit_scratch_only",
        "model_category": "Deep Learning Baseline",
        "description": "Scratch ViT (Epoch 34) CLS representation only",
        "parameters": count_trainable_parameters(vit_model),
        "best_epoch": 34,
        "val_loss": 0.4880,
        "val_acc": vit_val_metrics["accuracy"],
        "val_sens": vit_val_metrics["sensitivity"],
        "val_spec": vit_val_metrics["specificity"],
        "val_f1": vit_val_metrics["cancer_f1"],
        "test_acc": vit_test_metrics["accuracy"],
        "test_sens": vit_test_metrics["sensitivity"],
        "test_sens_ci": vit_test_metrics["sensitivity_ci"],
        "test_spec": vit_test_metrics["specificity"],
        "test_spec_ci": vit_test_metrics["specificity_ci"],
        "test_prec": vit_test_metrics["precision"],
        "test_f1": vit_test_metrics["cancer_f1"],
        "test_macro_f1": vit_test_metrics["macro_f1"],
        "test_weighted_f1": vit_test_metrics["weighted_f1"],
        "test_roc_auc": vit_test_metrics["roc_auc"],
        "test_roc_auc_ci": vit_test_metrics["roc_auc_ci"],
        "test_pr_auc": vit_test_metrics["pr_auc"],
        "confusion_matrix": vit_test_metrics["confusion_matrix"],
        "training_strategy": "Pre-trained scratch ViT (frozen)"
    })

    # -------------------------------------------------------------
    # Experiment 2: Selected GATv2-Only Baseline (Hybrid GATv2, Epoch 21)
    # -------------------------------------------------------------
    print("\n[Ablation 2/14] Evaluating Selected GATv2-Only Baseline (Hybrid GATv2, Epoch 21)...")
    gatv2_ckpt = torch.load("models/graph_hybrid_gatv2_best.pth", map_location=device, weights_only=False)
    gatv2_model = GraphClassifier(graph_type="hybrid", gnn_type="gatv2", pooling="mean", k=8).to(device)
    gatv2_model.load_state_dict(gatv2_ckpt["model_state_dict"])
    gatv2_model.eval()

    val_gat_rep = val_ds.graph_rep.to(device)
    test_gat_rep = test_ds.graph_rep.to(device)
    with torch.no_grad():
        val_logits = gatv2_model.head(val_gat_rep)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        test_logits = gatv2_model.head(test_gat_rep)
        test_probs = torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()

    gatv2_val_metrics = evaluate_predictions(val_ds.labels.numpy(), val_probs)
    gatv2_test_metrics = evaluate_predictions(test_ds.labels.numpy(), test_probs)

    all_results.append({
        "experiment_id": "02_selected_gatv2_hybrid_only",
        "model_category": "Graph Baseline",
        "description": "Selected Hybrid GATv2 (Epoch 21) pooled graph representation only",
        "parameters": count_trainable_parameters(gatv2_model),
        "best_epoch": 21,
        "val_loss": 0.4576,
        "val_acc": gatv2_val_metrics["accuracy"],
        "val_sens": gatv2_val_metrics["sensitivity"],
        "val_spec": gatv2_val_metrics["specificity"],
        "val_f1": gatv2_val_metrics["cancer_f1"],
        "test_acc": gatv2_test_metrics["accuracy"],
        "test_sens": gatv2_test_metrics["sensitivity"],
        "test_sens_ci": gatv2_test_metrics["sensitivity_ci"],
        "test_spec": gatv2_test_metrics["specificity"],
        "test_spec_ci": gatv2_test_metrics["specificity_ci"],
        "test_prec": gatv2_test_metrics["precision"],
        "test_f1": gatv2_test_metrics["cancer_f1"],
        "test_macro_f1": gatv2_test_metrics["macro_f1"],
        "test_weighted_f1": gatv2_test_metrics["weighted_f1"],
        "test_roc_auc": gatv2_test_metrics["roc_auc"],
        "test_roc_auc_ci": gatv2_test_metrics["roc_auc_ci"],
        "test_pr_auc": gatv2_test_metrics["pr_auc"],
        "confusion_matrix": gatv2_test_metrics["confusion_matrix"],
        "training_strategy": "Pre-trained scratch GATv2 (frozen)"
    })

    # -------------------------------------------------------------
    # Experiment 3: Comparative GAT-Only Baseline (Hybrid GAT, Epoch 26)
    # -------------------------------------------------------------
    print("\n[Ablation 3/14] Evaluating Comparative GAT-Only Baseline (Hybrid GAT, Epoch 26)...")
    gat_ckpt = torch.load("models/graph_hybrid_gat_best.pth", map_location=device, weights_only=False)
    gat_model = GraphClassifier(graph_type="hybrid", gnn_type="gat", pooling="mean", k=8).to(device)
    gat_model.load_state_dict(gat_ckpt["model_state_dict"])
    gat_model.eval()

    val_gat_rep_c = val_ds_gat.graph_rep.to(device)
    test_gat_rep_c = test_ds_gat.graph_rep.to(device)
    with torch.no_grad():
        val_logits = gat_model.head(val_gat_rep_c)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        test_logits = gat_model.head(test_gat_rep_c)
        test_probs = torch.softmax(test_logits, dim=-1)[:, 1].cpu().numpy()

    gat_val_metrics = evaluate_predictions(val_ds_gat.labels.numpy(), val_probs)
    gat_test_metrics = evaluate_predictions(test_ds_gat.labels.numpy(), test_probs)

    all_results.append({
        "experiment_id": "03_comparative_gat_hybrid_only",
        "model_category": "Graph Baseline",
        "description": "Comparative Hybrid GAT (Epoch 26) pooled graph representation only",
        "parameters": count_trainable_parameters(gat_model),
        "best_epoch": 26,
        "val_loss": 0.4557,
        "val_acc": gat_val_metrics["accuracy"],
        "val_sens": gat_val_metrics["sensitivity"],
        "val_spec": gat_val_metrics["specificity"],
        "val_f1": gat_val_metrics["cancer_f1"],
        "test_acc": gat_test_metrics["accuracy"],
        "test_sens": gat_test_metrics["sensitivity"],
        "test_sens_ci": gat_test_metrics["sensitivity_ci"],
        "test_spec": gat_test_metrics["specificity"],
        "test_spec_ci": gat_test_metrics["specificity_ci"],
        "test_prec": gat_test_metrics["precision"],
        "test_f1": gat_test_metrics["cancer_f1"],
        "test_macro_f1": gat_test_metrics["macro_f1"],
        "test_weighted_f1": gat_test_metrics["weighted_f1"],
        "test_roc_auc": gat_test_metrics["roc_auc"],
        "test_roc_auc_ci": gat_test_metrics["roc_auc_ci"],
        "test_pr_auc": gat_test_metrics["pr_auc"],
        "confusion_matrix": gat_test_metrics["confusion_matrix"],
        "training_strategy": "Pre-trained scratch GAT (frozen)"
    })

    # Helper function to run trainable experiment
    def run_trainable_exp(
        exp_id: str,
        category: str,
        desc: str,
        model: nn.Module,
        fwd_fn: Any,
        ckpt_name: str,
        t_loader: DataLoader = train_loader,
        v_loader: DataLoader = val_loader,
        te_loader: DataLoader = test_loader,
        lr: float = 1e-3
    ):
        print(f"\n[Ablation: {exp_id}] Training {desc}...")
        save_p = f"models/{ckpt_name}.pth"
        start_t = time.time()
        val_m, hist = train_single_model(
            model=model,
            model_name=exp_id,
            train_loader=t_loader,
            val_loader=v_loader,
            device=device,
            forward_fn=fwd_fn,
            epochs=60,
            lr=lr,
            patience=15,
            save_path=save_p
        )
        elapsed = time.time() - start_t
        all_histories.extend(hist)

        # Load best checkpoint strictly for test evaluation
        best_ckpt = torch.load(save_p, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_m = evaluate_model_on_test(model, te_loader, device, fwd_fn)

        all_results.append({
            "experiment_id": exp_id,
            "model_category": category,
            "description": desc,
            "parameters": count_trainable_parameters(model),
            "best_epoch": val_m["best_epoch"],
            "val_loss": val_m["val_loss"],
            "val_acc": val_m["val_acc"],
            "val_sens": val_m["val_sens"],
            "val_spec": val_m["val_spec"],
            "val_f1": val_m["val_f1"],
            "test_acc": test_m["accuracy"],
            "test_sens": test_m["sensitivity"],
            "test_sens_ci": test_m["sensitivity_ci"],
            "test_spec": test_m["specificity"],
            "test_spec_ci": test_m["specificity_ci"],
            "test_prec": test_m["precision"],
            "test_f1": test_m["cancer_f1"],
            "test_macro_f1": test_m["macro_f1"],
            "test_weighted_f1": test_m["weighted_f1"],
            "test_roc_auc": test_m["roc_auc"],
            "test_roc_auc_ci": test_m["roc_auc_ci"],
            "test_pr_auc": test_m["pr_auc"],
            "confusion_matrix": test_m["confusion_matrix"],
            "training_strategy": "Frozen feature extractors + scratch fusion head",
            "training_time_sec": round(elapsed, 2)
        })

    # -------------------------------------------------------------
    # Experiment 4: ViT + GAT Concatenation Fusion
    # -------------------------------------------------------------
    set_seed(42)
    m_concat = ConcatenationFusion(vit_dim=192, graph_dim=192)
    run_trainable_exp(
        "04_hybrid_vit_gat_concat",
        "Hybrid Fusion",
        "ViT CLS (192) + GATv2 Graph (192) Concatenation Fusion [B, 384]",
        m_concat,
        fwd_concat,
        "hybrid_vit_gat_concat_best"
    )

    # -------------------------------------------------------------
    # Experiment 5: ViT + GAT Gated Fusion
    # -------------------------------------------------------------
    set_seed(42)
    m_gated = GatedFusion(vit_dim=192, graph_dim=192)
    run_trainable_exp(
        "05_hybrid_vit_gat_gated",
        "Hybrid Fusion",
        "ViT CLS (192) + GATv2 Graph (192) Gated Fusion [B, 192]",
        m_gated,
        fwd_gated,
        "hybrid_vit_gat_gated_best"
    )

    # -------------------------------------------------------------
    # Experiment 6: ViT + GAT Residual Fusion
    # -------------------------------------------------------------
    set_seed(42)
    m_residual = ResidualFusion(vit_dim=192, graph_dim=192)
    run_trainable_exp(
        "06_hybrid_vit_gat_residual",
        "Hybrid Fusion",
        "ViT CLS (192) + GATv2 Graph (192) Residual Fusion [B, 192]",
        m_residual,
        fwd_residual,
        "hybrid_vit_gat_residual_best"
    )

    # -------------------------------------------------------------
    # Experiment 7: Handcrafted Features Only (Combined 143 dims)
    # -------------------------------------------------------------
    set_seed(42)
    m_hc_all = HandcraftedOnlyClassifier(in_dim=143)
    run_trainable_exp(
        "07_handcrafted_all_143",
        "Handcrafted Baseline",
        "All Handcrafted Features Only (LBP 59 + GLCM 24 + Color 46 + Edge 14 = 143 dims)",
        m_hc_all,
        fwd_hc_only,
        "handcrafted_all_143_best"
    )

    # Sub-family ablations
    sub_families = [
        ("07a_handcrafted_lbp_only", "LBP Texture Only (59 dims)", slice(0, 59), 59),
        ("07b_handcrafted_glcm_only", "GLCM Texture Only (24 dims)", slice(59, 83), 24),
        ("07c_handcrafted_color_only", "Color Moments & Histograms Only (46 dims)", slice(83, 129), 46),
        ("07d_handcrafted_edge_only", "Edge & Gradient Statistics Only (14 dims)", slice(129, 143), 14),
    ]

    for sub_id, sub_desc, s_slice, in_d in sub_families:
        set_seed(42)
        sub_train_ds, sub_val_ds, sub_test_ds = load_dataset_splits(gnn_name="hybrid_gatv2", hc_slice=s_slice)
        sub_t_loader = DataLoader(sub_train_ds, batch_size=16, shuffle=True)
        sub_v_loader = DataLoader(sub_val_ds, batch_size=16, shuffle=False)
        sub_te_loader = DataLoader(sub_test_ds, batch_size=16, shuffle=False)

        sub_model = HandcraftedOnlyClassifier(in_dim=in_d)
        run_trainable_exp(
            sub_id,
            "Handcrafted Sub-family",
            sub_desc,
            sub_model,
            fwd_hc_only,
            f"{sub_id}_best",
            t_loader=sub_t_loader,
            v_loader=sub_v_loader,
            te_loader=sub_te_loader
        )

    # -------------------------------------------------------------
    # Experiment 8: ViT + Handcrafted Features (Bi-modal)
    # -------------------------------------------------------------
    set_seed(42)
    m_vit_hc = BiModalHandcraftedFusion(deep_dim=192, hc_dim=143, hc_proj_dim=64)
    run_trainable_exp(
        "08_vit_plus_handcrafted",
        "Bi-Modal Fusion",
        "ViT CLS (192) + Handcrafted Projection (64) [B, 256]",
        m_vit_hc,
        fwd_vit_hc,
        "vit_plus_handcrafted_best"
    )

    # -------------------------------------------------------------
    # Experiment 9: GAT + Handcrafted Features (Bi-modal)
    # -------------------------------------------------------------
    set_seed(42)
    m_gat_hc = BiModalHandcraftedFusion(deep_dim=192, hc_dim=143, hc_proj_dim=64)
    run_trainable_exp(
        "09_gat_plus_handcrafted",
        "Bi-Modal Fusion",
        "GATv2 Graph (192) + Handcrafted Projection (64) [B, 256]",
        m_gat_hc,
        fwd_gat_hc,
        "gat_plus_handcrafted_best"
    )

    # -------------------------------------------------------------
    # Experiment 10: ViT + GAT + Handcrafted Features (Tri-Modal Concat)
    # -------------------------------------------------------------
    set_seed(42)
    m_tri_concat = TriModalConcatFusion(vit_dim=192, graph_dim=192, hc_dim=143, hc_proj_dim=64)
    run_trainable_exp(
        "10_trimodal_vit_gat_hc_concat",
        "Tri-Modal Fusion",
        "ViT CLS (192) + GATv2 Graph (192) + Handcrafted (64) Concatenation [B, 448]",
        m_tri_concat,
        fwd_trimodal_concat,
        "trimodal_vit_gat_hc_concat_best"
    )

    # -------------------------------------------------------------
    # Experiment 11: ViT + GAT + Handcrafted Features (Tri-Modal Gated)
    # -------------------------------------------------------------
    set_seed(42)
    m_tri_gated = TriModalGatedFusion(vit_dim=192, graph_dim=192, hc_dim=143, latent_dim=128)
    run_trainable_exp(
        "11_trimodal_vit_gat_hc_gated",
        "Tri-Modal Fusion",
        "ViT CLS (192) + GATv2 Graph (192) + Handcrafted (143) Modality-Gated [B, 128]",
        m_tri_gated,
        fwd_trimodal_gated,
        "trimodal_vit_gat_hc_gated_best"
    )

    # -------------------------------------------------------------
    # Experiment 12: Multi-Scale ViT Representation
    # -------------------------------------------------------------
    set_seed(42)
    m_ms_vit = MultiScaleViTClassifier(embed_dim=192)
    run_trainable_exp(
        "12_multiscale_vit_only",
        "Multi-Scale Representation",
        "Multi-Scale ViT (Global CLS + 7x7 Regional + Fine Patch Pool, 576 dims)",
        m_ms_vit,
        fwd_multiscale_vit,
        "multiscale_vit_best"
    )

    # -------------------------------------------------------------
    # Experiment 13: Multi-Scale Hybrid Representation
    # -------------------------------------------------------------
    set_seed(42)
    m_ms_hybrid = MultiScaleHybridClassifier(embed_dim=192, graph_dim=192)
    run_trainable_exp(
        "13_multiscale_hybrid",
        "Multi-Scale Representation",
        "Multi-Scale ViT (576 dims) + GATv2 Graph (192 dims) = 768 dims",
        m_ms_hybrid,
        fwd_multiscale_hybrid,
        "multiscale_hybrid_best"
    )

    # -------------------------------------------------------------
    # Experiment 14: Strategy B - Controlled End-to-End Fine-Tuning
    # -------------------------------------------------------------
    print("\n[Ablation 14/14] Evaluating Strategy B: Controlled End-to-End Fine-Tuning...")
    set_seed(42)
    # Instantiate End-to-End Architecture: ViT backbone + GAT backbone + ConcatenationFusion Head
    vit_e2e = build_vit_scratch(embed_dim=192, depth=4, num_heads=3, num_classes=2).to(device)
    vit_e2e.load_state_dict(vit_ckpt["model_state_dict"])

    gat_e2e = GraphClassifier(graph_type="hybrid", gnn_type="gatv2", pooling="mean", k=8).to(device)
    gat_e2e.load_state_dict(gatv2_ckpt["model_state_dict"])

    fusion_e2e = ConcatenationFusion(vit_dim=192, graph_dim=192).to(device)

    # Dataset with deterministic training transforms and data loader
    t_transform = get_transforms("train", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")
    v_transform = get_transforms("val", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")
    te_transform = get_transforms("test", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")

    e2e_train_ds = OralCancerDataset("data/splits_primary/train.csv", transform=t_transform)
    e2e_val_ds = OralCancerDataset("data/splits_primary/val.csv", transform=v_transform)
    e2e_test_ds = OralCancerDataset("data/splits_primary/test.csv", transform=te_transform)

    e2e_train_loader = DataLoader(e2e_train_ds, batch_size=8, shuffle=True)
    e2e_val_loader = DataLoader(e2e_val_ds, batch_size=8, shuffle=False)
    e2e_test_loader = DataLoader(e2e_test_ds, batch_size=8, shuffle=False)

    # Optimizer with differential learning rates: 1e-5 for backbones, 5e-4 for fusion head
    param_groups = [
        {"params": vit_e2e.parameters(), "lr": 1e-5},
        {"params": gat_e2e.parameters(), "lr": 1e-5},
        {"params": fusion_e2e.parameters(), "lr": 5e-4},
    ]
    opt_e2e = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    sched_e2e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e2e, T_max=15)

    w0 = 486.0 / (2.0 * 160.0)
    w1 = 486.0 / (2.0 * 326.0)
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    crit_e2e = nn.CrossEntropyLoss(weight=weights)

    best_e2e_val_loss = float("inf")
    best_e2e_epoch = 0
    best_e2e_metrics = {}
    e2e_save_p = "models/hybrid_e2e_finetuned_best.pth"

    start_e2e_t = time.time()
    for ep in range(1, 16):
        vit_e2e.train()
        gat_e2e.train()
        fusion_e2e.train()
        t_loss = 0.0
        t_corr = 0
        tot_t = 0

        for imgs, lbls in e2e_train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_e2e.zero_grad()

            v_feats = vit_e2e.forward_features(imgs)
            v_cls = v_feats["cls"]
            v_patches = v_feats["patches"]

            g_rep = gat_e2e.get_graph_representation(v_patches)
            logits, _ = fusion_e2e(v_cls, g_rep)

            loss = crit_e2e(logits, lbls)
            loss.backward()
            opt_e2e.step()

            t_loss += loss.item() * len(lbls)
            preds = torch.argmax(logits, dim=-1)
            t_corr += (preds == lbls).sum().item()
            tot_t += len(lbls)

        sched_e2e.step()
        t_loss /= tot_t
        t_acc = t_corr / tot_t

        # Validation
        vit_e2e.eval()
        gat_e2e.eval()
        fusion_e2e.eval()
        v_loss = 0.0
        v_corr = 0
        tot_v = 0
        v_preds, v_targets = [], []

        with torch.no_grad():
            for imgs, lbls in e2e_val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                v_feats = vit_e2e.forward_features(imgs)
                g_rep = gat_e2e.get_graph_representation(v_feats["patches"])
                logits, _ = fusion_e2e(v_feats["cls"], g_rep)
                loss = crit_e2e(logits, lbls)

                v_loss += loss.item() * len(lbls)
                preds = torch.argmax(logits, dim=-1)
                v_corr += (preds == lbls).sum().item()
                tot_v += len(lbls)
                v_preds.extend(preds.cpu().numpy().tolist())
                v_targets.extend(lbls.cpu().numpy().tolist())

        v_loss /= tot_v
        v_acc = v_corr / tot_v

        targets_np = np.array(v_targets)
        preds_np = np.array(v_preds)
        tp = int(np.sum((preds_np == 1) & (targets_np == 1)))
        tn = int(np.sum((preds_np == 0) & (targets_np == 0)))
        fp = int(np.sum((preds_np == 1) & (targets_np == 0)))
        fn = int(np.sum((preds_np == 0) & (targets_np == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        all_histories.append({
            "model_name": "14_hybrid_e2e_finetuned",
            "epoch": ep,
            "train_loss": t_loss,
            "train_acc": t_acc,
            "val_loss": v_loss,
            "val_acc": v_acc,
            "val_sens": sens,
            "val_spec": spec,
            "val_prec": prec,
            "val_f1": f1
        })

        if v_loss < best_e2e_val_loss:
            best_e2e_val_loss = v_loss
            best_e2e_epoch = ep
            best_e2e_metrics = {
                "val_loss": v_loss,
                "val_acc": v_acc,
                "val_sens": sens,
                "val_spec": spec,
                "val_prec": prec,
                "val_f1": f1,
                "best_epoch": ep
            }
            torch.save({
                "vit_state_dict": vit_e2e.state_dict(),
                "gat_state_dict": gat_e2e.state_dict(),
                "fusion_state_dict": fusion_e2e.state_dict(),
                "epoch": ep,
                "best_val_metrics": best_e2e_metrics
            }, e2e_save_p)

    # Evaluate best E2E checkpoint on test set
    e2e_ckpt = torch.load(e2e_save_p, map_location=device, weights_only=False)
    vit_e2e.load_state_dict(e2e_ckpt["vit_state_dict"])
    gat_e2e.load_state_dict(e2e_ckpt["gat_state_dict"])
    fusion_e2e.load_state_dict(e2e_ckpt["fusion_state_dict"])

    vit_e2e.eval()
    gat_e2e.eval()
    fusion_e2e.eval()
    e2e_te_targets, e2e_te_probs = [], []

    with torch.no_grad():
        for imgs, lbls in e2e_test_loader:
            imgs = imgs.to(device)
            v_feats = vit_e2e.forward_features(imgs)
            g_rep = gat_e2e.get_graph_representation(v_feats["patches"])
            logits, _ = fusion_e2e(v_feats["cls"], g_rep)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            e2e_te_targets.extend(lbls.numpy().tolist())
            e2e_te_probs.extend(probs.cpu().numpy().tolist())

    e2e_test_m = evaluate_predictions(np.array(e2e_te_targets), np.array(e2e_te_probs))

    all_results.append({
        "experiment_id": "14_hybrid_e2e_finetuned",
        "model_category": "Strategy B End-to-End",
        "description": "ViT + GATv2 + Concat Fusion fine-tuned end-to-end with low LR",
        "parameters": count_trainable_parameters(vit_e2e) + count_trainable_parameters(gat_e2e) + count_trainable_parameters(fusion_e2e),
        "best_epoch": best_e2e_metrics["best_epoch"],
        "val_loss": best_e2e_metrics["val_loss"],
        "val_acc": best_e2e_metrics["val_acc"],
        "val_sens": best_e2e_metrics["val_sens"],
        "val_spec": best_e2e_metrics["val_spec"],
        "val_f1": best_e2e_metrics["val_f1"],
        "test_acc": e2e_test_m["accuracy"],
        "test_sens": e2e_test_m["sensitivity"],
        "test_sens_ci": e2e_test_m["sensitivity_ci"],
        "test_spec": e2e_test_m["specificity"],
        "test_spec_ci": e2e_test_m["specificity_ci"],
        "test_prec": e2e_test_m["precision"],
        "test_f1": e2e_test_m["cancer_f1"],
        "test_macro_f1": e2e_test_m["macro_f1"],
        "test_weighted_f1": e2e_test_m["weighted_f1"],
        "test_roc_auc": e2e_test_m["roc_auc"],
        "test_roc_auc_ci": e2e_test_m["roc_auc_ci"],
        "test_pr_auc": e2e_test_m["pr_auc"],
        "confusion_matrix": e2e_test_m["confusion_matrix"],
        "training_strategy": "Strategy B: Controlled End-to-End Fine-Tuning (Differential LR)",
        "training_time_sec": round(time.time() - start_e2e_t, 2)
    })

    # -------------------------------------------------------------
    # Save Deliverables: Reports, CSVs, Manifests
    # -------------------------------------------------------------
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training history CSV
    df_hist = pd.DataFrame(all_histories)
    hist_path = reports_dir / "phase5_phase6_training_history.csv"
    df_hist.to_csv(hist_path, index=False)
    print(f"\nSaved training history -> {hist_path}")

    # 2. Ablation results CSV
    df_results = pd.DataFrame(all_results)
    ablation_path = reports_dir / "phase5_phase6_ablation_results.csv"
    df_results.to_csv(ablation_path, index=False)
    print(f"Saved ablation results -> {ablation_path}")

    # 3. Final comparison CSV (including Phase 2 Baseline, Phase 3 ViT, Phase 4 GAT/GATv2)
    comparison_records = []
    # Add Phase 2 baseline
    comparison_records.append({
        "Model": "Phase 2: EfficientNet-B0 Scratch Baseline",
        "Type": "CNN Baseline",
        "Test Accuracy": 0.8073,
        "Cancer Sensitivity": 0.8000,
        "Specificity": 0.8235,
        "Cancer Precision": 0.9091,
        "Cancer F1": 0.8511,
        "ROC-AUC": 0.8912,
        "PR-AUC": 0.9176,
        "Parameters": 4010110,
        "Key Distinction": "Standard CNN baseline (trained from scratch)"
    })
    for r in all_results:
        comparison_records.append({
            "Model": r["experiment_id"],
            "Type": r["model_category"],
            "Test Accuracy": round(r["test_acc"], 4),
            "Cancer Sensitivity": round(r["test_sens"], 4),
            "Specificity": round(r["test_spec"], 4),
            "Cancer Precision": round(r["test_prec"], 4),
            "Cancer F1": round(r["test_f1"], 4),
            "ROC-AUC": round(r["test_roc_auc"], 4),
            "PR-AUC": round(r["test_pr_auc"], 4),
            "Parameters": r["parameters"],
            "Key Distinction": r["description"]
        })
    df_comp = pd.DataFrame(comparison_records)
    comp_path = reports_dir / "phase5_phase6_final_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"Saved final comparison table -> {comp_path}")

    # 4. Experiment manifest JSON
    manifest_data = {
        "project": "Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI",
        "phases": "Phase 5 (ViT-GAT Hybrid Fusion) and Phase 6 (Handcrafted Feature Engineering & Ablations)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 42,
        "device": str(device),
        "dataset_split_counts": {"train": 486, "val": 108, "test": 109, "total": 703},
        "experiments_count": len(all_results),
        "results_summary": all_results
    }
    manifest_path = reports_dir / "phase5_phase6_experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved experiment manifest -> {manifest_path}")

    print("\nAll Phase 5 & 6 experiments executed and saved successfully!")


if __name__ == "__main__":
    run_all_ablations()
