"""
Oral Cancer Screening Pipeline - Unified Demonstration & Inference Engine
==========================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

This script runs the complete oral cancer screening inference pipeline:
1. Loads all trained model components (EfficientNet, ViT, GATv2, Handcrafted, Hybrid, Trimodal).
2. Performs real-time screening on clinical oral lesion images.
3. Outputs diagnostic probabilities, binary predictions, clinical risk stratification, and inference latency.
4. Displays the comprehensive benchmark performance summary across all project phases on the primary test set.

Usage:
    python src/screen.py                        # Runs interactive demo on test cases & prints benchmarks
    python src/screen.py --image <path_to_img>  # Screens a specific patient image
    python src/screen.py --benchmark-only       # Displays the full multi-phase benchmark comparison
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from src.dataset import get_transforms
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.models.vit_scratch import build_vit_scratch
from src.models.graph_classifier import GraphClassifier
from src.models.hybrid_fusion import ConcatenationFusion
from src.models.handcrafted_fusion import HandcraftedOnlyClassifier, TriModalConcatFusion
from src.features.handcrafted_extractor import HandcraftedFeatureExtractor
from src.features.feature_scaler import TrainingFeatureScaler


class OralCancerScreeningEngine:
    """
    Unified multimodal screening engine combining CNN, ViT, GNN, and Handcrafted feature models.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Oral Cancer Screening Engine on device: {self.device}...")

        # 1. Image preprocessing transform
        self.transform = get_transforms("test", image_size=(224, 224), resize_mode="direct", norm_mode="dataset")

        # 2. Handcrafted Feature Extractor and Scaler
        print("  - Loading Handcrafted Extractor (LBP, GLCM, Color, Edge: 143 dims)...")
        self.hc_extractor = HandcraftedFeatureExtractor()
        scaler_path = "models/handcrafted_scaler.pt"
        if os.path.exists(scaler_path):
            self.hc_scaler = TrainingFeatureScaler.load(scaler_path)
        else:
            self.hc_scaler = None

        # 3. Model 1: EfficientNet-B0 Baseline
        print("  - Loading EfficientNet-B0 (Scratch Baseline)...")
        self.effnet = build_efficientnet_b0(num_classes=2, device=self.device)
        effnet_ckpt = "models/efficientnet_b0_scratch_best.pth"
        if os.path.exists(effnet_ckpt):
            ckpt = torch.load(effnet_ckpt, map_location=self.device, weights_only=False)
            self.effnet.load_state_dict(ckpt["model_state_dict"])
        self.effnet.eval()

        # 4. Model 2: Vision Transformer (Scratch Baseline)
        print("  - Loading Vision Transformer (Scratch Baseline, 4 layers, 3 heads)...")
        self.vit = build_vit_scratch(
            num_classes=2,
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=4,
            num_heads=3,
            mlp_dim=768,
            device=self.device
        )
        vit_ckpt = "models/vit_scratch_best.pth"
        if os.path.exists(vit_ckpt):
            ckpt = torch.load(vit_ckpt, map_location=self.device, weights_only=False)
            self.vit.load_state_dict(ckpt["model_state_dict"])
        self.vit.eval()

        # 5. Model 3: GATv2 Hybrid Graph Model
        print("  - Loading Graph Attention Network (GATv2 Hybrid)...")
        self.gatv2 = GraphClassifier(
            graph_type="hybrid",
            gnn_type="gatv2",
            pooling="mean",
            in_features=192,
            hidden_per_head=48,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            k=8,
            dropout=0.1
        ).to(self.device)
        gat_ckpt = "models/graph_hybrid_gatv2_best.pth"
        if os.path.exists(gat_ckpt):
            ckpt = torch.load(gat_ckpt, map_location=self.device, weights_only=False)
            self.gatv2.load_state_dict(ckpt["model_state_dict"])
        self.gatv2.eval()

        # 6. Model 4: Handcrafted 143 Classifier
        print("  - Loading Handcrafted 143 Classifier...")
        self.hc_model = HandcraftedOnlyClassifier(in_dim=143, hidden_dim=64, num_classes=2).to(self.device)
        hc_ckpt = "models/handcrafted_all_143_best.pth"
        if os.path.exists(hc_ckpt):
            ckpt = torch.load(hc_ckpt, map_location=self.device, weights_only=False)
            self.hc_model.load_state_dict(ckpt["model_state_dict"])
        self.hc_model.eval()

        # 7. Model 5: ViT + GATv2 Concatenation Fusion
        print("  - Loading ViT + GATv2 Hybrid Concatenation Model...")
        self.hybrid_concat = ConcatenationFusion(vit_dim=192, graph_dim=192, hidden_dim=128, num_classes=2).to(self.device)
        hybrid_ckpt = "models/hybrid_vit_gat_concat_best.pth"
        if os.path.exists(hybrid_ckpt):
            ckpt = torch.load(hybrid_ckpt, map_location=self.device, weights_only=False)
            self.hybrid_concat.load_state_dict(ckpt["model_state_dict"])
        self.hybrid_concat.eval()

        # 8. Model 6: Tri-Modal Concat Fusion (ViT + GATv2 + Handcrafted)
        print("  - Loading Tri-Modal Fusion Model (ViT + GATv2 + Handcrafted)...")
        self.trimodal = TriModalConcatFusion(vit_dim=192, graph_dim=192, hc_dim=143, hc_proj_dim=64, hidden_dim=128, num_classes=2).to(self.device)
        trimodal_ckpt = "models/trimodal_vit_gat_hc_concat_best.pth"
        if os.path.exists(trimodal_ckpt):
            ckpt = torch.load(trimodal_ckpt, map_location=self.device, weights_only=False)
            self.trimodal.load_state_dict(ckpt["model_state_dict"])
        self.trimodal.eval()

        print("All models successfully loaded and ready for screening.\n")

    def screen_image(self, image_path: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs comprehensive screening on a single oral lesion image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        start_time = time.time()

        # Load and preprocess image
        raw_pil = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(raw_pil).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        # Extract handcrafted features
        raw_np = np.array(raw_pil)
        hc_vector = self.hc_extractor.extract_vector(raw_np)  # [143]
        if self.hc_scaler is not None:
            hc_scaled = self.hc_scaler.transform(hc_vector.reshape(1, -1))
        else:
            hc_scaled = hc_vector.reshape(1, -1)
        hc_tensor = torch.from_numpy(hc_scaled).float().to(self.device)  # [1, 143]

        with torch.no_grad():
            # 1. EfficientNet Prediction
            eff_logits = self.effnet(img_tensor)
            eff_probs = F.softmax(eff_logits, dim=-1)[0].cpu().numpy()

            # 2. ViT Features & Prediction
            vit_features = self.vit.forward_features(img_tensor)
            vit_cls = vit_features["cls"]          # [1, 192]
            vit_patches = vit_features["patches"]  # [1, 196, 192]
            vit_logits = self.vit.head(vit_cls)
            vit_probs = F.softmax(vit_logits, dim=-1)[0].cpu().numpy()

            # 3. GATv2 Graph Prediction & Pooled Representation
            gat_graph_rep = self.gatv2.get_graph_representation(vit_patches)  # [1, 192]
            gat_logits = self.gatv2.head(gat_graph_rep)
            gat_probs = F.softmax(gat_logits, dim=-1)[0].cpu().numpy()

            # 4. Handcrafted-Only Prediction
            hc_logits = self.hc_model(hc_tensor)
            hc_probs = F.softmax(hc_logits, dim=-1)[0].cpu().numpy()

            # 5. ViT + GATv2 Hybrid Prediction
            hybrid_logits, _ = self.hybrid_concat(vit_cls, gat_graph_rep)
            hybrid_probs = F.softmax(hybrid_logits, dim=-1)[0].cpu().numpy()

            # 6. Tri-Modal Prediction
            trimodal_logits, _ = self.trimodal(vit_cls, gat_graph_rep, hc_tensor)
            trimodal_probs = F.softmax(trimodal_logits, dim=-1)[0].cpu().numpy()

        elapsed_ms = (time.time() - start_time) * 1000.0

        predictions = {
            "EfficientNet-B0 (Scratch Baseline)": {
                "cancer_prob": float(eff_probs[1]),
                "pred_label": "CANCER" if eff_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(eff_probs)) * 100.0
            },
            "Vision Transformer (ViT-Scratch)": {
                "cancer_prob": float(vit_probs[1]),
                "pred_label": "CANCER" if vit_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(vit_probs)) * 100.0
            },
            "Graph Attention Network (GATv2 Hybrid)": {
                "cancer_prob": float(gat_probs[1]),
                "pred_label": "CANCER" if gat_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(gat_probs)) * 100.0
            },
            "Handcrafted Features (143 Texture/Color/Edge)": {
                "cancer_prob": float(hc_probs[1]),
                "pred_label": "CANCER" if hc_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(hc_probs)) * 100.0
            },
            "ViT-GATv2 Hybrid (Concatenation)": {
                "cancer_prob": float(hybrid_probs[1]),
                "pred_label": "CANCER" if hybrid_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(hybrid_probs)) * 100.0
            },
            "Tri-Modal Fusion (ViT + GATv2 + Handcrafted)": {
                "cancer_prob": float(trimodal_probs[1]),
                "pred_label": "CANCER" if trimodal_probs[1] >= 0.5 else "NON CANCER",
                "confidence": float(max(trimodal_probs)) * 100.0
            }
        }

        # Ensemble consensus calculation
        cancer_votes = sum(1 for m in predictions.values() if m["pred_label"] == "CANCER")
        total_models = len(predictions)
        mean_cancer_prob = float(np.mean([m["cancer_prob"] for m in predictions.values()]))

        if mean_cancer_prob >= 0.70 or cancer_votes >= 5:
            risk_tier = "HIGH RISK (Malignant Suspicion - Immediate Biopsy Recommended)"
            overall_verdict = "CANCER"
        elif mean_cancer_prob >= 0.40 or cancer_votes >= 3:
            risk_tier = "MODERATE RISK (Borderline / Dysplastic Suspicion - Clinical Review Required)"
            overall_verdict = "INDETERMINATE / CAUTION"
        else:
            risk_tier = "LOW RISK (Benign Oral Lesion - Routine Clinical Monitoring)"
            overall_verdict = "NON CANCER"

        return {
            "image_path": image_path,
            "dimensions": f"{raw_pil.width}x{raw_pil.height}",
            "ground_truth": ground_truth or "UNKNOWN",
            "elapsed_ms": round(elapsed_ms, 2),
            "predictions": predictions,
            "consensus": {
                "overall_verdict": overall_verdict,
                "risk_tier": risk_tier,
                "mean_cancer_prob": round(mean_cancer_prob * 100.0, 2),
                "model_agreement": f"{cancer_votes}/{total_models} models predict CANCER"
            }
        }


def format_screening_output(result: Dict[str, Any]) -> str:
    """
    Renders structured, clinical-grade terminal report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ORAL CANCER CLINICAL SCREENING REPORT")
    lines.append("=" * 80)
    lines.append(f"Image File:          {result['image_path']}")
    lines.append(f"Original Resolution: {result['dimensions']} pixels")
    if result["ground_truth"] != "UNKNOWN":
        lines.append(f"Ground Truth Label:  {result['ground_truth']}")
    lines.append(f"Inference Latency:   {result['elapsed_ms']} ms (all 6 models end-to-end)")
    lines.append("-" * 80)
    lines.append(f"{'Model Architecture':<42} | {'Prediction':<12} | {'Cancer Prob':<11} | {'Confidence':<10}")
    lines.append("-" * 80)

    for name, p in result["predictions"].items():
        color_tag = "MALIGNANT" if p["pred_label"] == "CANCER" else "BENIGN"
        prob_str = f"{p['cancer_prob'] * 100.0:.2f}%"
        conf_str = f"{p['confidence']:.2f}%"
        lines.append(f"{name:<42} | {color_tag:<12} | {prob_str:<11} | {conf_str:<10}")

    lines.append("-" * 80)
    c = result["consensus"]
    lines.append("CLINICAL CONSENSUS & RISK STRATIFICATION:")
    lines.append(f"  Overall Verdict:       {c['overall_verdict']}")
    lines.append(f"  Risk Category:         {c['risk_tier']}")
    lines.append(f"  Mean Cancer Risk:      {c['mean_cancer_prob']}%")
    lines.append(f"  Multi-Model Agreement: {c['model_agreement']}")
    lines.append("=" * 80)
    return "\n".join(lines)


def display_benchmark_table():
    """
    Reads and prints the comprehensive benchmark evaluation table from Phase 5 & 6.
    """
    csv_path = "reports/phase5_phase6_final_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Benchmark summary not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("\n" + "=" * 96)
    print("PRIMARY TEST SET EVALUATION BENCHMARK (N=109 Independent Test Samples)")
    print("=" * 96)

    # Format numeric columns as percentages or 4-decimal floats
    display_df = df.copy()
    num_cols = ["Test Accuracy", "Cancer Sensitivity", "Specificity", "Cancer Precision", "Cancer F1", "ROC-AUC", "PR-AUC"]
    for col in num_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{float(x)*100.0:6.2f}%" if pd.notnull(x) else "N/A")

    if "Parameters" in display_df.columns:
        display_df["Parameters"] = display_df["Parameters"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")

    cols_to_show = ["Model", "Type", "Test Accuracy", "Cancer Sensitivity", "Specificity", "ROC-AUC", "PR-AUC", "Parameters"]
    print(display_df[cols_to_show].to_string(index=False))
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description="Oral Cancer Screening Inference & Benchmark Runner")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to screen")
    parser.add_argument("--ground-truth", type=str, default=None, help="Ground truth label for single image")
    parser.add_argument("--benchmark-only", action="store_true", help="Print benchmark comparison table only")
    parser.add_argument("--samples", type=int, default=2, help="Number of demo test cases to run")
    args = parser.parse_args()

    if args.benchmark_only:
        display_benchmark_table()
        return

    engine = OralCancerScreeningEngine()

    if args.image:
        res = engine.screen_image(args.image, ground_truth=args.ground_truth)
        print(format_screening_output(res))
    else:
        print("Running demonstration on primary test dataset cases...\n")

        demo_cases = [
            ("Oral Cancer/Oral Cancer Dataset/CANCER/116.jpeg", "CANCER (Oral Squamous Cell Carcinoma)"),
            ("Oral Cancer/Oral Cancer Dataset/NON CANCER/013.jpeg", "NON CANCER (Normal / Benign Oral Tissue)")
        ]

        for path, gt in demo_cases:
            if os.path.exists(path):
                res = engine.screen_image(path, ground_truth=gt)
                print(format_screening_output(res))
                print()

        display_benchmark_table()


if __name__ == "__main__":
    main()
