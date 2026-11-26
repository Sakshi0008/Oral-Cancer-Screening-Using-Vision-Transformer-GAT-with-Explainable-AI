"""
ORALSCAN AI - Performance Metrics Display Tool
==============================================
Prints publication-grade performance tables across all models and benchmarks in terminal.
"""

import sys
import pandas as pd
from pathlib import Path

CSV_PATH = Path("reports/phase5_phase6_final_comparison.csv")

def format_percentage(val):
    try:
        f = float(val)
        return f"{f * 100:.2f}%"
    except:
        return val

def format_params(val):
    try:
        n = int(val)
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)
    except:
        return str(val)

def main():
    if not CSV_PATH.exists():
        print(f"Error: Metrics file not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    
    # Format columns for clinical presentation
    display_df = pd.DataFrame()
    display_df["Model Architecture"] = df["Model"].apply(lambda x: x.replace("_", " ").title())
    display_df["Category"] = df["Type"]
    display_df["Accuracy"] = df["Test Accuracy"].apply(format_percentage)
    display_df["Sensitivity"] = df["Cancer Sensitivity"].apply(format_percentage)
    display_df["Specificity"] = df["Specificity"].apply(format_percentage)
    display_df["Precision"] = df["Cancer Precision"].apply(format_percentage)
    display_df["F1-Score"] = df["Cancer F1"].apply(format_percentage)
    display_df["ROC-AUC"] = df["ROC-AUC"].apply(format_percentage)
    display_df["Params"] = df["Parameters"].apply(format_params)

    print("=" * 115)
    print(" ORALSCAN AI: COMPREHENSIVE PERFORMANCE BENCHMARK MATRIX (TEST SET, N=109)")
    print("=" * 115)
    print(display_df.to_string(index=False))
    print("=" * 115)
    print("Key Observations:")
    print("  1. Primary Research Architecture (ViT + GATv2 Concat): 74.31% Accuracy, 73.33% Sensitivity, 87.30% Precision.")
    print("  2. Handcrafted Texture Baseline (LBP+GLCM+Color+Edge): 87.16% Accuracy (highest empirical small-sample baseline).")
    print("  3. Multimodal Integration (ViT / GATv2 + Handcrafted): 80.73% Accuracy, 80.00% Sensitivity, 90.91% Precision.")
    print("=" * 115)

if __name__ == "__main__":
    main()
