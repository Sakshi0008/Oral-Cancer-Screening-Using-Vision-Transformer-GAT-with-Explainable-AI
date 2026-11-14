"""
Extract and Cache Graph Representations
======================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Extracts pooled graph representations [N, 192] from the frozen scratch-trained GNN checkpoints:
- models/graph_hybrid_gatv2_best.pth (Primary backbone, Val F1=0.8708)
- models/graph_hybrid_gat_best.pth   (Comparative baseline, Val Loss=0.4557)

Operates on the pre-extracted patch tokens:
- data/features_vit_scratch/train_patch_features.pt
- data/features_vit_scratch/val_patch_features.pt
- data/features_vit_scratch/test_patch_features.pt

Saves outputs to:
- data/features_graph/
  - train_graph_hybrid_gatv2.pt
  - val_graph_hybrid_gatv2.pt
  - test_graph_hybrid_gatv2.pt
  - train_graph_hybrid_gat.pt
  - val_graph_hybrid_gat.pt
  - test_graph_hybrid_gat.pt
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import torch

sys.path.insert(0, os.getcwd())
from src.models.graph_classifier import GraphClassifier


def load_gnn_model(checkpoint_path: str, device: torch.device) -> GraphClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GraphClassifier(
        graph_type=cfg["graph_type"],
        gnn_type=cfg["gnn_type"],
        pooling=cfg.get("pooling", "mean"),
        k=cfg.get("k", 8)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_for_checkpoint(
    checkpoint_path: str,
    gnn_name: str,
    out_dir: Path,
    device: torch.device
) -> None:
    print(f"\nLoading GNN checkpoint: {checkpoint_path}")
    model = load_gnn_model(checkpoint_path, device)

    splits = ["train", "val", "test"]
    for split in splits:
        feat_path = f"data/features_vit_scratch/{split}_patch_features.pt"
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        patch_tokens = data["patch_features"]  # [N, 196, 192]
        labels = data["labels"]
        filenames = data["filenames"]
        N = patch_tokens.shape[0]

        print(f"Extracting graph representation for '{split}' (N={N})...")
        batch_size = 16
        graph_reps = []

        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_patches = patch_tokens[i : i + batch_size].to(device)
                reps = model.get_graph_representation(batch_patches)
                graph_reps.append(reps.cpu())

        all_graph_reps = torch.cat(graph_reps, dim=0)
        assert all_graph_reps.shape == (N, 192), f"Expected ({N}, 192), got {all_graph_reps.shape}"
        assert not torch.isnan(all_graph_reps).any(), "NaN found in graph representations!"

        save_dict = {
            "split": split,
            "gnn_name": gnn_name,
            "checkpoint_path": checkpoint_path,
            "graph_representations": all_graph_reps,
            "labels": labels,
            "filenames": filenames,
        }
        out_file = out_dir / f"{split}_graph_{gnn_name}.pt"
        torch.save(save_dict, str(out_file))
        print(f"  Saved -> {out_file} (Shape: {all_graph_reps.shape})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    out_dir = Path("data/features_graph")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary selected graph backbone: hybrid_gatv2
    extract_for_checkpoint(
        "models/graph_hybrid_gatv2_best.pth",
        "hybrid_gatv2",
        out_dir,
        device
    )

    # 2. Comparative baseline graph backbone: hybrid_gat
    extract_for_checkpoint(
        "models/graph_hybrid_gat_best.pth",
        "hybrid_gat",
        out_dir,
        device
    )

    print("\nGraph representation extraction completed successfully.")


if __name__ == "__main__":
    main()
