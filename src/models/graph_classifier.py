"""
Graph Classification Model (GAT / GATv2 with Configurable Graph Strategies)
============================================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Combines:
1. Graph Construction: "spatial", "similarity", or "hybrid".
2. Graph Attention Network: GAT or GATv2 (2 layers, 4 heads, 192-dim).
3. Graph Pooling: Global Mean Pooling or Attention-based Graph Pooling.
4. Classification Head: Linear(192, 2).

Supports:
- Operating directly on cached/extracted ViT patch tokens [B, 196, 192].
- Operating end-to-end with an attached VisionTransformerScratch backbone (frozen or trainable).
- Exposes clean interfaces for Phase 5 ViT-GAT hybrid fusion.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.graph.graph_builder import build_image_graph
from src.graph.graph_pooling import build_graph_pooling
from src.models.graph_attention import build_gat, GAT
from src.models.graph_attention_v2 import build_gatv2, GATv2


class GraphClassifier(nn.Module):
    """
    Independent Graph Classification Model for Oral Cancer Screening.
    """
    def __init__(
        self,
        graph_type: str = "spatial",      # "spatial", "similarity", "hybrid"
        gnn_type: str = "gat",            # "gat" or "gatv2"
        pooling: str = "mean",            # "mean" or "attention"
        in_features: int = 192,
        hidden_per_head: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        k: int = 8,
        dropout: float = 0.1,
        vit_backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.is_pretrained = False
        self.graph_type = graph_type.lower().strip()
        self.gnn_type = gnn_type.lower().strip()
        self.pooling_type = pooling.lower().strip()
        self.k = k
        self.vit_backbone = vit_backbone

        # 1. Instantiate GNN Backbone (GAT or GATv2)
        if self.gnn_type == "gat":
            self.gnn = build_gat(
                in_features=in_features,
                hidden_features_per_head=hidden_per_head,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        elif self.gnn_type == "gatv2":
            self.gnn = build_gatv2(
                in_features=in_features,
                hidden_features_per_head=hidden_per_head,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown gnn_type '{gnn_type}'. Choose 'gat' or 'gatv2'.")

        out_features = hidden_per_head * num_heads  # 192

        # 2. Graph Pooling Layer
        self.pooling = build_graph_pooling(
            method=self.pooling_type,
            in_dim=out_features,
            hidden_dim=64,
            dropout=dropout
        )

        # 3. Final Classification Head
        self.head = nn.Linear(out_features, num_classes)

        # Initialize classification head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def freeze_backbone(self):
        """Freeze ViT backbone if attached."""
        if self.vit_backbone is not None:
            for p in self.vit_backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for end-to-end fine-tuning."""
        if self.vit_backbone is not None:
            for p in self.vit_backbone.parameters():
                p.requires_grad = True

    def forward_nodes(
        self,
        features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Forward pass through graph construction and GNN layers.

        Args:
            features: Patch tokens [B, 196, 192] or raw images [B, 3, 224, 224].
            adj: Optional precomputed adjacency matrix [B, 196, 196].

        Returns:
            updated_nodes: Node representations [B, 196, 192].
            adj: Adjacency matrix used [B, 196, 196].
            attns: List of attention weight tensors.
        """
        if features.dim() == 4:  # (B, 3, 224, 224)
            assert self.vit_backbone is not None, "Raw image passed but vit_backbone is None"
            features = self.vit_backbone.get_patch_embeddings(features)

        assert features.dim() == 3, f"Expected node features of shape (B, N, D), got {features.shape}"

        # Build adjacency matrix if not provided
        if adj is None:
            adj = build_image_graph(
                strategy=self.graph_type,
                features=features,
                k=self.k,
                add_self_loops=True
            )

        updated_nodes, attns = self.gnn(features, adj)
        return updated_nodes, adj, attns

    def get_graph_representation(
        self,
        features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract pooled graph-level representation [B, 192] for Phase 5 hybrid integration.
        """
        updated_nodes, _, _ = self.forward_nodes(features, adj)
        graph_rep = self.pooling(updated_nodes)
        return graph_rep

    def forward(
        self,
        features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard forward pass returning classification logits [B, 2].
        """
        updated_nodes, _, _ = self.forward_nodes(features, adj)
        graph_rep = self.pooling(updated_nodes)
        logits = self.head(graph_rep)
        return logits

    def forward_full(
        self,
        features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, list]]:
        """
        Full forward pass returning logits, graph representation, updated nodes, and attention weights.
        """
        updated_nodes, adj_used, attns = self.forward_nodes(features, adj)
        graph_rep = self.pooling(updated_nodes)
        logits = self.head(graph_rep)
        return {
            "logits": logits,
            "graph_rep": graph_rep,
            "node_reps": updated_nodes,
            "adj": adj_used,
            "attns": attns
        }


def build_graph_classifier(
    graph_type: str = "spatial",
    gnn_type: str = "gat",
    pooling: str = "mean",
    in_features: int = 192,
    hidden_per_head: int = 48,
    num_heads: int = 4,
    num_layers: int = 2,
    num_classes: int = 2,
    k: int = 8,
    dropout: float = 0.1,
    vit_backbone: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> GraphClassifier:
    """Factory function for GraphClassifier."""
    model = GraphClassifier(
        graph_type=graph_type,
        gnn_type=gnn_type,
        pooling=pooling,
        in_features=in_features,
        hidden_per_head=hidden_per_head,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        k=k,
        dropout=dropout,
        vit_backbone=vit_backbone
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    dummy_feats = torch.randn(2, 196, 192)
    for g_type in ["spatial", "similarity", "hybrid"]:
        for m_type in ["gat", "gatv2"]:
            clf = build_graph_classifier(graph_type=g_type, gnn_type=m_type)
            res = clf.forward_full(dummy_feats)
            print(f"[{g_type.upper():10s} + {m_type.upper():5s}] Logits: {res['logits'].shape}, GraphRep: {res['graph_rep'].shape}")
            assert res["logits"].shape == (2, 2)
            assert res["graph_rep"].shape == (2, 192)
    print("GraphClassifier Verification PASSED!")
