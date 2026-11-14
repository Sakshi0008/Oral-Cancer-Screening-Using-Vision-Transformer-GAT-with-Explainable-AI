"""
Training-Only Feature Scaler
============================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Provides a deterministic standard scaler (Z-score normalization: (x - mu) / sigma)
strictly fitted on the training split to prevent any data leakage.
Saves and loads mean, std, and metadata.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch


class TrainingFeatureScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Strictly fitted on training data only.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_dim_: Optional[int] = None
        self.n_samples_seen_: int = 0
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, torch.Tensor], split_name: str = "train") -> "TrainingFeatureScaler":
        """
        Fits the scaler on training data.
        Enforces documentation that this must be training data.
        """
        if split_name.lower() != "train":
            raise ValueError(
                f"Data leakage protection: FeatureScaler must ONLY be fitted on 'train' split! "
                f"Attempted to fit on '{split_name}'."
            )

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {X_np.shape}")

        self.n_samples_seen_ = X_np.shape[0]
        self.feature_dim_ = X_np.shape[1]
        self.mean_ = np.mean(X_np, axis=0, dtype=np.float64)
        self.std_ = np.std(X_np, axis=0, dtype=np.float64)

        # Replace zero / near-zero standard deviation with 1.0 to prevent division by zero
        self.std_[self.std_ < self.eps] = 1.0
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transforms input features using the fitted training statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before fitting. Scaler must be fitted on training data first.")

        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {X_np.shape}")

        if X_np.shape[1] != self.feature_dim_:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim_}, got {X_np.shape[1]}")

        scaled = (X_np.astype(np.float64) - self.mean_) / self.std_
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if is_torch:
            return torch.from_numpy(scaled.astype(np.float32))
        return scaled.astype(np.float32)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], split_name: str = "train") -> np.ndarray:
        self.fit(X, split_name=split_name)
        return self.transform(X)

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves scaler parameters and metadata to a .pt checkpoint.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "mean": torch.from_numpy(self.mean_.astype(np.float32)),
            "std": torch.from_numpy(self.std_.astype(np.float32)),
            "feature_dim": self.feature_dim_,
            "n_samples_seen": self.n_samples_seen_,
            "metadata": metadata or {},
        }
        torch.save(payload, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainingFeatureScaler":
        """
        Loads scaler parameters from a saved checkpoint.
        """
        payload = torch.load(str(filepath), map_location="cpu", weights_only=False)
        scaler = cls()
        scaler.mean_ = payload["mean"].numpy().astype(np.float64)
        scaler.std_ = payload["std"].numpy().astype(np.float64)
        scaler.feature_dim_ = int(payload["feature_dim"])
        scaler.n_samples_seen_ = int(payload["n_samples_seen"])
        scaler.is_fitted = True
        return scaler
