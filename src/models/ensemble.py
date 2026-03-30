"""
Ensemble of PINNs for uncertainty quantification.

Training multiple PINNs with different random seeds gives a
distribution over predictions. The variance across ensemble members
is used as a proxy for epistemic (model) uncertainty.

This is particularly useful in MPC to:
    1. Detect out-of-distribution states (large uncertainty → distrust prediction).
    2. Implement robust / risk-aware MPC by penalizing high-uncertainty regions.

Reference:
    Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
    Simple and scalable predictive uncertainty estimation using deep ensembles.
    NeurIPS.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import numpy as np

from .pinn import PINN, PINNConfig


class EnsemblePINN(nn.Module):
    """
    Deep Ensemble of PINNs.

    Provides mean prediction and uncertainty estimates.
    """

    def __init__(self, config: PINNConfig, n_members: int = 5):
        super().__init__()
        self.config = config
        self.n_members = n_members
        self.members = nn.ModuleList([PINN(config) for _ in range(n_members)])

    def forward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and standard deviation of dx/dt.

        Returns:
            mean_pred: (..., state_dim)
            std_pred:  (..., state_dim)
        """
        preds = torch.stack(
            [m(x, u) for m in self.members], dim=0
        )  # (n_members, ..., state_dim)

        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        return mean_pred, std_pred

    def predict_numpy(
        self, x: np.ndarray, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std from numpy arrays."""
        squeezed = x.ndim == 1
        if squeezed:
            x = x[np.newaxis, :]
            u = u[np.newaxis, :]

        x_t = torch.FloatTensor(x)
        u_t = torch.FloatTensor(u)

        with torch.no_grad():
            mean, std = self.forward(x_t, u_t)

        mean = mean.numpy()
        std = std.numpy()

        if squeezed:
            return mean[0], std[0]
        return mean, std

    def get_member(self, idx: int) -> PINN:
        return self.members[idx]

    def save(self, path: str):
        torch.save(
            {
                "state_dicts": [m.state_dict() for m in self.members],
                "config": self.config,
                "n_members": self.n_members,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "EnsemblePINN":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        obj = cls(ckpt["config"], ckpt["n_members"])
        for m, sd in zip(obj.members, ckpt["state_dicts"]):
            m.load_state_dict(sd)
        obj.eval()
        return obj

    def count_parameters(self) -> int:
        return sum(m.count_parameters() for m in self.members)

    def __repr__(self):
        return (
            f"EnsemblePINN(members={self.n_members}, "
            f"per_member_params={self.members[0].count_parameters():,})"
        )
