"""
Physics-Informed Loss Functions for PINN training.

The total loss combines:
1. DataLoss        — MSE between predicted dx/dt and measured/estimated derivatives.
2. PhysicsResidualLoss — Residuals of known ODE structure at collocation points.
3. InitialConditionLoss — Penalise deviation from observed initial conditions.

Adaptive loss weighting (NTK-inspired):
    Dynamically re-balances λ_data and λ_phys during training based on
    the gradient magnitudes of each loss component.

Reference:
    Wang, S., Teng, Y., & Perdikaris, P. (2021).
    Understanding and mitigating gradient flow pathologies in
    physics-informed neural networks.
    SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
"""

from __future__ import annotations
from typing import Optional, Callable
import torch
import torch.nn as nn
import numpy as np


class DataLoss(nn.Module):
    """
    Supervised data loss: MSE between PINN prediction and observed dx/dt.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        dxdt_pred: torch.Tensor,
        dxdt_true: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dxdt_pred: (N, state_dim) predicted derivatives.
            dxdt_true: (N, state_dim) ground-truth derivatives.
            weights: (N,) optional per-sample weights.

        Returns:
            Scalar loss.
        """
        residual = (dxdt_pred - dxdt_true) ** 2

        if weights is not None:
            residual = residual * weights.unsqueeze(-1)

        if self.reduction == "mean":
            return residual.mean()
        return residual.sum()


class PhysicsResidualLoss(nn.Module):
    """
    Physics residual loss at collocation points.

    For a known ODE component g_k(x, u) = 0 (possibly a partial constraint),
    this loss penalises ||g_k(x_col, u_col)||^2.

    The physics_fn should accept (model, x_col, u_col) and return residuals.
    """

    def __init__(
        self,
        physics_fn: Callable,
        reduction: str = "mean",
    ):
        """
        Args:
            physics_fn: Callable(model, x, u) → residual tensor (N, state_dim).
        """
        super().__init__()
        self.physics_fn = physics_fn
        self.reduction = reduction

    def forward(
        self,
        model: nn.Module,
        x_col: torch.Tensor,
        u_col: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model: The PINN model.
            x_col: Collocation states (N_col, state_dim).
            u_col: Collocation controls (N_col, control_dim).

        Returns:
            Scalar physics residual loss.
        """
        residuals = self.physics_fn(model, x_col, u_col)  # (N_col, state_dim)
        sq = residuals ** 2

        if self.reduction == "mean":
            return sq.mean()
        return sq.sum()


class InitialConditionLoss(nn.Module):
    """
    Penalise mismatch at observed initial conditions.
    For systems where IC data is available.
    """

    def forward(
        self,
        x0_pred: torch.Tensor,
        x0_true: torch.Tensor,
    ) -> torch.Tensor:
        return ((x0_pred - x0_true) ** 2).mean()


class PINNLoss(nn.Module):
    """
    Composite PINN loss with adaptive weighting.

    L = λ_data * L_data + λ_phys * L_phys

    Supports:
        - Fixed weights
        - NTK-adaptive rebalancing
        - Causal training (weight early time steps more)
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.1,
        physics_fn: Optional[Callable] = None,
        adaptive: bool = True,
        adapt_every: int = 100,
        adapt_alpha: float = 0.9,
    ):
        """
        Args:
            lambda_data: Initial data loss weight.
            lambda_phys: Initial physics loss weight.
            physics_fn: Optional physics residual function.
            adaptive: Use NTK-inspired adaptive weighting.
            adapt_every: Re-compute weights every N steps.
            adapt_alpha: EMA smoothing for weight adaptation.
        """
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.physics_fn = physics_fn
        self.adaptive = adaptive
        self.adapt_every = adapt_every
        self.adapt_alpha = adapt_alpha

        self.data_loss_fn = DataLoss()
        if physics_fn is not None:
            self.phys_loss_fn = PhysicsResidualLoss(physics_fn)
        else:
            self.phys_loss_fn = None

        self._step = 0
        self._lambda_data_ema = lambda_data
        self._lambda_phys_ema = lambda_phys

    def forward(
        self,
        model: nn.Module,
        dxdt_pred: torch.Tensor,
        dxdt_true: torch.Tensor,
        x_col: Optional[torch.Tensor] = None,
        u_col: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute total loss and individual components.

        Args:
            model: PINN model.
            dxdt_pred: Predicted derivatives (N, state_dim).
            dxdt_true: Observed derivatives (N, state_dim).
            x_col: Collocation states (N_col, state_dim).
            u_col: Collocation controls (N_col, control_dim).

        Returns:
            dict with 'total', 'data', 'physics' scalar tensors.
        """
        L_data = self.data_loss_fn(dxdt_pred, dxdt_true)

        losses = {"data": L_data, "physics": torch.tensor(0.0)}

        if self.phys_loss_fn is not None and x_col is not None:
            L_phys = self.phys_loss_fn(model, x_col, u_col)
            losses["physics"] = L_phys
        else:
            L_phys = torch.tensor(0.0, requires_grad=False)

        # Adaptive weighting
        if self.adaptive and self._step % self.adapt_every == 0 and self._step > 0:
            self._adapt_weights(model, L_data, L_phys)

        total = self._lambda_data_ema * L_data + self._lambda_phys_ema * L_phys
        losses["total"] = total
        losses["lambda_data"] = self._lambda_data_ema
        losses["lambda_phys"] = self._lambda_phys_ema

        self._step += 1
        return losses

    def _adapt_weights(
        self,
        model: nn.Module,
        L_data: torch.Tensor,
        L_phys: torch.Tensor,
    ):
        """
        NTK-inspired adaptive rebalancing.
        Sets λ so that each loss has roughly equal gradient magnitude.
        """
        def _grad_norm(loss, model):
            if not loss.requires_grad:
                return 1.0
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, allow_unused=True
            )
            total = sum(
                g.norm().item() ** 2 for g in grads if g is not None
            )
            return max(total ** 0.5, 1e-8)

        norm_data = _grad_norm(L_data, model)
        norm_phys = _grad_norm(L_phys, model) if L_phys.requires_grad else 1e-8

        # Balance so both loss gradients have similar magnitude
        mean_norm = (norm_data + norm_phys) / 2.0
        new_lambda_data = mean_norm / max(norm_data, 1e-8)
        new_lambda_phys = mean_norm / max(norm_phys, 1e-8)

        alpha = self.adapt_alpha
        self._lambda_data_ema = (
            alpha * self._lambda_data_ema + (1 - alpha) * new_lambda_data
        )
        self._lambda_phys_ema = (
            alpha * self._lambda_phys_ema + (1 - alpha) * new_lambda_phys
        )
