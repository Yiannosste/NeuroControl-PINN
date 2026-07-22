"""Tests for physics residual wiring (src/training/physics.py)."""

import numpy as np
import torch

from src.models import PINN, PINNConfig
from src.training import PINNLoss
from src.training.physics import (
    PHYSICS_FN_FACTORY,
    make_van_der_pol_physics_fn,
    make_cartpole_physics_fn,
)
from src.utils.data_generation import DataNormalizer


def _identity_normalizer(dim: int) -> DataNormalizer:
    """A normalizer with mean=0, std=1 so normalised == physical units."""
    norm = DataNormalizer()
    norm.x_mean = np.zeros(dim)
    norm.x_std = np.ones(dim)
    norm.dxdt_mean = np.zeros(dim)
    norm.dxdt_std = np.ones(dim)
    return norm


class TestPhysicsRegistry:
    def test_registry_keys(self):
        assert set(PHYSICS_FN_FACTORY) == {"van_der_pol", "cartpole", "cstr"}

    def test_cstr_has_no_constraint(self):
        assert PHYSICS_FN_FACTORY["cstr"] is None


class TestVanDerPolPhysicsFn:
    def test_zero_residual_when_identity_holds(self):
        """If the model exactly predicts dx1/dt = x2, the residual must vanish
        (checked under an identity normalizer so physical == normalised units)."""
        norm = _identity_normalizer(2)
        physics_fn = make_van_der_pol_physics_fn(norm)

        class ExactModel(torch.nn.Module):
            def forward(self, x, u):
                dxdt = torch.zeros_like(x)
                dxdt[:, 0] = x[:, 1]  # exact kinematic identity
                dxdt[:, 1] = 0.0
                return dxdt

        x_col = torch.randn(16, 2)
        u_col = torch.randn(16, 1)
        residual = physics_fn(ExactModel(), x_col, u_col)
        assert residual.shape == (16, 1)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)

    def test_nonzero_residual_when_identity_violated(self):
        norm = _identity_normalizer(2)
        physics_fn = make_van_der_pol_physics_fn(norm)

        class WrongModel(torch.nn.Module):
            def forward(self, x, u):
                return torch.ones(x.shape[0], 2)

        x_col = torch.zeros(4, 2)  # x2 = 0, so dx1/dt should be 0, model says 1
        u_col = torch.zeros(4, 1)
        residual = physics_fn(WrongModel(), x_col, u_col)
        assert torch.allclose(residual, torch.ones_like(residual), atol=1e-6)

    def test_accounts_for_normalisation(self):
        """A non-trivial normalizer must be correctly unscaled/rescaled."""
        norm = DataNormalizer()
        norm.x_mean = np.array([1.0, 2.0])
        norm.x_std = np.array([3.0, 4.0])
        norm.dxdt_mean = np.array([0.5, 0.0])
        norm.dxdt_std = np.array([2.0, 1.0])
        physics_fn = make_van_der_pol_physics_fn(norm)

        x_col = torch.tensor([[0.0, 0.0]])  # x2_norm=0 -> x2_phys = 2.0
        u_col = torch.zeros(1, 1)
        # target_norm = (2.0 - 0.5) / 2.0 = 0.75
        expected_target = 0.75

        class ConstModel(torch.nn.Module):
            def forward(self, x, u):
                return torch.tensor([[expected_target, 0.0]])

        residual = physics_fn(ConstModel(), x_col, u_col)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)

    def test_integrates_with_pinn_loss(self):
        """physics_fn should plug into PINNLoss and produce a nonzero,
        differentiable physics loss term."""
        norm = _identity_normalizer(2)
        physics_fn = make_van_der_pol_physics_fn(norm)
        model = PINN(PINNConfig(state_dim=2, control_dim=1, hidden_dims=[16, 16]))
        loss_fn = PINNLoss(lambda_data=1.0, lambda_phys=0.1, physics_fn=physics_fn, adaptive=False)

        x = torch.randn(8, 2)
        u = torch.randn(8, 1)
        dxdt_true = torch.randn(8, 2)
        x_col = torch.randn(32, 2)
        u_col = torch.randn(32, 1)

        dxdt_pred = model(x, u)
        losses = loss_fn(model, dxdt_pred, dxdt_true, x_col, u_col)
        assert losses["physics"].item() != 0.0
        losses["total"].backward()
        assert any(p.grad is not None for p in model.parameters())


class TestCartPolePhysicsFn:
    def test_shape_and_zero_residual(self):
        norm = _identity_normalizer(4)
        physics_fn = make_cartpole_physics_fn(norm)

        class ExactModel(torch.nn.Module):
            def forward(self, x, u):
                dxdt = torch.zeros_like(x)
                dxdt[:, 0] = x[:, 1]  # dp/dt = p_dot
                dxdt[:, 2] = x[:, 3]  # dtheta/dt = theta_dot
                return dxdt

        x_col = torch.randn(10, 4)
        u_col = torch.randn(10, 1)
        residual = physics_fn(ExactModel(), x_col, u_col)
        assert residual.shape == (10, 2)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
