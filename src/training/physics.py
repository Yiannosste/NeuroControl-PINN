"""
Physics residual functions for known structural constraints, per system.

Each factory returns a function with signature (model, x_col, u_col) -> residual,
matching the interface expected by PhysicsResidualLoss / PINNLoss.

Design principle
-----------------
We only enforce constraints that are *exactly* known regardless of the
uncertain physical parameters the PINN is meant to learn. For every system
here that means the kinematic identity between a position-like state and
its own time derivative (velocity) — e.g. dp/dt = p_dot. This holds by
definition of what a "position" and "velocity" state are, independent of
mass, length, friction, or any other unknown parameter.

Components of the dynamics that *do* depend on the uncertain physical
parameters (the nonlinear restoring/damping term in Van der Pol, the
pendulum's torque balance in Cart-Pole, the Arrhenius reaction rate in the
CSTR) are deliberately left unconstrained — enforcing them would mean
hand-coding the answer the network is supposed to learn from data.

Normalisation
-------------
Training happens in normalised space (zero mean, unit variance), but the
kinematic identities are only exact in *physical* units. Each factory closes
over the fitted DataNormalizer so the residual is computed correctly:
the normalised state is unscaled back to physical units, the identity is
applied, and the result is rescaled into the same normalised space as the
model's own output before the two are compared.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch

from ..utils.data_generation import DataNormalizer


def make_van_der_pol_physics_fn(normalizer: DataNormalizer) -> Callable:
    """
    Exact kinematic identity: dx1/dt = x2 (position derivative is velocity),
    independent of the Van der Pol nonlinearity parameter mu.
    """
    x_std1 = float(normalizer.x_std[1])
    x_mean1 = float(normalizer.x_mean[1])
    d_mean0 = float(normalizer.dxdt_mean[0])
    d_std0 = float(normalizer.dxdt_std[0])

    def physics_fn(model, x_col: torch.Tensor, u_col: torch.Tensor) -> torch.Tensor:
        dxdt_pred = model(x_col, u_col)
        x2_phys = x_col[:, 1:2] * x_std1 + x_mean1
        target_norm = (x2_phys - d_mean0) / d_std0
        return dxdt_pred[:, 0:1] - target_norm

    return physics_fn


def make_cartpole_physics_fn(normalizer: DataNormalizer) -> Callable:
    """
    Exact kinematic identities: dp/dt = p_dot and dtheta/dt = theta_dot,
    independent of cart/pole mass, length, or gravity.
    """
    x_std = normalizer.x_std
    x_mean = normalizer.x_mean
    d_mean = normalizer.dxdt_mean
    d_std = normalizer.dxdt_std

    p_dot_std, p_dot_mean = float(x_std[1]), float(x_mean[1])
    th_dot_std, th_dot_mean = float(x_std[3]), float(x_mean[3])
    dp_mean, dp_std = float(d_mean[0]), float(d_std[0])
    dth_mean, dth_std = float(d_mean[2]), float(d_std[2])

    def physics_fn(model, x_col: torch.Tensor, u_col: torch.Tensor) -> torch.Tensor:
        dxdt_pred = model(x_col, u_col)

        p_dot_phys = x_col[:, 1:2] * p_dot_std + p_dot_mean
        target_p = (p_dot_phys - dp_mean) / dp_std
        res_p = dxdt_pred[:, 0:1] - target_p

        th_dot_phys = x_col[:, 3:4] * th_dot_std + th_dot_mean
        target_th = (th_dot_phys - dth_mean) / dth_std
        res_th = dxdt_pred[:, 2:3] - target_th

        return torch.cat([res_p, res_th], dim=-1)

    return physics_fn


# CSTR has no algebraic/kinematic identity that holds independent of the
# unknown reaction kinetics — both dCA/dt and dT/dt depend on the Arrhenius
# rate term that the PINN is meant to learn from data. No lossless
# structural constraint is available, so CSTR is intentionally left out of
# this registry and trains in pure data-driven mode (lambda_phys is ignored
# when physics_fn is None).
PHYSICS_FN_FACTORY: dict[str, Optional[Callable]] = {
    "van_der_pol": make_van_der_pol_physics_fn,
    "cartpole": make_cartpole_physics_fn,
    "cstr": None,
}
