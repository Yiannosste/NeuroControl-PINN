"""
Performance metrics for PINN surrogate accuracy and closed-loop control quality.

Metrics are grouped into three categories:
    1. Surrogate accuracy:  how well the PINN approximates the true dynamics.
    2. Tracking performance: how well the MPC tracks a reference trajectory.
    3. Control effort:       energy/smoothness of the control signal.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


# ─── Surrogate Accuracy ───────────────────────────────────────────────────────

def rmse(y_pred: np.ndarray, y_true: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root mean squared error."""
    return np.sqrt(((y_pred - y_true) ** 2).mean(axis=axis))


def mae(y_pred: np.ndarray, y_true: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean absolute error."""
    return np.abs(y_pred - y_true).mean(axis=axis)


def relative_l2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Relative L2 error (dimensionless)."""
    return (
        np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-12)
    )


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Coefficient of determination R² per output dimension.

    R² = 1 - SS_res / SS_tot
    Perfect prediction → R² = 1. Predicting mean → R² = 0.
    """
    ss_res = ((y_pred - y_true) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def physics_residual_norm(
    system,
    x: np.ndarray,
    u: np.ndarray,
    dxdt_pred: np.ndarray,
) -> float:
    """
    Mean L2 norm of the physics residuals (ODE constraint violation).

    ||f̂(x, u) - f(x, u)||₂ averaged over all test points.
    """
    dxdt_true = np.array([
        system.dynamics(0.0, x[i], u[i]) for i in range(len(x))
    ])
    return rmse(dxdt_pred, dxdt_true).mean()


# ─── Control Performance ──────────────────────────────────────────────────────

def tracking_error(
    x_traj: np.ndarray,
    x_ref: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute tracking error statistics for a closed-loop trajectory.

    Args:
        x_traj: State trajectory (T, state_dim).
        x_ref: Reference state (state_dim,) or trajectory (T, state_dim).
        weights: Per-state weights for the error norm.

    Returns:
        dict with 'rmse', 'mae', 'max_error', 'ise' (integral squared error).
    """
    if x_ref.ndim == 1:
        x_ref = np.tile(x_ref, (len(x_traj), 1))

    error = x_traj - x_ref
    if weights is not None:
        error = error * weights[np.newaxis, :]

    norms = np.linalg.norm(error, axis=1)  # (T,)

    return {
        "rmse": float(np.sqrt((norms ** 2).mean())),
        "mae": float(norms.mean()),
        "max_error": float(norms.max()),
        "ise": float((norms ** 2).sum()),       # integral of squared error
        "iae": float(norms.sum()),               # integral of absolute error
        "overshoot": float(norms.max() / (norms[0] + 1e-12)),
    }


def control_effort(u_traj: np.ndarray, dt: float = 1.0) -> dict:
    """
    Compute control effort metrics.

    Args:
        u_traj: Control trajectory (T, control_dim).
        dt: Sampling time.

    Returns:
        dict with 'ise_u', 'total_variation', 'peak'.
    """
    ise_u = float((u_traj ** 2).sum() * dt)
    tv = float(np.abs(np.diff(u_traj, axis=0)).sum())
    peak = float(np.abs(u_traj).max())

    return {
        "ise_u": ise_u,
        "total_variation": tv,
        "peak": peak,
    }


def settling_time(
    x_traj: np.ndarray,
    x_ref: np.ndarray,
    t: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """
    Estimate settling time: first time the normalised error stays below threshold.

    Args:
        x_traj: (T, state_dim)
        x_ref: (state_dim,)
        t: (T,) time array
        threshold: Fraction of initial error to define "settled".

    Returns:
        Settling time in seconds, or np.inf if never settled.
    """
    error = np.linalg.norm(x_traj - x_ref, axis=1)
    e0 = error[0] + 1e-12
    norm_error = error / e0

    for i in range(len(norm_error)):
        if np.all(norm_error[i:] < threshold):
            return float(t[i])
    return float("inf")


def compute_metrics(
    result: dict,
    system,
    x_ref: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute a comprehensive set of metrics from a closed-loop result dict.

    Args:
        result: Output of controller.run_closed_loop().
        system: DynamicalSystem instance.
        x_ref: Reference state (defaults to origin).

    Returns:
        Flat dict of scalar metrics.
    """
    if x_ref is None:
        x_ref = np.zeros(system.state_dim)

    dt = result["t"][1] - result["t"][0] if len(result["t"]) > 1 else 1.0

    te = tracking_error(result["x"], x_ref)
    ce = control_effort(result["u"], dt)
    st = settling_time(result["x"], x_ref, result["t"])

    avg_solve_ms = float(result["solve_time_ms"].mean()) if len(result["solve_time_ms"]) > 0 else 0.0

    return {
        "controller": result.get("controller", "unknown"),
        **te,
        **ce,
        "settling_time_s": st,
        "avg_solve_time_ms": avg_solve_ms,
    }
