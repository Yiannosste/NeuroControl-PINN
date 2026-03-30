"""
Visualisation utilities for NeuroControl-PINN.

Provides publication-quality plots for:
    - PINN training convergence
    - Phase portraits of dynamical systems
    - Closed-loop state/control trajectories
    - Benchmark comparisons across controllers
    - PINN prediction accuracy scatter plots
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─── Style ────────────────────────────────────────────────────────────────────
STYLE = {
    "PINN-MPC":      {"color": "#2196F3", "ls": "-",  "lw": 2.0},
    "Classical MPC": {"color": "#4CAF50", "ls": "--", "lw": 2.0},
    "PID":           {"color": "#FF5722", "ls": ":",  "lw": 1.8},
    "reference":     {"color": "#9E9E9E", "ls": "-.", "lw": 1.4},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})


def _save_or_show(fig: plt.Figure, path: Optional[str]):
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved figure → {path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Training History ─────────────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    title: str = "PINN Training History",
    save_path: Optional[str] = None,
):
    """Plot Adam + L-BFGS training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.array(history["epoch"])
    phases = np.array(history["phase"])
    total = np.array(history["total"])
    data_loss = np.array(history["data"])
    phys_loss = np.array(history["physics"])

    adam_mask = phases == "adam"
    lbfgs_mask = phases == "lbfgs"

    # Panel 1 — Total loss
    ax = axes[0]
    if adam_mask.any():
        ax.semilogy(epochs[adam_mask], total[adam_mask],
                    color="#2196F3", label="Adam — Total")
        ax.semilogy(epochs[adam_mask], data_loss[adam_mask],
                    color="#2196F3", ls="--", alpha=0.6, label="Adam — Data")
        ax.semilogy(epochs[adam_mask], phys_loss[adam_mask] + 1e-16,
                    color="#2196F3", ls=":", alpha=0.6, label="Adam — Physics")
    if lbfgs_mask.any():
        offset = epochs[adam_mask].max() if adam_mask.any() else 0
        ax.semilogy(epochs[lbfgs_mask] + offset, total[lbfgs_mask],
                    color="#FF5722", label="L-BFGS — Total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2 — Validation
    ax = axes[1]
    val = [v for v in history["val_total"] if v is not None]
    if val:
        val_epochs = [history["epoch"][i] for i, v in enumerate(history["val_total"]) if v is not None]
        ax.semilogy(val_epochs, val, color="#4CAF50", label="Validation Loss")
        ax.semilogy(epochs[adam_mask], total[adam_mask],
                    color="#2196F3", alpha=0.5, label="Train Loss (Adam)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Validation")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No validation data", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    fig.suptitle(title, fontweight="bold")
    _save_or_show(fig, save_path)


# ─── Phase Portrait ───────────────────────────────────────────────────────────

def plot_phase_portrait(
    system,
    trajectories: Optional[list[dict]] = None,
    x_bounds: Optional[tuple] = None,
    resolution: int = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Phase portrait for 2D systems (uses first two state dimensions).
    Overlays vector field and optional simulated trajectories.
    """
    assert system.state_dim >= 2, "Phase portrait requires at least 2 state dims."

    if x_bounds is None:
        x_min, x_max = system.state_bounds
    else:
        x_min, x_max = x_bounds
    x_min, x_max = x_min[:2], x_max[:2]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Vector field
    x1 = np.linspace(x_min[0], x_max[0], resolution)
    x2 = np.linspace(x_min[1], x_max[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)

    U = np.zeros_like(X1)
    V = np.zeros_like(X2)

    for i in range(resolution):
        for j in range(resolution):
            x_ij = np.zeros(system.state_dim)
            x_ij[0], x_ij[1] = X1[i, j], X2[i, j]
            dxdt = system.dynamics(0.0, x_ij, np.zeros(system.control_dim))
            U[i, j] = dxdt[0]
            V[i, j] = dxdt[1]

    speed = np.sqrt(U ** 2 + V ** 2) + 1e-10
    ax.streamplot(
        X1, X2, U, V,
        color=speed / speed.max(),
        cmap="Blues",
        linewidth=0.8,
        arrowsize=0.8,
        density=1.0,
    )

    # Trajectories
    if trajectories:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        for traj, col in zip(trajectories, colors):
            x = traj["x"]
            ax.plot(x[:, 0], x[:, 1], color=col, lw=1.5, alpha=0.8)
            ax.plot(x[0, 0], x[0, 1], "o", color=col, ms=6)
            ax.plot(x[-1, 0], x[-1, 1], "s", color=col, ms=6)

    # Equilibrium
    eq = np.zeros(system.state_dim)
    ax.plot(eq[0], eq[1], "k*", ms=12, label="Equilibrium", zorder=10)

    ax.set_xlim(x_min[0], x_max[0])
    ax.set_ylim(x_min[1], x_max[1])
    ax.set_xlabel(system.state_labels[0])
    ax.set_ylabel(system.state_labels[1])
    ax.set_title(title or f"Phase Portrait — {system.name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    _save_or_show(fig, save_path)


# ─── Closed-Loop Results ──────────────────────────────────────────────────────

def plot_closed_loop(
    result: dict,
    system,
    x_ref: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot state trajectories and control inputs from a closed-loop simulation.
    """
    n = system.state_dim
    m = system.control_dim
    t = result["t"]
    x = result["x"]
    u = result["u"]
    controller = result.get("controller", "Controller")

    n_plots = n + m
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.2 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    style = STYLE.get(controller, {"color": "#333333", "ls": "-", "lw": 2.0})

    for i in range(n):
        ax = axes[i]
        ax.plot(t, x[:, i], label=controller, **style)
        if x_ref is not None:
            ax.axhline(x_ref[i], color=STYLE["reference"]["color"],
                       ls=STYLE["reference"]["ls"], lw=STYLE["reference"]["lw"],
                       label="Reference")
        ax.set_ylabel(system.state_labels[i])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(m):
        ax = axes[n + j]
        ax.step(t[:-1], u[:, j], where="post", label=system.control_labels[j], **style)
        if system.control_bounds:
            u_min, u_max = system.control_bounds
            ax.axhline(u_min[j], color="red", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(u_max[j], color="red", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(system.control_labels[j])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title or f"Closed-Loop — {controller} — {system.name}", fontweight="bold")
    _save_or_show(fig, save_path)


# ─── Benchmark Comparison ────────────────────────────────────────────────────

def plot_benchmark_comparison(
    results: list[dict],
    system,
    x_ref: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Overlay multiple controller trajectories for side-by-side comparison.
    """
    n = system.state_dim
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.5 * (n + 1)), sharex=True)

    legend_handles = []

    for result in results:
        t = result["t"]
        x = result["x"]
        u = result["u"]
        controller = result.get("controller", "?")
        style = STYLE.get(controller, {"color": "#999", "ls": "-", "lw": 1.5})

        for i in range(n):
            axes[i].plot(t, x[:, i], label=controller, **style)
        axes[n].step(t[:-1], u[:, 0], where="post", label=controller, **style)

        legend_handles.append(
            Line2D([0], [0], label=controller, **style)
        )

    if x_ref is not None:
        for i in range(n):
            axes[i].axhline(
                x_ref[i],
                color=STYLE["reference"]["color"],
                ls=STYLE["reference"]["ls"],
                lw=STYLE["reference"]["lw"],
                label="Reference",
            )
        legend_handles.append(
            Line2D(
                [0], [0],
                label="Reference",
                color=STYLE["reference"]["color"],
                ls=STYLE["reference"]["ls"],
            )
        )

    for i, ax in enumerate(axes[:-1]):
        ax.set_ylabel(system.state_labels[i])
        ax.grid(True, alpha=0.3)

    axes[n].set_ylabel(system.control_labels[0])
    axes[n].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")

    axes[0].legend(handles=legend_handles, loc="upper right")
    fig.suptitle(
        title or f"Controller Comparison — {system.name}",
        fontweight="bold",
    )
    _save_or_show(fig, save_path)


# ─── PINN Accuracy ────────────────────────────────────────────────────────────

def plot_pinn_predictions(
    model,
    x_test: np.ndarray,
    u_test: np.ndarray,
    dxdt_test: np.ndarray,
    system,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of PINN predictions vs. ground-truth derivatives.
    One subplot per state dimension.
    """
    import torch

    x_t = torch.FloatTensor(x_test)
    u_t = torch.FloatTensor(u_test)
    with torch.no_grad():
        dxdt_pred = model(x_t, u_t).numpy()

    n = system.state_dim
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        y_true = dxdt_test[:, i]
        y_pred = dxdt_pred[:, i]

        # Scatter
        ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#2196F3", rasterized=True)

        # Perfect-fit line
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

        r2 = 1 - ((y_pred - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2 + 1e-12).sum()
        rmse_val = np.sqrt(((y_pred - y_true) ** 2).mean())

        ax.set_xlabel(f"True d{system.state_labels[i]}/dt")
        ax.set_ylabel(f"Predicted")
        ax.set_title(f"d{system.state_labels[i]}/dt\nR²={r2:.4f}, RMSE={rmse_val:.4e}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PINN Prediction Accuracy — {system.name}", fontweight="bold")
    _save_or_show(fig, save_path)
