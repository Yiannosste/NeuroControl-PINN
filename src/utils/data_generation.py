"""
Dataset generation, splitting, and normalisation utilities.

These tools bridge the raw simulation outputs from DynamicalSystem
and the training pipeline of the PINN.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def generate_dataset(
    system,
    n_trajectories: int = 50,
    t_end: float = 10.0,
    dt: float = 0.05,
    noise_std: float = 0.05,
    control_mode: str = "random",  # 'random', 'chirp', 'prbs'
    seed: int = 42,
) -> dict:
    """
    Generate a training dataset from multiple simulated trajectories.

    Args:
        system: DynamicalSystem instance.
        n_trajectories: Number of independent trajectories.
        t_end: Duration of each trajectory (s).
        dt: Time step.
        noise_std: Measurement noise.
        control_mode: Type of excitation signal.
        seed: Random seed.

    Returns:
        dict with keys 't', 'x', 'u', 'dxdt' (concatenated).
    """
    rng = np.random.RandomState(seed)
    x_min, x_max = system.state_bounds
    u_min, u_max = system.control_bounds
    n_steps = int(t_end / dt)

    all_t, all_x, all_u, all_dxdt = [], [], [], []

    for i in range(n_trajectories):
        x0 = rng.uniform(x_min, x_max)

        if control_mode == "random":
            u_traj = rng.uniform(u_min, u_max, size=(n_steps, system.control_dim))
        elif control_mode == "chirp":
            t_arr = np.linspace(0, t_end, n_steps)
            freq = 0.5 + i * 0.1
            u_traj = np.zeros((n_steps, system.control_dim))
            for j in range(system.control_dim):
                amp = (u_max[j] - u_min[j]) / 2.0
                u_traj[:, j] = amp * np.sin(2 * np.pi * freq * t_arr)
        elif control_mode == "prbs":
            # Pseudo-random binary sequence
            u_traj = np.zeros((n_steps, system.control_dim))
            for j in range(system.control_dim):
                switches = rng.randint(1, 20)
                sign = rng.choice([-1, 1])
                for _ in range(n_steps):
                    pass
                change_points = sorted(rng.choice(n_steps, switches, replace=False))
                amp = (u_max[j] - u_min[j]) / 2.0
                val = amp * sign
                prev_cp = 0
                for cp in change_points:
                    u_traj[prev_cp:cp, j] = val
                    val = -val
                    prev_cp = cp
                u_traj[prev_cp:, j] = val
        else:
            raise ValueError(f"Unknown control_mode: {control_mode}")

        data = system.simulate(
            x0=x0,
            u_traj=u_traj,
            t_span=(0.0, t_end),
            dt=dt,
            noise_std=noise_std,
        )
        all_t.append(data["t"])
        all_x.append(data["x"])
        all_u.append(data["u"])
        all_dxdt.append(data["dxdt"])

    return {
        "t": np.concatenate(all_t),
        "x": np.concatenate(all_x),
        "u": np.concatenate(all_u),
        "dxdt": np.concatenate(all_dxdt),
    }


def split_dataset(
    data: dict,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 0,
) -> tuple[dict, dict, dict]:
    """
    Randomly split a dataset into train / validation / test sets.

    Returns:
        (train_data, val_data, test_data) — each a dict with same keys.
    """
    N = len(data["t"])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    n_test = int(N * test_fraction)
    n_val = int(N * val_fraction)
    n_train = N - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def _subset(d, indices):
        return {k: v[indices] for k, v in d.items()}

    return (
        _subset(data, train_idx),
        _subset(data, val_idx),
        _subset(data, test_idx),
    )


class DataNormalizer:
    """
    Normalise state, control, and derivative arrays to zero mean, unit variance.
    Stores statistics for inverse transformation and PINN output de-normalisation.
    """

    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.u_mean = None
        self.u_std = None
        self.dxdt_mean = None
        self.dxdt_std = None

    def fit(self, data: dict):
        self.x_mean = data["x"].mean(axis=0)
        self.x_std = data["x"].std(axis=0) + 1e-8
        self.u_mean = data["u"].mean(axis=0)
        self.u_std = data["u"].std(axis=0) + 1e-8
        self.dxdt_mean = data["dxdt"].mean(axis=0)
        self.dxdt_std = data["dxdt"].std(axis=0) + 1e-8
        return self

    def transform(self, data: dict) -> dict:
        return {
            "t": data["t"],
            "x": (data["x"] - self.x_mean) / self.x_std,
            "u": (data["u"] - self.u_mean) / self.u_std,
            "dxdt": (data["dxdt"] - self.dxdt_mean) / self.dxdt_std,
        }

    def inverse_transform_x(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.x_std + self.x_mean

    def inverse_transform_dxdt(self, dxdt_norm: np.ndarray) -> np.ndarray:
        return dxdt_norm * self.dxdt_std + self.dxdt_mean

    def fit_transform(self, data: dict) -> dict:
        return self.fit(data).transform(data)


def normalize_dataset(
    train: dict, val: dict, test: dict
) -> tuple[dict, dict, dict, DataNormalizer]:
    """
    Fit normaliser on training data and apply to all splits.

    Returns:
        (train_norm, val_norm, test_norm, normalizer)
    """
    norm = DataNormalizer()
    return (
        norm.fit_transform(train),
        norm.transform(val),
        norm.transform(test),
        norm,
    )
