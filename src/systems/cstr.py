"""
Continuous Stirred Tank Reactor (CSTR) — Exothermic First-Order Reaction.

A → B  (irreversible, exothermic)

Mass and energy balances:
    dCA/dt = (q/V) * (CAf - CA) - k0 * exp(-E/R/T) * CA
    dT/dt  = (q/V) * (Tf - T)
             + (-ΔH) / (rho * Cp) * k0 * exp(-E/R/T) * CA
             + (UA) / (rho * Cp * V) * (Tc - T)

State:   x = [CA, T]   (concentration mol/L, temperature K)
Control: u = [Tc]      (coolant temperature K)

Objective: regulate CA and T at a desired setpoint despite disturbances.

Parameters from:
    Henson & Seborg (1997). Nonlinear Process Control. Prentice Hall.
"""

import numpy as np
from .base import DynamicalSystem


class CSTRSystem(DynamicalSystem):
    """
    Exothermic CSTR with coolant temperature as control input.

    State:   x = [CA (mol/L), T (K)]
    Control: u = [Tc (K)]
    """

    def __init__(self):
        super().__init__(state_dim=2, control_dim=1, name="CSTR")

        # ── Units: mol, L, K, hours ──────────────────────────────────────────
        # All rates are per hour for numerical stability (no overflow/underflow).

        # Kinetic parameters
        self.k0 = 7.2e10        # pre-exponential factor (1/hr)
        self.E_over_R = 8750.0  # activation energy / gas constant (K)

        # Thermodynamic parameters (all in kcal)
        # neg_dH / (rho * Cp) has units of K·L/mol (combined parameter)
        self.neg_dH_over_rho_Cp = 209.2    # K·L/mol   (-ΔH / ρCp)
        # UA / (rho * Cp * V) has units of 1/hr (cooling coefficient)
        self.UA_over_rho_Cp_V = 0.083      # hr^-1·K^-1 — moderate cooling

        # Process parameters
        self.V = 100.0          # reactor volume (L)
        self.q = 100.0          # volumetric flow rate (L/hr)

        # Feed conditions
        self.CAf = 1.0          # feed concentration (mol/L)
        self.Tf = 350.0         # feed temperature (K)

    def reaction_rate(self, CA: float, T: float) -> float:
        return self.k0 * np.exp(-self.E_over_R / T) * CA

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        CA, T = x[0], x[1]
        Tc = float(u[0]) if hasattr(u, "__len__") else float(u)

        r = self.reaction_rate(CA, T)
        q_over_V = self.q / self.V

        dCA = q_over_V * (self.CAf - CA) - r
        dT = (
            q_over_V * (self.Tf - T)
            + self.neg_dH_over_rho_Cp * r
            + self.UA_over_rho_Cp_V * (Tc - T)
        )
        return np.array([dCA, dT])

    @property
    def state_labels(self) -> list[str]:
        return ["C_A (mol/L)", "T (K)"]

    @property
    def control_labels(self) -> list[str]:
        return ["T_c (K)"]

    @property
    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0, 300.0]), np.array([1.0, 450.0])

    @property
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([250.0]), np.array([350.0])

    @property
    def steady_states(self) -> list[dict]:
        """
        Approximate steady state regions for the CSTR at Tc = 300 K.
        Exact values depend on parameters; use operating_point for a verified SS.
        """
        return [
            {"CA": 0.90, "T": 315.0, "label": "Low conversion region"},
            {"CA": 0.50, "T": 350.0, "label": "Mid conversion region"},
            {"CA": 0.10, "T": 395.0, "label": "High conversion region"},
        ]

    @property
    def operating_point(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Desired operating point (high-conversion region).
        Used as the MPC reference setpoint.
        """
        x_op = np.array([0.10, 395.0])
        u_op = np.array([300.0])
        return x_op, u_op

    def generate_training_data(
        self,
        n_trajectories: int = 100,
        t_end: float = 30.0,
        dt: float = 0.1,
        noise_std: float = 0.01,
        seed: int = 7,
    ) -> dict:
        """
        Generate training trajectories around all three operating regions.
        """
        rng = np.random.RandomState(seed)
        u_min, u_max = self.control_bounds

        all_data = {"t": [], "x": [], "u": [], "dxdt": []}

        for i in range(n_trajectories):
            # Sample initial conditions spread across all operating regions
            if i % 3 == 0:
                CA0 = rng.uniform(0.05, 0.30)
                T0 = rng.uniform(385.0, 415.0)
            elif i % 3 == 1:
                CA0 = rng.uniform(0.40, 0.65)
                T0 = rng.uniform(350.0, 380.0)
            else:
                CA0 = rng.uniform(0.70, 0.95)
                T0 = rng.uniform(310.0, 340.0)

            x0 = np.array([CA0, T0])
            n_steps = int(t_end / dt)
            u_traj = rng.uniform(u_min, u_max, size=(n_steps, 1))

            data = self.simulate(
                x0=x0,
                u_traj=u_traj,
                t_span=(0.0, t_end),
                dt=dt,
                noise_std=noise_std,
            )
            for key in all_data:
                all_data[key].append(data[key])

        return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
