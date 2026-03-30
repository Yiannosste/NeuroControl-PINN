"""
PID Controller — baseline comparator.

A standard Proportional-Integral-Derivative controller.
Included as a simple classical baseline to highlight the advantages
of model-based MPC approaches for nonlinear systems.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class PIDController:
    """
    Multi-output PID controller.

    Controls a single scalar output (or all outputs independently).
    For multi-state systems, only the first (primary) state is controlled.
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        state_idx: int = 0,
    ):
        """
        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
            dt: Sampling time.
            u_min, u_max: Output saturation limits.
            state_idx: Index of the state to control.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.state_idx = state_idx

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def step(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Compute control action.

        Args:
            x: Current state.
            x_ref: Reference state.

        Returns:
            u: Control action (scalar, wrapped in array).
        """
        error = x_ref[self.state_idx] - x[self.state_idx]
        self._integral += error * self.dt

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / self.dt

        self._prev_error = error
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        if self.u_min is not None:
            u = np.clip(u, self.u_min, self.u_max)
            # Anti-windup
            if u == self.u_min or u == self.u_max:
                self._integral -= error * self.dt

        return np.array([u])

    def run_closed_loop(
        self,
        system,
        x0: np.ndarray,
        n_steps: int,
        noise_std: float = 0.0,
        x_ref_traj: Optional[np.ndarray] = None,
    ) -> dict:
        """Run closed-loop simulation (same interface as PINNMPC)."""
        dt = self.dt
        n = system.state_dim
        m = system.control_dim

        if x_ref_traj is None:
            x_ref_traj = np.zeros((n_steps, n))

        t_log = np.zeros(n_steps + 1)
        x_log = np.zeros((n_steps + 1, n))
        u_log = np.zeros((n_steps, m))

        x = x0.copy()
        x_log[0] = x
        self.reset()

        print(f"  Running PID closed loop ({n_steps} steps)...")

        for k in range(n_steps):
            t_k = k * dt
            x_meas = x + np.random.normal(0, noise_std, n) if noise_std > 0 else x
            u = self.step(x_meas, x_ref_traj[k])

            sim = system.simulate(
                x0=x,
                u_traj=u[np.newaxis, :],
                t_span=(t_k, t_k + dt),
                dt=dt / 10,
                noise_std=0.0,
            )
            x = sim["x_clean"][-1]

            x_log[k + 1] = x
            u_log[k] = u
            t_log[k] = t_k

        t_log[-1] = n_steps * dt
        return {
            "t": t_log,
            "x": x_log,
            "u": u_log,
            "cost": np.zeros(n_steps),
            "solve_time_ms": np.zeros(n_steps),
            "controller": "PID",
        }
