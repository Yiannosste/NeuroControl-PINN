"""
Classical MPC with perfect model knowledge.

Uses the true system equations as the forward model — this is the
upper bound on achievable control performance (oracle / ground-truth MPC).

Comparing PINN-MPC against this baseline quantifies the degradation
due to model approximation errors.
"""

from __future__ import annotations
from typing import Optional
import time
import numpy as np
from scipy.optimize import minimize, Bounds

from .pinn_mpc import MPCConfig


class ClassicalMPC:
    """
    MPC controller with exact knowledge of the system dynamics.
    Used as the gold-standard baseline for performance comparison.
    """

    def __init__(
        self,
        system,
        config: MPCConfig,
        x_ref: Optional[np.ndarray] = None,
    ):
        """
        Args:
            system: True DynamicalSystem instance.
            config: Shared MPCConfig (same horizon, weights, bounds as PINN-MPC).
            x_ref: Reference state.
        """
        self.system = system
        self.config = config
        self.state_dim = system.state_dim
        self.control_dim = system.control_dim

        self.x_ref = np.zeros(self.state_dim) if x_ref is None else np.array(x_ref)

        cfg = config
        Q = cfg.Q if cfg.Q is not None else np.eye(self.state_dim)
        R = cfg.R if cfg.R is not None else 0.01 * np.eye(self.control_dim)
        P = cfg.P if cfg.P is not None else 10.0 * Q

        self.Q = Q
        self.R = R
        self.P = P

        N, m = config.horizon, self.control_dim
        if cfg.u_min is not None and cfg.u_max is not None:
            lb = np.tile(cfg.u_min, N)
            ub = np.tile(cfg.u_max, N)
            self._bounds = Bounds(lb, ub)
        else:
            self._bounds = None

        self._u_prev: Optional[np.ndarray] = None
        self.solve_times: list[float] = []
        self.n_iterations: list[int] = []

    def set_reference(self, x_ref: np.ndarray):
        self.x_ref = np.array(x_ref)

    def _rollout(self, x0: np.ndarray, u_flat: np.ndarray) -> np.ndarray:
        """Simulate the true system over the horizon (Euler steps for speed)."""
        N, m, dt = self.config.horizon, self.control_dim, self.config.dt
        u_seq = u_flat.reshape(N, m)
        traj = np.zeros((N + 1, self.state_dim))
        traj[0] = x0

        for k in range(N):
            x_k = traj[k]
            u_k = u_seq[k]
            # RK4 step
            k1 = self.system.dynamics(0.0, x_k, u_k)
            k2 = self.system.dynamics(0.0, x_k + 0.5 * dt * k1, u_k)
            k3 = self.system.dynamics(0.0, x_k + 0.5 * dt * k2, u_k)
            k4 = self.system.dynamics(0.0, x_k + dt * k3, u_k)
            traj[k + 1] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return traj

    def _cost(self, u_flat: np.ndarray, x0: np.ndarray) -> float:
        N, m = self.config.horizon, self.control_dim
        u_seq = u_flat.reshape(N, m)
        traj = self._rollout(x0, u_flat)

        cost = 0.0
        for k in range(N):
            e = traj[k] - self.x_ref
            cost += e @ self.Q @ e + u_seq[k] @ self.R @ u_seq[k]

        e_T = traj[-1] - self.x_ref
        cost += e_T @ self.P @ e_T
        return cost

    def _grad(self, u_flat: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Finite-difference gradient (fast enough for classical MPC demo)."""
        eps = 1e-4
        grad = np.zeros_like(u_flat)
        f0 = self._cost(u_flat, x0)
        for i in range(len(u_flat)):
            u_plus = u_flat.copy()
            u_plus[i] += eps
            grad[i] = (self._cost(u_plus, x0) - f0) / eps
        return grad

    def solve(self, x0: np.ndarray) -> tuple[np.ndarray, dict]:
        N, m = self.config.horizon, self.control_dim

        if self.config.warm_start and self._u_prev is not None:
            u0 = np.concatenate([self._u_prev[m:], self._u_prev[-m:]])
        else:
            u0 = np.zeros(N * m)

        t_start = time.perf_counter()
        result = minimize(
            fun=self._cost,
            x0=u0,
            args=(x0,),
            jac=self._grad,
            method="SLSQP",
            bounds=self._bounds,
            options={"maxiter": self.config.max_iter, "ftol": self.config.tol, "disp": False},
        )
        t_elapsed = time.perf_counter() - t_start

        self.solve_times.append(t_elapsed)
        self.n_iterations.append(result.nit)
        self._u_prev = result.x.copy()

        u_opt = result.x[:m]
        if self.config.u_min is not None:
            u_opt = np.clip(u_opt, self.config.u_min, self.config.u_max)

        return u_opt, {
            "cost": result.fun,
            "success": result.success,
            "n_iter": result.nit,
            "solve_time_ms": t_elapsed * 1000,
        }

    def run_closed_loop(
        self,
        system,
        x0: np.ndarray,
        n_steps: int,
        noise_std: float = 0.0,
        x_ref_traj: Optional[np.ndarray] = None,
    ) -> dict:
        """Run closed-loop simulation (same interface as PINNMPC)."""
        dt = self.config.dt
        n, m = self.state_dim, self.control_dim

        t_log = np.zeros(n_steps + 1)
        x_log = np.zeros((n_steps + 1, n))
        u_log = np.zeros((n_steps, m))
        cost_log = np.zeros(n_steps)
        stime_log = np.zeros(n_steps)

        x = x0.copy()
        x_log[0] = x

        print(f"  Running Classical MPC closed loop ({n_steps} steps)...")

        for k in range(n_steps):
            t_k = k * dt
            if x_ref_traj is not None:
                self.set_reference(x_ref_traj[k])

            x_meas = x + np.random.normal(0, noise_std, n) if noise_std > 0 else x
            u_opt, info = self.solve(x_meas)

            sim = system.simulate(
                x0=x,
                u_traj=u_opt[np.newaxis, :],
                t_span=(t_k, t_k + dt),
                dt=dt / 10,
                noise_std=0.0,
            )
            x = sim["x_clean"][-1]

            x_log[k + 1] = x
            u_log[k] = u_opt
            cost_log[k] = info["cost"]
            stime_log[k] = info["solve_time_ms"]

        t_log[-1] = n_steps * dt
        print(f"  Done. Avg solve time: {stime_log.mean():.1f} ms")

        return {
            "t": t_log,
            "x": x_log,
            "u": u_log,
            "cost": cost_log,
            "solve_time_ms": stime_log,
            "controller": "Classical MPC",
        }
