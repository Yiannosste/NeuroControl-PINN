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
import warnings
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
    ) -> None:
        """
        Args:
            system: True DynamicalSystem instance.
            config: Shared MPCConfig (same horizon, weights, bounds as PINN-MPC).
            x_ref: Reference state.
        """
        self.system = system
        self.config = config
        self.state_dim: int = system.state_dim
        self.control_dim: int = system.control_dim

        self.x_ref: np.ndarray = (
            np.zeros(self.state_dim) if x_ref is None else np.array(x_ref)
        )

        cfg = config
        Q = cfg.Q if cfg.Q is not None else np.eye(self.state_dim)
        R = cfg.R if cfg.R is not None else 0.01 * np.eye(self.control_dim)
        P = cfg.P if cfg.P is not None else 10.0 * Q

        self.Q = Q
        self.R = R
        self.P = P

        N = config.horizon
        if cfg.u_min is not None and cfg.u_max is not None:
            lb = np.tile(cfg.u_min, N)
            ub = np.tile(cfg.u_max, N)
            self._bounds = Bounds(lb, ub)
        else:
            self._bounds = None

        self._u_prev: Optional[np.ndarray] = None
        self.solve_times: list[float] = []
        self.n_iterations: list[int] = []

    def set_reference(self, x_ref: np.ndarray) -> None:
        self.x_ref = np.array(x_ref)

    def _rollout(self, x0: np.ndarray, u_flat: np.ndarray) -> np.ndarray:
        """Simulate the true system over the horizon using RK4."""
        N, m, dt = self.config.horizon, self.control_dim, self.config.dt
        u_seq = u_flat.reshape(N, m)
        traj = np.zeros((N + 1, self.state_dim))
        traj[0] = x0

        for k in range(N):
            x_k = traj[k]
            u_k = u_seq[k]
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
            cost += float(e @ self.Q @ e + u_seq[k] @ self.R @ u_seq[k])

        e_T = traj[-1] - self.x_ref
        cost += float(e_T @ self.P @ e_T)
        return cost

    def _grad(self, u_flat: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Central-difference gradient of the MPC cost w.r.t. the control sequence.

        Central differences are O(eps²) accurate vs. O(eps) for forward
        differences, which matters significantly for the highly nonlinear
        rollouts seen in systems like Van der Pol.  The epsilon is sourced
        from MPCConfig.grad_eps to keep it configuration-managed.
        """
        eps = self.config.grad_eps
        grad = np.zeros_like(u_flat)
        for i in range(len(u_flat)):
            u_plus = u_flat.copy()
            u_plus[i] += eps
            u_minus = u_flat.copy()
            u_minus[i] -= eps
            grad[i] = (self._cost(u_plus, x0) - self._cost(u_minus, x0)) / (2.0 * eps)
        return grad

    def solve(self, x0: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Solve the MPC optimisation from state x0.

        On solver failure the method issues a RuntimeWarning and falls back
        to a zero-order hold (previous solution shifted by one step).  This
        prevents silent failure where the controller outputs zero effort and
        appears to "teleport" the state to the reference line.

        Returns:
            u_opt: Optimal first control action (control_dim,).
            info:  Solver diagnostics dict.
        """
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

        if not result.success:
            warnings.warn(
                f"ClassicalMPC solver did not converge: '{result.message}'. "
                "Applying zero-order hold (previous control action / zero fallback).",
                RuntimeWarning,
                stacklevel=2,
            )
            # Zero-order hold: shift previous solution or use zeros.
            if self._u_prev is not None:
                u_opt = self._u_prev[:m].copy()
            else:
                u_opt = np.zeros(m)

            # Still record the (failed) candidate so warm-start can recover.
            self._u_prev = result.x.copy()

            if self.config.u_min is not None:
                u_opt = np.clip(u_opt, self.config.u_min, self.config.u_max)

            return u_opt, {
                "cost": result.fun,
                "success": False,
                "n_iter": result.nit,
                "solve_time_ms": t_elapsed * 1000,
            }

        self._u_prev = result.x.copy()
        u_opt = result.x[:m].copy()

        # ── Bounds enforcement ───────────────────────────────────────────────
        if self.config.u_min is not None:
            violation = float(
                max(
                    np.max(self.config.u_min - u_opt),
                    np.max(u_opt - self.config.u_max),
                    0.0,
                )
            )
            if violation > 1e-4:
                warnings.warn(
                    f"ClassicalMPC: control bound violation of {violation:.2e} detected "
                    "after optimisation (expected < 1e-4 for SLSQP with Bounds). "
                    "Clipping to feasible range.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            u_opt = np.clip(u_opt, self.config.u_min, self.config.u_max)

            # Hard assertion: post-clip u_opt must be strictly feasible.
            assert np.all(u_opt >= self.config.u_min) and np.all(
                u_opt <= self.config.u_max
            ), (
                f"Control bounds assertion failed after clipping: "
                f"u_opt={u_opt}, u_min={self.config.u_min}, u_max={self.config.u_max}"
            )

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
            t_log[k] = t_k
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
