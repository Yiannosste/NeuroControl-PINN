"""
PINN-based Model Predictive Control (PINN-MPC).

Uses a trained PINN as the surrogate forward model inside an MPC
optimisation loop. At each timestep:

    1. Measure current state x_k.
    2. Solve the finite-horizon optimal control problem:

        min_{u_0,...,u_{N-1}}  Σ_{t=0}^{N-1} [x_t' Q x_t + u_t' R u_t]
                                + x_N' P x_N

        s.t.  x_{t+1} = x_t + dt * PINN(x_t, u_t)    (PINN dynamics)
              u_min ≤ u_t ≤ u_max                       (actuator limits)
              x_min ≤ x_t ≤ x_max                       (state constraints)

    3. Apply the first control action u_0^* (receding horizon).
    4. Advance time: k ← k + 1, repeat.

The optimisation uses scipy.optimize.minimize with the SLSQP method,
exploiting gradients computed through PyTorch autograd.

References:
    Camacho, E. F., & Alba, C. B. (2013). Model predictive control.
    Springer Science & Business Media.

    Chen, W. H., et al. (2012). Disturbance-observer-based control and
    related methods—An overview. IEEE TIE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import torch
from scipy.optimize import minimize, Bounds, LinearConstraint

from ..models.pinn import PINN


@dataclass
class MPCConfig:
    """Configuration for the MPC controller."""
    horizon: int = 15          # prediction horizon N
    dt: float = 0.05           # time step (s)
    integration: str = "rk4"  # 'euler' or 'rk4'

    # Stage cost weights
    Q: Optional[np.ndarray] = None  # state error weight (state_dim, state_dim)
    R: Optional[np.ndarray] = None  # control effort weight (control_dim, control_dim)
    P: Optional[np.ndarray] = None  # terminal state weight (state_dim, state_dim)

    # Constraints
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None

    # Solver
    max_iter: int = 200
    tol: float = 1e-6
    warm_start: bool = True        # warm-start from previous solution


class PINNMPC:
    """
    Model Predictive Controller using a PINN surrogate model.

    The cost function and gradients are computed using PyTorch autograd,
    which enables efficient gradient-based optimisation.
    """

    def __init__(
        self,
        model: PINN,
        config: MPCConfig,
        x_ref: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model: Trained PINN dynamics model.
            config: MPC configuration.
            x_ref: Desired reference state (state_dim,). Defaults to origin.
        """
        self.model = model
        self.model.eval()
        self.config = config
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim

        # Reference state
        if x_ref is None:
            self.x_ref = np.zeros(self.state_dim)
        else:
            self.x_ref = np.array(x_ref)

        # Cost matrices
        cfg = config
        Q = cfg.Q if cfg.Q is not None else np.eye(self.state_dim)
        R = cfg.R if cfg.R is not None else 0.01 * np.eye(self.control_dim)
        P = cfg.P if cfg.P is not None else 10.0 * Q

        self.Q = torch.FloatTensor(Q)
        self.R = torch.FloatTensor(R)
        self.P = torch.FloatTensor(P)
        self.x_ref_t = torch.FloatTensor(self.x_ref)

        # Decision variable bounds: u_seq flattened (N * control_dim,)
        N, m = config.horizon, self.control_dim
        if cfg.u_min is not None and cfg.u_max is not None:
            lb = np.tile(cfg.u_min, N)
            ub = np.tile(cfg.u_max, N)
            self._bounds = Bounds(lb, ub)
        else:
            self._bounds = None

        # Warm-start initial guess
        self._u_prev: Optional[np.ndarray] = None

        # Performance counters
        self.solve_times: list[float] = []
        self.n_iterations: list[int] = []

    def set_reference(self, x_ref: np.ndarray):
        """Update the reference state."""
        self.x_ref = np.array(x_ref)
        self.x_ref_t = torch.FloatTensor(self.x_ref)

    def _rollout_torch(
        self, x0: torch.Tensor, u_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate state using the PINN over the horizon.

        Args:
            x0: Initial state (state_dim,).
            u_seq: Control sequence (N, control_dim).

        Returns:
            Trajectory (N+1, state_dim).
        """
        N = self.config.horizon
        dt = self.config.dt
        traj = [x0.unsqueeze(0)]  # (1, state_dim)
        x = x0

        for k in range(N):
            u_k = u_seq[k].unsqueeze(0)  # (1, control_dim)
            x_batch = x.unsqueeze(0)     # (1, state_dim)

            if self.config.integration == "euler":
                dxdt = self.model(x_batch, u_k).squeeze(0)
                x = x + dt * dxdt
            else:  # rk4
                k1 = self.model(x_batch, u_k).squeeze(0)
                k2 = self.model((x + 0.5 * dt * k1).unsqueeze(0), u_k).squeeze(0)
                k3 = self.model((x + 0.5 * dt * k2).unsqueeze(0), u_k).squeeze(0)
                k4 = self.model((x + dt * k3).unsqueeze(0), u_k).squeeze(0)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj.append(x.unsqueeze(0))

        return torch.cat(traj, dim=0)  # (N+1, state_dim)

    def _cost_and_grad(
        self, u_flat: np.ndarray, x0: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute MPC cost and gradient wrt the flattened control sequence.

        Args:
            u_flat: Flattened control sequence (N * control_dim,).
            x0: Current state (state_dim,).

        Returns:
            (cost, gradient) pair for scipy.optimize.minimize.
        """
        N, m = self.config.horizon, self.control_dim

        u_t = torch.FloatTensor(u_flat.astype(np.float32).reshape(N, m)).requires_grad_(True)
        x0_t = torch.FloatTensor(x0.astype(np.float32))

        # Rollout
        traj = self._rollout_torch(x0_t, u_t)  # (N+1, state_dim)

        # Stage costs
        e = traj[:-1] - self.x_ref_t  # (N, state_dim) error
        stage = (e @ self.Q * e).sum() + (u_t @ self.R * u_t).sum()

        # Terminal cost
        e_T = traj[-1] - self.x_ref_t
        terminal = (e_T @ self.P * e_T).sum()

        cost = stage + terminal
        cost.backward()

        grad = u_t.grad.detach().numpy().flatten().astype(np.float64)
        return float(cost.item()), grad

    def solve(self, x0: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Solve the MPC optimisation problem from state x0.

        Args:
            x0: Current measured state (state_dim,).

        Returns:
            u_opt: Optimal first control action (control_dim,).
            info: Solver diagnostics dict.
        """
        N, m = self.config.horizon, self.control_dim

        # Warm start
        if self.config.warm_start and self._u_prev is not None:
            # Shift previous solution and repeat last action
            u0 = np.concatenate([
                self._u_prev[m:],
                self._u_prev[-m:]
            ]).astype(np.float64)
        else:
            u0 = np.zeros(N * m, dtype=np.float64)

        t_start = time.perf_counter()

        result = minimize(
            fun=self._cost_and_grad,
            x0=u0,
            args=(x0,),
            method="SLSQP",
            jac=True,
            bounds=self._bounds,
            options={
                "maxiter": self.config.max_iter,
                "ftol": self.config.tol,
                "disp": False,
            },
        )

        t_elapsed = time.perf_counter() - t_start
        self.solve_times.append(t_elapsed)
        self.n_iterations.append(result.nit)

        u_opt_flat = result.x
        self._u_prev = u_opt_flat.copy()

        u_opt = u_opt_flat[:m]

        # Clip to bounds
        if self.config.u_min is not None:
            u_opt = np.clip(u_opt, self.config.u_min, self.config.u_max)

        info = {
            "cost": result.fun,
            "success": result.success,
            "n_iter": result.nit,
            "solve_time_ms": t_elapsed * 1000,
            "u_sequence": u_opt_flat.reshape(N, m),
        }
        return u_opt, info

    def run_closed_loop(
        self,
        system,
        x0: np.ndarray,
        n_steps: int,
        noise_std: float = 0.0,
        x_ref_traj: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run a full closed-loop MPC simulation.

        Args:
            system: A DynamicalSystem instance (used as the "true" plant).
            x0: Initial state.
            n_steps: Number of control steps.
            noise_std: Measurement noise std dev.
            x_ref_traj: Optional time-varying reference (n_steps, state_dim).

        Returns:
            dict with 't', 'x', 'u', 'cost', 'solve_time'.
        """
        dt = self.config.dt
        n = self.state_dim
        m = self.control_dim

        t_log = np.zeros(n_steps + 1)
        x_log = np.zeros((n_steps + 1, n))
        u_log = np.zeros((n_steps, m))
        cost_log = np.zeros(n_steps)
        stime_log = np.zeros(n_steps)

        x = x0.copy()
        x_log[0] = x

        print(f"  Running PINN-MPC closed loop ({n_steps} steps)...")

        for k in range(n_steps):
            t_k = k * dt
            t_log[k] = t_k

            # Update reference if time-varying
            if x_ref_traj is not None:
                self.set_reference(x_ref_traj[k])

            # Add measurement noise
            x_meas = x + np.random.normal(0, noise_std, n) if noise_std > 0 else x

            # Solve MPC
            u_opt, info = self.solve(x_meas)

            # Apply control to real system
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
            "controller": "PINN-MPC",
        }
