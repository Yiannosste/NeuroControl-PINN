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
    Camacho, E. F., & Bordons, C. (2007). Model predictive control.
    Springer Science & Business Media.

    Chen, W. H., et al. (2012). Disturbance-observer-based control and
    related methods—An overview. IEEE TIE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import warnings
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
    rollout_substeps: int = 1  # RK4/Euler substeps per dt — raise for stiff systems
                               # where a single step of size dt is unstable (e.g. CSTR)

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
    grad_eps: float = 1e-5         # finite-difference epsilon (used by ClassicalMPC)


class PINNMPC:
    """
    Model Predictive Controller using a PINN surrogate model.

    The cost function and gradients are computed using PyTorch autograd,
    which enables efficient gradient-based optimisation.

    Device handling: the controller operates entirely on whichever device
    the loaded PINN model resides on.  Cost matrices, reference tensors, and
    all intermediate computations are kept on that device; gradients are only
    transferred to CPU (as float64 NumPy arrays) when SciPy needs them.
    """

    def __init__(
        self,
        model: PINN,
        config: MPCConfig,
        x_ref: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            model: Trained PINN dynamics model.
            config: MPC configuration.
            x_ref: Desired reference state (state_dim,). Defaults to origin.
        """
        self.model = model
        self.model.eval()
        self.config = config
        self.state_dim: int = model.state_dim
        self.control_dim: int = model.control_dim

        # Infer device from model parameters — works for CPU and any CUDA device.
        self._device: torch.device = next(model.parameters()).device

        # Reference state
        self.x_ref: np.ndarray = (
            np.zeros(self.state_dim) if x_ref is None else np.array(x_ref, dtype=np.float64)
        )

        # Cost matrices — constructed once and kept on the target device.
        cfg = config
        Q = cfg.Q if cfg.Q is not None else np.eye(self.state_dim)
        R = cfg.R if cfg.R is not None else 0.01 * np.eye(self.control_dim)
        P = cfg.P if cfg.P is not None else 10.0 * Q

        self.Q: torch.Tensor = torch.as_tensor(Q, dtype=torch.float32, device=self._device)
        self.R: torch.Tensor = torch.as_tensor(R, dtype=torch.float32, device=self._device)
        self.P: torch.Tensor = torch.as_tensor(P, dtype=torch.float32, device=self._device)
        self.x_ref_t: torch.Tensor = torch.as_tensor(
            self.x_ref, dtype=torch.float32, device=self._device
        )

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

        # ── Persistent device buffers (avoids per-call tensor allocation) ──────
        # These are no-grad storage tensors updated in-place on every solver call.
        self._u_buf: torch.Tensor = torch.zeros(N, m, dtype=torch.float32, device=self._device)
        self._x0_buf: torch.Tensor = torch.zeros(
            self.state_dim, dtype=torch.float32, device=self._device
        )

        # Performance counters
        self.solve_times: list[float] = []
        self.n_iterations: list[int] = []

    def set_reference(self, x_ref: np.ndarray) -> None:
        """Update the reference state."""
        self.x_ref = np.array(x_ref, dtype=np.float64)
        self.x_ref_t = torch.as_tensor(
            self.x_ref, dtype=torch.float32, device=self._device
        )

    def _rollout_torch(
        self, x0: torch.Tensor, u_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate state using the PINN over the horizon.

        Args:
            x0: Initial state (state_dim,) on self._device.
            u_seq: Control sequence (N, control_dim) on self._device.

        Returns:
            Trajectory tensor (N+1, state_dim) on self._device.
        """
        N = self.config.horizon
        n_sub = max(1, self.config.rollout_substeps)
        h = self.config.dt / n_sub
        traj = [x0.unsqueeze(0)]  # (1, state_dim)
        x = x0

        for k in range(N):
            u_k = u_seq[k].unsqueeze(0)   # (1, control_dim)

            for _ in range(n_sub):
                x_batch = x.unsqueeze(0)   # (1, state_dim)
                if self.config.integration == "euler":
                    dxdt = self.model(x_batch, u_k).squeeze(0)
                    x = x + h * dxdt
                else:  # rk4
                    k1 = self.model(x_batch, u_k).squeeze(0)
                    k2 = self.model((x + 0.5 * h * k1).unsqueeze(0), u_k).squeeze(0)
                    k3 = self.model((x + 0.5 * h * k2).unsqueeze(0), u_k).squeeze(0)
                    k4 = self.model((x + h * k3).unsqueeze(0), u_k).squeeze(0)
                    x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj.append(x.unsqueeze(0))

        return torch.cat(traj, dim=0)  # (N+1, state_dim)

    def _cost_and_grad(
        self, u_flat: np.ndarray, x0: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute MPC cost and gradient w.r.t. the flattened control sequence.

        Uses pre-allocated device buffers so that no new tensor objects are
        created on the heap per call.  Only the final scalar and gradient
        are transferred back to CPU as NumPy arrays.

        Args:
            u_flat: Flattened control sequence (N * control_dim,), float64 NumPy.
            x0: Current state (state_dim,), float64 NumPy.

        Returns:
            (cost, gradient) pair consumed by scipy.optimize.minimize.
        """
        N, m = self.config.horizon, self.control_dim

        # ── Update persistent device buffers in-place ────────────────────────
        # Avoids per-call Python-object allocation.  casting='unsafe' converts
        # float64 → float32 without a copy when dtypes already match.
        self._u_buf.copy_(
            torch.from_numpy(u_flat.astype(np.float32, copy=False)).view(N, m)
        )
        self._x0_buf.copy_(
            torch.from_numpy(x0.astype(np.float32, copy=False))
        )

        # Create a leaf tensor for autograd (clone breaks the shared-memory
        # link to _u_buf so that backward() can populate .grad cleanly).
        u_t: torch.Tensor = self._u_buf.clone().requires_grad_(True)
        x0_t: torch.Tensor = self._x0_buf  # no grad needed for state

        # Rollout through PINN
        traj: torch.Tensor = self._rollout_torch(x0_t, u_t)  # (N+1, state_dim)

        # Stage costs: Σ e_k' Q e_k + u_k' R u_k
        e: torch.Tensor = traj[:-1] - self.x_ref_t  # (N, state_dim)
        stage: torch.Tensor = (e @ self.Q * e).sum() + (u_t @ self.R * u_t).sum()

        # Terminal cost: e_N' P e_N
        e_T: torch.Tensor = traj[-1] - self.x_ref_t
        terminal: torch.Tensor = (e_T @ self.P * e_T).sum()

        cost: torch.Tensor = stage + terminal
        cost.backward()

        # Transfer gradient to CPU as float64 NumPy — safe for any device.
        grad: np.ndarray = (
            u_t.grad.detach().cpu().numpy().flatten().astype(np.float64)
        )
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

        # Warm start: shift previous solution, repeat last action.
        if self.config.warm_start and self._u_prev is not None:
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

        if not result.success:
            warnings.warn(
                f"PINNMPC solver did not converge: '{result.message}'. "
                "Applying zero-order hold (warm-start / zero fallback).",
                RuntimeWarning,
                stacklevel=2,
            )

        u_opt_flat: np.ndarray = result.x
        self._u_prev = u_opt_flat.copy()

        u_opt: np.ndarray = u_opt_flat[:m]

        # ── Bounds enforcement ───────────────────────────────────────────────
        if self.config.u_min is not None:
            # Detect and warn about significant bound violations from the solver.
            violation = float(
                max(
                    np.max(self.config.u_min - u_opt),
                    np.max(u_opt - self.config.u_max),
                    0.0,
                )
            )
            if violation > 1e-4:
                warnings.warn(
                    f"PINNMPC: control bound violation of {violation:.2e} detected "
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

        info: dict = {
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
            "controller": "PINN-MPC",
        }
