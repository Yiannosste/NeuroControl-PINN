"""Abstract base class for all dynamical systems."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem(ABC):
    """
    Base class for continuous-time dynamical systems of the form:
        dx/dt = f(t, x, u)
    where x is the state and u is the control input.
    """

    def __init__(self, state_dim: int, control_dim: int, name: str):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.name = name

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the time derivative of the state.

        Args:
            t: Current time (seconds).
            x: State vector of shape (state_dim,).
            u: Control input of shape (control_dim,).

        Returns:
            dxdt: State derivative of shape (state_dim,).
        """

    @property
    @abstractmethod
    def state_labels(self) -> list[str]:
        """Names of the state variables (for plotting)."""

    @property
    @abstractmethod
    def control_labels(self) -> list[str]:
        """Names of the control inputs (for plotting)."""

    @property
    @abstractmethod
    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Typical state bounds (x_min, x_max) for normalization / MPC constraints.
        Returns two arrays of shape (state_dim,).
        """

    @property
    @abstractmethod
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Control input bounds (u_min, u_max).
        Returns two arrays of shape (control_dim,).
        """

    def simulate(
        self,
        x0: np.ndarray,
        u_traj: np.ndarray,
        t_span: tuple[float, float],
        dt: float = 0.01,
        noise_std: float = 0.0,
    ) -> dict:
        """
        Simulate the system given an open-loop control trajectory.

        Args:
            x0: Initial state (state_dim,).
            u_traj: Control trajectory of shape (N, control_dim) or (N,).
            t_span: (t_start, t_end).
            dt: Time step for output.
            noise_std: Standard deviation of additive Gaussian measurement noise.

        Returns:
            dict with keys: 't', 'x', 'u', 'dxdt'.
        """
        t_start, t_end = t_span
        t_eval = np.arange(t_start, t_end, dt)
        # Guard against floating-point overshoot beyond t_end
        t_eval = t_eval[t_eval <= t_end + 1e-12]
        t_eval = np.clip(t_eval, t_start, t_end)
        N = len(t_eval)

        if u_traj.ndim == 1:
            u_traj = u_traj.reshape(-1, 1)

        # Interpolate control to match integration times
        u_times = np.linspace(t_start, t_end, len(u_traj))

        def _get_u(t):
            idx = np.searchsorted(u_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(u_traj) - 1)
            return u_traj[idx]

        def ode_rhs(t, x):
            u = _get_u(t)
            return self.dynamics(t, x, u)

        sol = solve_ivp(
            ode_rhs,
            t_span,
            x0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
        )

        x = sol.y.T  # (N, state_dim)
        t = sol.t    # (N,)

        # Collect control at each output step
        u_out = np.array([_get_u(ti) for ti in t])  # (N, control_dim)

        # Compute derivatives
        dxdt = np.array([
            self.dynamics(t[i], x[i], u_out[i]) for i in range(len(t))
        ])

        # Add measurement noise
        if noise_std > 0:
            x_noisy = x + np.random.normal(0, noise_std, x.shape)
        else:
            x_noisy = x.copy()

        return {
            "t": t,
            "x": x_noisy,
            "x_clean": x,
            "u": u_out,
            "dxdt": dxdt,
        }

    def linearize(
        self, x_eq: np.ndarray, u_eq: np.ndarray, eps: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Numerically linearize the system around an equilibrium (x_eq, u_eq).

        Returns:
            A: (state_dim, state_dim) Jacobian wrt state.
            B: (state_dim, control_dim) Jacobian wrt control.
        """
        n, m = self.state_dim, self.control_dim
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        f0 = self.dynamics(0.0, x_eq, u_eq)

        for i in range(n):
            x_plus = x_eq.copy()
            x_plus[i] += eps
            A[:, i] = (self.dynamics(0.0, x_plus, u_eq) - f0) / eps

        for j in range(m):
            u_plus = u_eq.copy()
            u_plus[j] += eps
            B[:, j] = (self.dynamics(0.0, x_eq, u_plus) - f0) / eps

        return A, B

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"control_dim={self.control_dim})"
        )
