"""
Van der Pol Oscillator with external forcing.

Equations:
    dx1/dt = x2
    dx2/dt = mu * (1 - x1^2) * x2 - x1 + u

The unforced system (u=0) exhibits a stable limit cycle.
With control input u, the objective is typically to drive the system
to the unstable equilibrium (0, 0) or to track a reference trajectory.

Reference:
    Van der Pol, B. (1926). On "relaxation-oscillations".
    The London, Edinburgh, and Dublin Philosophical Magazine, 2(11), 978-992.
"""

import numpy as np
from .base import DynamicalSystem


class VanDerPolSystem(DynamicalSystem):
    """
    Controlled Van der Pol oscillator.

    State:  x = [x1, x2]  (position, velocity)
    Control: u  (scalar forcing)
    """

    def __init__(self, mu: float = 1.0):
        """
        Args:
            mu: Nonlinearity parameter. mu > 0 gives a stable limit cycle.
                At mu = 0 the system reduces to a harmonic oscillator.
        """
        super().__init__(state_dim=2, control_dim=1, name="VanDerPol")
        self.mu = mu

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        u_val = float(u[0]) if hasattr(u, "__len__") else float(u)
        dx1 = x2
        dx2 = self.mu * (1.0 - x1**2) * x2 - x1 + u_val
        return np.array([dx1, dx2])

    @property
    def state_labels(self) -> list[str]:
        return ["x₁ (position)", "x₂ (velocity)"]

    @property
    def control_labels(self) -> list[str]:
        return ["u (force)"]

    @property
    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-3.0, -5.0]), np.array([3.0, 5.0])

    @property
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-5.0]), np.array([5.0])

    @property
    def equilibrium(self) -> np.ndarray:
        """Unstable equilibrium at the origin."""
        return np.zeros(2)

    def physics_residual(
        self, x: np.ndarray, u: np.ndarray, dxdt_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute the physics residual for use in PINN training.

        The known structure of dx1/dt = x2 is exact, so we only penalise
        the mismatch in x2's equation.

        Returns residual of shape (..., 2).
        """
        # x shape: (..., 2), u shape: (..., 1), dxdt_pred shape: (..., 2)
        x1, x2 = x[..., 0:1], x[..., 1:2]
        u_val = u[..., 0:1]

        # Exact residuals from ODE structure
        res1 = dxdt_pred[..., 0:1] - x2               # dx1/dt = x2 (exact)
        res2 = (
            dxdt_pred[..., 1:2]
            - (self.mu * (1.0 - x1**2) * x2 - x1 + u_val)
        )
        import numpy as np
        return np.concatenate([res1, res2], axis=-1)

    def generate_training_data(
        self,
        n_trajectories: int = 50,
        t_end: float = 10.0,
        dt: float = 0.05,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> dict:
        """
        Generate training data by simulating multiple trajectories with
        random initial conditions and random control inputs.

        Returns:
            dict with concatenated arrays 't', 'x', 'u', 'dxdt'.
        """
        rng = np.random.RandomState(seed)
        x_min, x_max = self.state_bounds
        u_min, u_max = self.control_bounds

        all_data = {"t": [], "x": [], "u": [], "dxdt": []}

        for _ in range(n_trajectories):
            x0 = rng.uniform(x_min, x_max)
            n_steps = int(t_end / dt)
            # Random piecewise-constant control
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
