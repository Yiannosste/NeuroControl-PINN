"""
Inverted Pendulum on a Cart (Cart-Pole).

Equations of motion (Euler-Lagrange):
    (M + m) * x_ddot  - m * l * cos(θ) * θ_ddot + m * l * sin(θ) * θ_dot^2 = F
    m * l^2 * θ_ddot  - m * l * cos(θ) * x_ddot - m * g * l * sin(θ) = 0

State: x = [p, p_dot, θ, θ_dot]
    p     : cart position (m)
    p_dot : cart velocity (m/s)
    θ     : pole angle from upright (rad)  [θ=0 → upright]
    θ_dot : pole angular velocity (rad/s)

Control: u = F (horizontal force on cart, N)

Goal: Stabilize the pole in the upright position (θ = 0) while
      keeping the cart near the center (p = 0).

Reference:
    Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983).
    Neuronlike adaptive elements that can solve difficult learning control problems.
    IEEE transactions on systems, man, and cybernetics, (5), 834-846.
"""

import numpy as np
from .base import DynamicalSystem


class CartPoleSystem(DynamicalSystem):
    """
    Inverted Pendulum on a Cart.

    State:   x = [p, p_dot, theta, theta_dot]
    Control: u = [F]  (horizontal force on cart)
    """

    def __init__(
        self,
        M: float = 1.0,   # cart mass (kg)
        m: float = 0.1,   # pole mass (kg)
        l: float = 0.5,   # half-pole length (m)
        g: float = 9.81,  # gravity (m/s^2)
    ):
        super().__init__(state_dim=4, control_dim=1, name="CartPole")
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p, p_dot, theta, theta_dot = x
        F = float(u[0]) if hasattr(u, "__len__") else float(u)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        M, m, l, g = self.M, self.m, self.l, self.g

        total_mass = M + m
        ml = m * l

        # Denominator (from Euler-Lagrange)
        denom = total_mass - m * cos_theta**2

        theta_ddot = (
            total_mass * g * sin_theta
            - cos_theta * (F + ml * theta_dot**2 * sin_theta)
        ) / (l * denom)

        p_ddot = (
            F
            + ml * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)
        ) / total_mass

        return np.array([p_dot, p_ddot, theta_dot, theta_ddot])

    @property
    def state_labels(self) -> list[str]:
        return [
            "p (cart pos, m)",
            "ṗ (cart vel, m/s)",
            "θ (pole angle, rad)",
            "θ̇ (pole ang vel, rad/s)",
        ]

    @property
    def control_labels(self) -> list[str]:
        return ["F (force, N)"]

    @property
    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-2.4, -3.0, -0.3, -3.0]),
            np.array([ 2.4,  3.0,  0.3,  3.0]),
        )

    @property
    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-10.0]), np.array([10.0])

    @property
    def equilibrium(self) -> np.ndarray:
        """Upright unstable equilibrium."""
        return np.zeros(4)

    def generate_training_data(
        self,
        n_trajectories: int = 80,
        t_end: float = 5.0,
        dt: float = 0.02,
        noise_std: float = 0.02,
        seed: int = 0,
    ) -> dict:
        """
        Generate training data from random initial conditions and controls.
        Focuses on the near-upright region to capture the relevant dynamics.
        """
        rng = np.random.RandomState(seed)
        u_min, u_max = self.control_bounds

        all_data = {"t": [], "x": [], "u": [], "dxdt": []}

        for _ in range(n_trajectories):
            # Sample near the upright equilibrium (harder, more useful data)
            x0 = rng.uniform(
                [-0.5, -0.5, -0.25, -0.5],
                [ 0.5,  0.5,  0.25,  0.5],
            )
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
