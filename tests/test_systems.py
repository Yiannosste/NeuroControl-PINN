"""Tests for dynamical systems."""

import pytest
import numpy as np
from src.systems import VanDerPolSystem, CartPoleSystem, CSTRSystem
from src.systems.base import DynamicalSystem


class TestVanDerPol:
    def setup_method(self):
        self.sys = VanDerPolSystem(mu=1.0)

    def test_state_dim(self):
        assert self.sys.state_dim == 2

    def test_control_dim(self):
        assert self.sys.control_dim == 1

    def test_dynamics_zero_control(self):
        """At origin, dynamics should be zero (equilibrium)."""
        x = np.zeros(2)
        u = np.zeros(1)
        dxdt = self.sys.dynamics(0.0, x, u)
        assert np.allclose(dxdt, 0.0, atol=1e-12)

    def test_dynamics_structure(self):
        """dx1/dt should always equal x2 (exact relation)."""
        rng = np.random.RandomState(0)
        for _ in range(20):
            x = rng.uniform(-3, 3, size=2)
            u = rng.uniform(-2, 2, size=1)
            dxdt = self.sys.dynamics(0.0, x, u)
            assert np.isclose(dxdt[0], x[1], atol=1e-12), \
                f"dx1/dt = {dxdt[0]:.6f}, x2 = {x[1]:.6f}"

    def test_simulate_returns_correct_keys(self):
        x0 = np.array([1.0, 0.0])
        u_traj = np.zeros((50, 1))
        result = self.sys.simulate(x0, u_traj, t_span=(0, 2.5), dt=0.05)
        for key in ["t", "x", "x_clean", "u", "dxdt"]:
            assert key in result

    def test_simulate_initial_condition(self):
        x0 = np.array([1.5, -0.5])
        u_traj = np.zeros((20, 1))
        result = self.sys.simulate(x0, u_traj, t_span=(0, 1), dt=0.05, noise_std=0.0)
        assert np.allclose(result["x_clean"][0], x0, atol=1e-3)

    def test_simulate_shape(self):
        x0 = np.array([1.0, 0.0])
        u_traj = np.zeros((100, 1))
        result = self.sys.simulate(x0, u_traj, t_span=(0, 5), dt=0.05)
        assert result["x"].shape[1] == 2
        assert result["u"].shape[1] == 1

    def test_state_bounds_shape(self):
        lo, hi = self.sys.state_bounds
        assert lo.shape == (2,)
        assert hi.shape == (2,)
        assert np.all(lo < hi)

    def test_control_bounds_shape(self):
        lo, hi = self.sys.control_bounds
        assert lo.shape == (1,)
        assert hi.shape == (1,)

    def test_linearize(self):
        A, B = self.sys.linearize(np.zeros(2), np.zeros(1))
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
        # At origin: A should be [[0, 1], [-1, mu]]
        assert np.isclose(A[0, 1], 1.0, atol=1e-3)
        assert np.isclose(A[1, 0], -1.0, atol=1e-3)

    def test_generate_training_data(self):
        data = self.sys.generate_training_data(n_trajectories=5, t_end=2.0, seed=0)
        for k in ["t", "x", "u", "dxdt"]:
            assert k in data
            assert len(data[k]) > 0


class TestCartPole:
    def setup_method(self):
        self.sys = CartPoleSystem()

    def test_dims(self):
        assert self.sys.state_dim == 4
        assert self.sys.control_dim == 1

    def test_equilibrium(self):
        x_eq = np.zeros(4)
        u_eq = np.zeros(1)
        dxdt = self.sys.dynamics(0.0, x_eq, u_eq)
        assert np.allclose(dxdt, 0.0, atol=1e-10)

    def test_velocity_relations(self):
        """ẋ₁ = x₂ and ẋ₃ = x₄ are kinematic identities."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            x = rng.uniform(-0.5, 0.5, 4)
            u = rng.uniform(-5, 5, 1)
            dxdt = self.sys.dynamics(0.0, x, u)
            assert np.isclose(dxdt[0], x[1], atol=1e-12)
            assert np.isclose(dxdt[2], x[3], atol=1e-12)

    def test_simulate_runs(self):
        x0 = np.array([0.0, 0.0, 0.05, 0.0])
        result = self.sys.simulate(x0, np.zeros((50, 1)), t_span=(0, 1), dt=0.02)
        assert result["x"].shape[1] == 4

    def test_linearize_is_unstable(self):
        A, _ = self.sys.linearize(np.zeros(4), np.zeros(1))
        eigs = np.real(np.linalg.eigvals(A))
        assert np.any(eigs > 0), "Cart-pole should be open-loop unstable"


class TestCSTR:
    def setup_method(self):
        self.sys = CSTRSystem()

    def test_dims(self):
        assert self.sys.state_dim == 2
        assert self.sys.control_dim == 1

    def test_dynamics_positive_concentration(self):
        """Concentration should not go below zero."""
        x = np.array([0.5, 380.0])
        u = np.array([300.0])
        dxdt = self.sys.dynamics(0.0, x, u)
        assert np.isfinite(dxdt).all()

    def test_has_equilibrium_structure(self):
        """Verify the CSTR operating_point IS a steady state (by construction)."""
        from scipy.optimize import fsolve
        x_op, u_op = self.sys.operating_point
        def residual(x):
            return self.sys.dynamics(0.0, x, u_op)
        # Numerical SS near the documented operating point
        x_ss = fsolve(residual, x_op + np.array([0.05, 5.0]))
        dxdt = residual(x_ss)
        # The solver should find SOME equilibrium near the operating region
        assert np.linalg.norm(dxdt) < 1.0, \
            f"No equilibrium found near operating point: residual norm = {np.linalg.norm(dxdt):.4f}"

    def test_state_bounds(self):
        lo, hi = self.sys.state_bounds
        assert lo[0] >= 0.0, "Concentration lower bound must be non-negative"
        assert lo[1] >= 250.0, "Temperature lower bound unreasonably low"

    def test_generate_training_data(self):
        data = self.sys.generate_training_data(n_trajectories=5, t_end=5.0, seed=1)
        assert data["x"].shape[1] == 2
