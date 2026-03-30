"""Tests for MPC and PID controllers."""

import pytest
import numpy as np
import torch

from src.systems import VanDerPolSystem
from src.models import PINN, PINNConfig
from src.control import PINNMPC, ClassicalMPC, PIDController, MPCConfig


def make_vdp_and_mpc_config():
    system = VanDerPolSystem(mu=1.0)
    mpc_cfg = MPCConfig(
        horizon=5,
        dt=0.05,
        integration="euler",
        Q=np.eye(2),
        R=np.eye(1) * 0.01,
        P=np.eye(2) * 5.0,
        u_min=np.array([-5.0]),
        u_max=np.array([5.0]),
        warm_start=False,
        max_iter=20,
    )
    return system, mpc_cfg


def make_pinn(system):
    cfg = PINNConfig(
        state_dim=system.state_dim,
        control_dim=system.control_dim,
        hidden_dims=[16, 16],
    )
    return PINN(cfg)


class TestPINNMPC:
    def setup_method(self):
        self.system, self.mpc_cfg = make_vdp_and_mpc_config()
        self.model = make_pinn(self.system)
        self.model.eval()

    def test_solve_returns_correct_shape(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg, x_ref=np.zeros(2))
        x0 = np.array([1.0, 0.5])
        u_opt, info = ctrl.solve(x0)
        assert u_opt.shape == (1,)

    def test_solve_respects_bounds(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg, x_ref=np.zeros(2))
        x0 = np.array([2.0, 0.0])
        u_opt, _ = ctrl.solve(x0)
        assert u_opt[0] >= -5.0 - 1e-6
        assert u_opt[0] <= 5.0 + 1e-6

    def test_solve_info_keys(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg, x_ref=np.zeros(2))
        _, info = ctrl.solve(np.zeros(2))
        for key in ["cost", "success", "n_iter", "solve_time_ms"]:
            assert key in info

    def test_run_closed_loop_shape(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg, x_ref=np.zeros(2))
        result = ctrl.run_closed_loop(self.system, np.array([1.0, 0.0]), n_steps=10)
        assert result["x"].shape == (11, 2)
        assert result["u"].shape == (10, 1)
        assert result["t"].shape == (11,)

    def test_set_reference(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg)
        ctrl.set_reference(np.array([1.0, 2.0]))
        assert np.allclose(ctrl.x_ref, [1.0, 2.0])

    def test_solve_times_logged(self):
        ctrl = PINNMPC(self.model, self.mpc_cfg, x_ref=np.zeros(2))
        for _ in range(3):
            ctrl.solve(np.array([0.5, 0.5]))
        assert len(ctrl.solve_times) == 3
        assert all(t > 0 for t in ctrl.solve_times)


class TestClassicalMPC:
    def setup_method(self):
        self.system, self.mpc_cfg = make_vdp_and_mpc_config()

    def test_solve_returns_correct_shape(self):
        ctrl = ClassicalMPC(self.system, self.mpc_cfg, x_ref=np.zeros(2))
        u_opt, info = ctrl.solve(np.array([1.0, 0.0]))
        assert u_opt.shape == (1,)

    def test_solve_respects_bounds(self):
        ctrl = ClassicalMPC(self.system, self.mpc_cfg, x_ref=np.zeros(2))
        u_opt, _ = ctrl.solve(np.array([2.5, 1.0]))
        assert u_opt[0] >= -5.0 - 1e-6
        assert u_opt[0] <= 5.0 + 1e-6

    def test_run_closed_loop_shape(self):
        ctrl = ClassicalMPC(self.system, self.mpc_cfg, x_ref=np.zeros(2))
        result = ctrl.run_closed_loop(self.system, np.array([1.0, 0.0]), n_steps=5)
        assert result["x"].shape == (6, 2)
        assert result["u"].shape == (5, 1)


class TestPIDController:
    def setup_method(self):
        self.system = VanDerPolSystem()
        self.pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, dt=0.05,
                                 u_min=-5.0, u_max=5.0, state_idx=0)

    def test_step_returns_scalar_array(self):
        u = self.pid.step(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert u.shape == (1,)

    def test_step_respects_bounds(self):
        """Even for large errors the output should stay within bounds."""
        for _ in range(20):
            x = np.array([100.0, 0.0])
            u = self.pid.step(x, np.zeros(2))
            assert u[0] >= -5.0 - 1e-6
            assert u[0] <= 5.0 + 1e-6

    def test_reset_clears_integral(self):
        for _ in range(10):
            self.pid.step(np.array([1.0, 0.0]), np.zeros(2))
        self.pid.reset()
        assert self.pid._integral == 0.0
        assert self.pid._prev_error is None

    def test_run_closed_loop_shape(self):
        result = self.pid.run_closed_loop(
            self.system, np.array([1.0, 0.0]), n_steps=10,
            x_ref_traj=np.zeros((10, 2))
        )
        assert result["x"].shape == (11, 2)
        assert result["u"].shape == (10, 1)
