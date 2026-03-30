"""Tests for PINN model architecture."""

import pytest
import numpy as np
import torch

from src.models import PINN, PINNConfig, EnsemblePINN


class TestPINNConfig:
    def test_default_config(self):
        cfg = PINNConfig()
        assert cfg.state_dim == 2
        assert cfg.control_dim == 1
        assert len(cfg.hidden_dims) > 0

    def test_custom_config(self):
        cfg = PINNConfig(state_dim=4, control_dim=2, hidden_dims=[128, 128])
        assert cfg.state_dim == 4
        assert cfg.hidden_dims == [128, 128]


class TestPINN:
    def setup_method(self):
        self.cfg = PINNConfig(state_dim=2, control_dim=1, hidden_dims=[32, 32])
        self.model = PINN(self.cfg)

    def test_forward_shape(self):
        x = torch.randn(16, 2)
        u = torch.randn(16, 1)
        out = self.model(x, u)
        assert out.shape == (16, 2)

    def test_forward_single_sample(self):
        x = torch.randn(1, 2)
        u = torch.randn(1, 1)
        out = self.model(x, u)
        assert out.shape == (1, 2)

    def test_predict_numpy(self):
        x = np.random.randn(2)
        u = np.random.randn(1)
        out = self.model.predict_numpy(x, u)
        assert out.shape == (2,)

    def test_predict_numpy_batch(self):
        x = np.random.randn(10, 2)
        u = np.random.randn(10, 1)
        out = self.model.predict_numpy(x, u)
        assert out.shape == (10, 2)

    def test_output_is_finite(self):
        x = torch.randn(32, 2)
        u = torch.randn(32, 1)
        out = self.model(x, u)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self):
        x = torch.randn(8, 2, requires_grad=True)
        u = torch.randn(8, 1, requires_grad=True)
        out = self.model(x, u)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert u.grad is not None

    def test_count_parameters_positive(self):
        assert self.model.count_parameters() > 0

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "model.pt")
        x = torch.randn(4, 2)
        u = torch.randn(4, 1)

        with torch.no_grad():
            out_before = self.model(x, u).clone()

        self.model.save(path)
        loaded = PINN.load(path)
        loaded.eval()

        with torch.no_grad():
            out_after = loaded(x, u)

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_rollout_shape(self):
        N = 10
        x0 = torch.randn(2)
        u_seq = torch.randn(N, 1)
        traj = self.model.rollout(x0, u_seq, dt=0.05, method="rk4")
        assert traj.shape == (N + 1, 2)

    def test_rollout_rk4_vs_euler_close(self):
        """RK4 and Euler should agree for small dt."""
        x0 = torch.zeros(2)
        u_seq = torch.zeros(5, 1)
        traj_euler = self.model.rollout(x0, u_seq, dt=0.001, method="euler")
        traj_rk4   = self.model.rollout(x0, u_seq, dt=0.001, method="rk4")
        assert torch.allclose(traj_euler, traj_rk4, atol=1e-4)

    def test_fourier_feature_variant(self):
        cfg = PINNConfig(
            state_dim=2, control_dim=1,
            hidden_dims=[32, 32],
            use_fourier_features=True,
            fourier_dim=16,
        )
        model = PINN(cfg)
        x = torch.randn(8, 2)
        u = torch.randn(8, 1)
        out = model(x, u)
        assert out.shape == (8, 2)

    def test_residual_vs_plain(self):
        """Both residual and plain MLP should produce finite outputs."""
        for use_res in [True, False]:
            cfg = PINNConfig(state_dim=2, control_dim=1,
                             hidden_dims=[32, 32], use_residual=use_res)
            m = PINN(cfg)
            x = torch.randn(4, 2)
            u = torch.randn(4, 1)
            out = m(x, u)
            assert torch.isfinite(out).all()


class TestEnsemblePINN:
    def setup_method(self):
        self.cfg = PINNConfig(state_dim=2, control_dim=1, hidden_dims=[16, 16])
        self.ensemble = EnsemblePINN(self.cfg, n_members=3)

    def test_mean_shape(self):
        x = torch.randn(8, 2)
        u = torch.randn(8, 1)
        mean, std = self.ensemble(x, u)
        assert mean.shape == (8, 2)
        assert std.shape == (8, 2)

    def test_std_nonnegative(self):
        x = torch.randn(16, 2)
        u = torch.randn(16, 1)
        _, std = self.ensemble(x, u)
        assert (std >= 0).all()

    def test_predict_numpy(self):
        x = np.random.randn(2)
        u = np.random.randn(1)
        mean, std = self.ensemble.predict_numpy(x, u)
        assert mean.shape == (2,)
        assert std.shape == (2,)

    def test_save_load(self, tmp_path):
        path = str(tmp_path / "ensemble.pt")
        x = torch.randn(4, 2)
        u = torch.randn(4, 1)
        with torch.no_grad():
            mean_before, _ = self.ensemble(x, u)

        self.ensemble.save(path)
        loaded = EnsemblePINN.load(path)
        loaded.eval()

        with torch.no_grad():
            mean_after, _ = loaded(x, u)

        assert torch.allclose(mean_before, mean_after, atol=1e-6)
