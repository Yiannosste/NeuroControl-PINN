"""
Physics-Informed Neural Network (PINN) for learning dynamical system equations.

Architecture:
    Input:  [x (state), u (control)]  →  hidden layers  →  Output: dx/dt

The network is trained with a composite loss:
    L = λ_data * L_data + λ_phys * L_phys + λ_ic * L_ic

where:
    L_data: MSE between predicted and observed state derivatives
    L_phys: Residuals of known physics constraints (ODE structure)
    L_ic  : MSE at initial conditions

The architecture uses a modified MLP with:
    - Fourier feature embedding for temporal inputs (optional)
    - Tanh activations (standard for PINNs, smooth derivatives)
    - Xavier initialization
    - Optional residual (skip) connections

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks: A deep learning framework for solving
    forward and inverse problems involving nonlinear PDEs.
    Journal of Computational Physics, 378, 686-707.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray


@dataclass
class PINNConfig:
    """Configuration for the PINN architecture."""
    state_dim: int = 2
    control_dim: int = 1
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"         # 'tanh', 'silu', 'gelu'
    use_residual: bool = True         # residual skip connections
    use_fourier_features: bool = False
    fourier_dim: int = 32             # number of Fourier features
    fourier_scale: float = 1.0        # frequency scale
    output_scaling: bool = True       # learnable output scale / shift
    dropout: float = 0.0              # dropout rate (0 = disabled)


class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature embedding for improved spectral bias mitigation.

    Maps input x → [sin(Bx), cos(Bx)] where B ~ N(0, σ²).

    Reference:
        Tancik, M. et al. (2020). Fourier Features Let Networks Learn
        High Frequency Functions in Low Dimensional Domains. NeurIPS.
    """

    def __init__(self, input_dim: int, num_features: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(input_dim, num_features) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xB = x @ self.B  # (..., num_features)
        return torch.cat([torch.sin(xB), torch.cos(xB)], dim=-1)  # (..., 2*num_features)

    @property
    def output_dim(self) -> int:
        return 2 * self.B.shape[1]


class ResidualBlock(nn.Module):
    """Single residual block: Linear → Activation → Linear + skip."""

    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim),
        ]
        if dropout > 0:
            layers.insert(2, nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
        self.act = activation.__class__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for learning state-space dynamics:
        f̂(x, u) ≈ dx/dt

    The network takes [state, control] as input and predicts the
    time derivative of the state.

    Args:
        config: PINNConfig instance.
    """

    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.control_dim = config.control_dim
        input_dim = config.state_dim + config.control_dim

        # Activation function
        act_map = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        ActClass = act_map.get(config.activation, nn.Tanh)

        # --- Input embedding ---
        if config.use_fourier_features:
            self.embedding = FourierFeatureEmbedding(
                input_dim, config.fourier_dim, config.fourier_scale
            )
            embed_dim = self.embedding.output_dim
        else:
            self.embedding = None
            embed_dim = input_dim

        # --- Build MLP ---
        dims = [embed_dim] + config.hidden_dims

        if config.use_residual:
            # Project to first hidden dim
            self.input_proj = nn.Linear(embed_dim, config.hidden_dims[0])
            blocks = []
            projections = []
            prev_dim = config.hidden_dims[0]
            for d in config.hidden_dims:
                # A ResidualBlock adds its input back to its output, so the
                # incoming tensor must already be at width `d`. Project
                # between blocks whenever hidden_dims changes width (e.g.
                # a [128, 128, 64] taper) — Identity when the width repeats.
                projections.append(
                    nn.Linear(prev_dim, d) if d != prev_dim else nn.Identity()
                )
                blocks.append(ResidualBlock(d, ActClass(), config.dropout))
                prev_dim = d
            self.block_projections = nn.ModuleList(projections)
            self.blocks = nn.ModuleList(blocks)
            out_in = config.hidden_dims[-1]
        else:
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(ActClass())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
            self.net = nn.Sequential(*layers)
            out_in = config.hidden_dims[-1]

        # --- Output head ---
        self.output_layer = nn.Linear(out_in, config.state_dim)

        if config.output_scaling:
            self.output_scale = nn.Parameter(torch.ones(config.state_dim))
            self.output_shift = nn.Parameter(torch.zeros(config.state_dim))
        else:
            self.output_scale = None
            self.output_shift = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stable PINN training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Predict dx/dt given state x and control u.

        Args:
            x: State tensor of shape (..., state_dim).
            u: Control tensor of shape (..., control_dim).

        Returns:
            dxdt: Predicted state derivative of shape (..., state_dim).
        """
        inp = torch.cat([x, u], dim=-1)  # (..., state_dim + control_dim)

        if self.embedding is not None:
            inp = self.embedding(inp)

        if self.config.use_residual:
            h = self.input_proj(inp)
            for proj, block in zip(self.block_projections, self.blocks):
                h = block(proj(h))
        else:
            h = self.net(inp)

        out = self.output_layer(h)

        if self.output_scale is not None:
            out = out * self.output_scale + self.output_shift

        return out

    def predict_numpy(
        self, x: NDArray[np.float32], u: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Predict dx/dt from numpy arrays (for use in scipy MPC).

        The model may reside on any device (CPU or GPU); this method
        always returns a CPU NumPy array regardless.

        Args:
            x: State array of shape (state_dim,) or (N, state_dim).
            u: Control array of shape (control_dim,) or (N, control_dim).

        Returns:
            dxdt: NumPy array of shape matching input.
        """
        squeezed = x.ndim == 1
        if squeezed:
            x = x[np.newaxis, :]
            u = u[np.newaxis, :]

        device = next(self.parameters()).device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        u_t = torch.as_tensor(u, dtype=torch.float32, device=device)

        with torch.no_grad():
            out: NDArray[np.float32] = (
                self.forward(x_t, u_t).detach().cpu().numpy()
            )

        return out[0] if squeezed else out

    def predict_with_grad(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict dx/dt and compute Jacobians wrt x and u.
        Used for gradient-based MPC optimization.

        Returns:
            dxdt: (batch, state_dim)
            J_x:  (batch, state_dim, state_dim)  — Jacobian wrt x
            J_u:  (batch, state_dim, control_dim) — Jacobian wrt u
        """
        x = x.requires_grad_(True)
        u = u.requires_grad_(True)

        dxdt = self.forward(x, u)  # (batch, state_dim)

        J_x_rows, J_u_rows = [], []
        for i in range(self.state_dim):
            grad_x = torch.autograd.grad(
                dxdt[:, i].sum(), x, create_graph=True, retain_graph=True
            )[0]
            grad_u = torch.autograd.grad(
                dxdt[:, i].sum(), u, create_graph=True, retain_graph=True
            )[0]
            J_x_rows.append(grad_x.unsqueeze(1))
            J_u_rows.append(grad_u.unsqueeze(1))

        J_x = torch.cat(J_x_rows, dim=1)  # (batch, state_dim, state_dim)
        J_u = torch.cat(J_u_rows, dim=1)  # (batch, state_dim, control_dim)

        return dxdt, J_x, J_u

    def rollout(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor,
        dt: float,
        method: str = "rk4",
    ) -> torch.Tensor:  # (N+1, state_dim) or (batch, N+1, state_dim)
        """
        Integrate the PINN dynamics over a control sequence.

        Args:
            x0: Initial state (..., state_dim).
            u_seq: Control sequence (N, control_dim) or (batch, N, control_dim).
            dt: Time step.
            method: Integration method ('euler', 'rk4').

        Returns:
            x_traj: State trajectory (N+1, state_dim) or (batch, N+1, state_dim).
        """
        N = u_seq.shape[-2]
        traj = [x0]
        x = x0

        for k in range(N):
            u_k = u_seq[..., k, :]

            if method == "euler":
                x = x + dt * self.forward(x, u_k)
            elif method == "rk4":
                k1 = self.forward(x, u_k)
                k2 = self.forward(x + 0.5 * dt * k1, u_k)
                k3 = self.forward(x + 0.5 * dt * k2, u_k)
                k4 = self.forward(x + dt * k3, u_k)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj.append(x)

        return torch.stack(traj, dim=-2)  # (..., N+1, state_dim)

    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PINN":
        """Load model from file."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"PINN(state_dim={self.state_dim}, control_dim={self.control_dim}, "
            f"hidden={self.config.hidden_dims}, params={self.count_parameters():,})"
        )
