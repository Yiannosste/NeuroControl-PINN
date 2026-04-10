"""
PINN Trainer with Adam + L-BFGS two-phase training strategy.

Phase 1 — Adam:  Fast exploration of the loss landscape. Adam is robust
                 to noise in the gradients, converges quickly to a good basin.

Phase 2 — L-BFGS: Fine-tuning with second-order information. L-BFGS
                   can achieve much lower loss values than Adam for PINN-type
                   objectives (smooth, low-noise).

This two-phase strategy is the de-facto standard in the PINN literature.

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks. Journal of Computational Physics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .losses import PINNLoss
from ..models.pinn import PINN


@dataclass
class TrainingConfig:
    """Hyperparameters for two-phase PINN training."""
    # Adam phase
    adam_epochs: int = 5000
    adam_lr: float = 1e-3
    adam_lr_decay_step: int = 1000
    adam_lr_decay_gamma: float = 0.5
    adam_weight_decay: float = 1e-5
    batch_size: int = 1024

    # L-BFGS phase
    lbfgs_epochs: int = 500
    lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 50
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1e-9
    lbfgs_tolerance_change: float = 1e-12

    # Collocation
    n_collocation: int = 2048        # points sampled per epoch for physics loss
    collocation_resample: bool = True # re-sample collocation points each epoch

    # Misc
    seed: int = 42
    device: str = "cpu"
    log_every: int = 100
    patience: int = 1000             # early stopping patience (Adam steps)
    min_delta: float = 1e-7


class PINNTrainer:
    """
    Two-phase PINN trainer.

    Usage:
        trainer = PINNTrainer(model, loss_fn, config)
        history = trainer.fit(
            x_train, u_train, dxdt_train,
            x_bounds=system.state_bounds,
            u_bounds=system.control_bounds,   # empirical control bounds
        )
    """

    def __init__(
        self,
        model: PINN,
        loss_fn: PINNLoss,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.history: dict[str, list] = {
            "total": [], "data": [], "physics": [],
            "val_total": [], "val_data": [],
            "epoch": [], "phase": [],
        }

    def _to_device(self, *tensors):
        return tuple(t.to(self.device) for t in tensors)

    def _make_dataset(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dxdt: np.ndarray,
    ) -> TensorDataset:
        X = torch.FloatTensor(x)
        U = torch.FloatTensor(u)
        D = torch.FloatTensor(dxdt)
        return TensorDataset(X, U, D)

    def _sample_collocation(
        self,
        x_bounds: tuple[np.ndarray, np.ndarray],
        u_bounds: tuple[np.ndarray, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random collocation points uniformly in state-control space.

        Control inputs are sampled within the empirical bounds derived from
        the training dataset (normalised space), ensuring the physics loss
        is enforced across the full operational envelope rather than a
        hardcoded [-1, 1] interval.

        Args:
            x_bounds: (x_min, x_max) arrays of shape (state_dim,).
            u_bounds: (u_min, u_max) arrays of shape (control_dim,).

        Returns:
            (x_col, u_col) tensors on self.device.
        """
        x_min, x_max = x_bounds
        u_min, u_max = u_bounds
        n = self.config.n_collocation
        n_state = len(x_min)
        n_ctrl = len(u_min)

        x_col = np.random.uniform(
            x_min, x_max, size=(n, n_state)
        ).astype(np.float32)
        u_col = np.random.uniform(
            u_min, u_max, size=(n, n_ctrl)
        ).astype(np.float32)

        return (
            torch.FloatTensor(x_col).to(self.device),
            torch.FloatTensor(u_col).to(self.device),
        )

    def fit(
        self,
        x_train: np.ndarray,
        u_train: np.ndarray,
        dxdt_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        u_val: Optional[np.ndarray] = None,
        dxdt_val: Optional[np.ndarray] = None,
        x_bounds: Optional[tuple] = None,
        u_bounds: Optional[tuple] = None,
    ) -> dict:
        """
        Train the PINN on trajectory data.

        Args:
            x_train: States (N, state_dim).
            u_train: Controls (N, control_dim).
            dxdt_train: State derivatives (N, state_dim).
            x_val, u_val, dxdt_val: Optional validation data.
            x_bounds: (x_min, x_max) for state collocation sampling.
                      Defaults to empirical min/max of x_train.
            u_bounds: (u_min, u_max) for control collocation sampling.
                      Defaults to empirical min/max of u_train.  Pass the
                      system's control_bounds for normalised datasets so
                      the physics loss covers the full operational envelope.

        Returns:
            Training history dict.
        """
        if u_train.ndim == 1:
            u_train = u_train[:, np.newaxis]
        if u_val is not None and u_val.ndim == 1:
            u_val = u_val[:, np.newaxis]

        train_dataset = self._make_dataset(x_train, u_train, dxdt_train)
        loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_tensors = None
        if x_val is not None:
            val_tensors = tuple(self._to_device(*[
                torch.FloatTensor(arr) for arr in [x_val, u_val, dxdt_val]
            ]))

        # Compute state bounds from data if not provided.
        if x_bounds is None:
            x_min = x_train.min(axis=0)
            x_max = x_train.max(axis=0)
            x_bounds = (x_min, x_max)

        # Compute control bounds from data if not provided.
        # Using the empirical dataset range ensures collocation covers
        # every operating condition seen during data collection.
        if u_bounds is None:
            u_min = u_train.min(axis=0)
            u_max = u_train.max(axis=0)
            u_bounds = (u_min, u_max)

        print("=" * 60)
        print(f"  PINN Training — {self.model}")
        print(f"  Train samples: {len(x_train):,} | Batch size: {self.config.batch_size}")
        print(f"  Adam epochs: {self.config.adam_epochs} | L-BFGS epochs: {self.config.lbfgs_epochs}")
        print("=" * 60)

        # ── Phase 1: Adam ────────────────────────────────────────────
        self._train_adam(loader, val_tensors, x_bounds, u_bounds)

        # ── Phase 2: L-BFGS ──────────────────────────────────────────
        x_all = torch.FloatTensor(x_train).to(self.device)
        u_all = torch.FloatTensor(u_train).to(self.device)
        d_all = torch.FloatTensor(dxdt_train).to(self.device)
        self._train_lbfgs(x_all, u_all, d_all, x_bounds, u_bounds)

        return self.history

    def _train_adam(self, loader, val_tensors, x_bounds, u_bounds):
        cfg = self.config
        opt = optim.Adam(
            self.model.parameters(),
            lr=cfg.adam_lr,
            weight_decay=cfg.adam_weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            opt, step_size=cfg.adam_lr_decay_step, gamma=cfg.adam_lr_decay_gamma
        )

        best_loss = float("inf")
        patience_count = 0
        t0 = time.time()

        for epoch in range(cfg.adam_epochs):
            self.model.train()
            epoch_losses = {"total": 0.0, "data": 0.0, "physics": 0.0}
            n_batches = 0

            # Re-sample collocation points each epoch (or once if disabled).
            if cfg.collocation_resample:
                x_col, u_col = self._sample_collocation(x_bounds, u_bounds)
            else:
                if epoch == 0:
                    x_col, u_col = self._sample_collocation(x_bounds, u_bounds)

            for x_b, u_b, d_b in loader:
                x_b, u_b, d_b = self._to_device(x_b, u_b, d_b)

                opt.zero_grad()
                dxdt_pred = self.model(x_b, u_b)
                losses = self.loss_fn(
                    self.model, dxdt_pred, d_b, x_col, u_col
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()
                n_batches += 1

            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] /= n_batches

            # Validation
            val_loss = self._evaluate(val_tensors) if val_tensors else None

            self.history["total"].append(epoch_losses["total"])
            self.history["data"].append(epoch_losses["data"])
            self.history["physics"].append(epoch_losses["physics"])
            self.history["val_total"].append(val_loss)
            self.history["val_data"].append(val_loss)
            self.history["epoch"].append(epoch)
            self.history["phase"].append("adam")

            # Early stopping
            if epoch_losses["total"] < best_loss - cfg.min_delta:
                best_loss = epoch_losses["total"]
                patience_count = 0
                self._save_best()
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            if epoch % cfg.log_every == 0:
                elapsed = time.time() - t0
                val_str = f" | val: {val_loss:.4e}" if val_loss else ""
                print(
                    f"  [Adam {epoch:5d}/{cfg.adam_epochs}] "
                    f"total: {epoch_losses['total']:.4e} "
                    f"data: {epoch_losses['data']:.4e} "
                    f"phys: {epoch_losses['physics']:.4e}"
                    f"{val_str} | {elapsed:.1f}s"
                )

        self._load_best()
        print(f"  Adam done. Best loss: {best_loss:.4e}")

    def _train_lbfgs(self, x_all, u_all, d_all, x_bounds, u_bounds):
        """
        L-BFGS fine-tuning phase.

        Collocation points are resampled once per epoch (i.e. once per
        call to opt.step) rather than once per closure invocation.
        L-BFGS performs multiple closure evaluations per step during its
        line-search; resampling inside the closure would make the
        objective stochastic within a single step, breaking the Wolfe
        conditions and causing divergence.  Resampling per epoch
        preserves a consistent objective within each step while still
        preventing overfitting to a fixed collocation set.

        When collocation_resample=False the points are fixed for the
        entire L-BFGS phase (original behaviour; useful for memory-
        constrained environments or deterministic reproducibility).
        """
        cfg = self.config
        opt = optim.LBFGS(
            self.model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )

        # Sample the initial collocation set used when resampling is disabled.
        x_col, u_col = self._sample_collocation(x_bounds, u_bounds)

        best_loss = float("inf")
        t0 = time.time()

        for epoch in range(cfg.lbfgs_epochs):
            self.model.train()

            # Resample at the start of each epoch so the closure is
            # consistent throughout the entire line-search for this step.
            if cfg.collocation_resample and epoch > 0:
                x_col, u_col = self._sample_collocation(x_bounds, u_bounds)

            # Capture the epoch-local collocation tensors in the closure
            # via default arguments to avoid late-binding issues.
            def closure(xc=x_col, uc=u_col):
                opt.zero_grad()
                dxdt_pred = self.model(x_all, u_all)
                losses = self.loss_fn(self.model, dxdt_pred, d_all, xc, uc)
                losses["total"].backward()
                return losses["total"]

            loss = opt.step(closure)
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)

            self.history["total"].append(loss_val)
            self.history["data"].append(loss_val)
            self.history["physics"].append(0.0)
            self.history["val_total"].append(None)
            self.history["val_data"].append(None)
            self.history["epoch"].append(epoch)
            self.history["phase"].append("lbfgs")

            if loss_val < best_loss:
                best_loss = loss_val
                self._save_best()

            if epoch % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [L-BFGS {epoch:4d}/{cfg.lbfgs_epochs}] "
                    f"loss: {loss_val:.4e} | {elapsed:.1f}s"
                )

        self._load_best()
        print(f"  L-BFGS done. Best loss: {best_loss:.4e}")

    def _evaluate(self, val_tensors) -> float:
        """Compute validation loss."""
        self.model.eval()
        x_v, u_v, d_v = val_tensors
        with torch.no_grad():
            dxdt_pred = self.model(x_v, u_v)
            loss = ((dxdt_pred - d_v) ** 2).mean().item()
        return loss

    def _save_best(self):
        self._best_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

    def _load_best(self):
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
