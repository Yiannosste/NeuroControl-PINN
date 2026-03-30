"""
Experiment 1 — PINN Training.

Trains a Physics-Informed Neural Network on trajectory data from a
specified dynamical system.

Usage:
    python experiments/train_pinn.py --system van_der_pol
    python experiments/train_pinn.py --system cartpole --device cuda
    python experiments/train_pinn.py --system cstr --config experiments/configs/cstr.yaml
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems import VanDerPolSystem, CartPoleSystem, CSTRSystem
from src.models import PINN, PINNConfig
from src.training import PINNTrainer, TrainingConfig, PINNLoss
from src.utils import (
    generate_dataset,
    split_dataset,
    normalize_dataset,
    plot_training_history,
    plot_pinn_predictions,
)


SYSTEM_MAP = {
    "van_der_pol": VanDerPolSystem,
    "cartpole": CartPoleSystem,
    "cstr": CSTRSystem,
}

CONFIG_MAP = {
    "van_der_pol": "experiments/configs/van_der_pol.yaml",
    "cartpole": "experiments/configs/cartpole.yaml",
    "cstr": "experiments/configs/cstr.yaml",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_system(cfg: dict):
    name = cfg["system"]["name"].lower()
    if name == "vanderpol":
        return VanDerPolSystem(mu=cfg["system"].get("mu", 1.0))
    elif name == "cartpole":
        return CartPoleSystem(
            M=cfg["system"].get("M", 1.0),
            m=cfg["system"].get("m", 0.1),
            l=cfg["system"].get("l", 0.5),
        )
    elif name == "cstr":
        return CSTRSystem()
    raise ValueError(f"Unknown system: {name}")


def build_model(cfg: dict, system) -> PINN:
    mc = cfg["model"]
    pinn_cfg = PINNConfig(
        state_dim=system.state_dim,
        control_dim=system.control_dim,
        hidden_dims=mc["hidden_dims"],
        activation=mc.get("activation", "tanh"),
        use_residual=mc.get("use_residual", True),
        use_fourier_features=mc.get("use_fourier_features", False),
        dropout=mc.get("dropout", 0.0),
        output_scaling=mc.get("output_scaling", True),
    )
    return PINN(pinn_cfg)


def build_trainer_config(cfg: dict) -> TrainingConfig:
    tc = cfg["training"]
    return TrainingConfig(
        adam_epochs=tc.get("adam_epochs", 5000),
        adam_lr=tc.get("adam_lr", 1e-3),
        adam_lr_decay_step=tc.get("adam_lr_decay_step", 1000),
        adam_lr_decay_gamma=tc.get("adam_lr_decay_gamma", 0.5),
        adam_weight_decay=tc.get("adam_weight_decay", 1e-5),
        batch_size=tc.get("batch_size", 512),
        lbfgs_epochs=tc.get("lbfgs_epochs", 300),
        lbfgs_lr=tc.get("lbfgs_lr", 0.1),
        n_collocation=tc.get("n_collocation", 2000),
        seed=tc.get("seed", 0),
        device=tc.get("device", "cpu"),
        log_every=100,
        patience=tc.get("patience", 800),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a PINN on dynamical system data.")
    parser.add_argument("--system", type=str, default="van_der_pol",
                        choices=list(SYSTEM_MAP.keys()),
                        help="Dynamical system to train on.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: use built-in config).")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cpu' or 'cuda'.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or CONFIG_MAP[args.system]
    cfg = load_config(config_path)

    if args.device:
        cfg["training"]["device"] = args.device
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    out_dir = cfg.get("output_dir", f"results/{args.system}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Build system ──────────────────────────────────────────────────────────
    system = build_system(cfg)
    print(f"\n{'='*60}")
    print(f"  System: {system.name}")
    print(f"  State dim: {system.state_dim} | Control dim: {system.control_dim}")
    print(f"{'='*60}\n")

    # ── Generate dataset ──────────────────────────────────────────────────────
    dc = cfg["dataset"]
    print("Generating training data...")
    data = generate_dataset(
        system=system,
        n_trajectories=dc["n_trajectories"],
        t_end=dc["t_end"],
        dt=dc["dt"],
        noise_std=dc["noise_std"],
        control_mode=dc.get("control_mode", "random"),
        seed=dc["seed"],
    )
    print(f"  Total samples: {len(data['t']):,}")

    train, val, test = split_dataset(
        data,
        val_fraction=dc.get("val_fraction", 0.15),
        test_fraction=dc.get("test_fraction", 0.10),
        seed=dc["seed"],
    )
    train_norm, val_norm, test_norm, normalizer = normalize_dataset(train, val, test)

    print(f"  Train: {len(train['t']):,} | Val: {len(val['t']):,} | Test: {len(test['t']):,}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(cfg, system)
    print(f"\nModel: {model}\n")

    # ── Loss function ─────────────────────────────────────────────────────────
    tc = cfg["training"]
    loss_fn = PINNLoss(
        lambda_data=tc.get("lambda_data", 1.0),
        lambda_phys=tc.get("lambda_phys", 0.1),
        adaptive=tc.get("adaptive_weights", True),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_cfg = build_trainer_config(cfg)
    trainer = PINNTrainer(model, loss_fn, trainer_cfg)

    history = trainer.fit(
        x_train=train_norm["x"],
        u_train=train_norm["u"],
        dxdt_train=train_norm["dxdt"],
        x_val=val_norm["x"],
        u_val=val_norm["u"],
        dxdt_val=val_norm["dxdt"],
        x_bounds=(train_norm["x"].min(0), train_norm["x"].max(0)),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "pinn_model.pt")
    model.save(model_path)
    print(f"\nModel saved → {model_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_history(
        history,
        title=f"PINN Training — {system.name}",
        save_path=os.path.join(out_dir, "training_history.png"),
    )

    model.eval()
    plot_pinn_predictions(
        model,
        x_test=test_norm["x"],
        u_test=test_norm["u"],
        dxdt_test=test_norm["dxdt"],
        system=system,
        save_path=os.path.join(out_dir, "pinn_accuracy.png"),
    )

    # ── Test metrics ──────────────────────────────────────────────────────────
    import torch
    x_t = torch.FloatTensor(test_norm["x"])
    u_t = torch.FloatTensor(test_norm["u"])
    with torch.no_grad():
        dxdt_pred = model(x_t, u_t).numpy()

    from src.utils.metrics import rmse, r2_score, relative_l2
    test_rmse = rmse(dxdt_pred, test_norm["dxdt"]).mean()
    test_r2 = r2_score(dxdt_pred, test_norm["dxdt"]).mean()
    test_rel_l2 = relative_l2(dxdt_pred, test_norm["dxdt"])

    print(f"\n{'='*60}")
    print(f"  Test Set Metrics (normalised space)")
    print(f"  RMSE:      {test_rmse:.4e}")
    print(f"  R²:        {test_r2:.4f}")
    print(f"  Rel. L²:   {test_rel_l2:.4e}")
    print(f"{'='*60}\n")

    # Save metrics
    np.save(os.path.join(out_dir, "test_metrics.npy"), {
        "rmse": test_rmse,
        "r2": test_r2,
        "rel_l2": test_rel_l2,
    })

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
