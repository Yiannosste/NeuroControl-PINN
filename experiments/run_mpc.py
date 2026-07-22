"""
Experiment 2 — Closed-Loop MPC Control.

Loads a pre-trained PINN and runs closed-loop MPC on the specified system.
Saves state/control trajectories and plots.

Usage:
    python experiments/run_mpc.py --system van_der_pol
    python experiments/run_mpc.py --system cartpole --model results/cartpole/pinn_model.pt
    python experiments/run_mpc.py --system cstr --noise 0.01
"""

import argparse
import os
import sys
import numpy as np
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems import VanDerPolSystem, CartPoleSystem, CSTRSystem
from src.models import PINN
from src.control import PINNMPC, MPCConfig
from src.utils import plot_closed_loop
from src.utils.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system(cfg: dict):
    name = cfg["system"]["name"].lower()
    if name == "vanderpol":
        return VanDerPolSystem(mu=cfg["system"].get("mu", 1.0))
    elif name == "cartpole":
        return CartPoleSystem()
    elif name == "cstr":
        return CSTRSystem()
    raise ValueError(f"Unknown system: {name}")


def build_mpc_config(cfg: dict, system) -> MPCConfig:
    mc = cfg["mpc"]
    n, m = system.state_dim, system.control_dim

    Q_diag = mc.get("Q", [1.0] * n)
    R_diag = mc.get("R", [0.01] * m)
    P_scale = mc.get("P_scale", 10.0)

    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    P = P_scale * Q

    u_min_arr = np.array(mc["u_min"]) if mc.get("u_min") else None
    u_max_arr = np.array(mc["u_max"]) if mc.get("u_max") else None

    return MPCConfig(
        horizon=mc.get("horizon", 15),
        dt=mc.get("dt", 0.05),
        integration=mc.get("integration", "rk4"),
        rollout_substeps=mc.get("rollout_substeps", 1),
        Q=Q, R=R, P=P,
        u_min=u_min_arr,
        u_max=u_max_arr,
        warm_start=mc.get("warm_start", True),
        max_iter=mc.get("max_iter", 200),
    )


def main():
    parser = argparse.ArgumentParser(description="Run PINN-MPC closed-loop control.")
    parser.add_argument("--system", type=str, default="van_der_pol",
                        choices=["van_der_pol", "cartpole", "cstr"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PINN model (.pt). "
                             "Defaults to results/<system>/pinn_model.pt")
    parser.add_argument("--noise", type=float, default=None,
                        help="Override measurement noise std.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config_map = {
        "van_der_pol": "experiments/configs/van_der_pol.yaml",
        "cartpole": "experiments/configs/cartpole.yaml",
        "cstr": "experiments/configs/cstr.yaml",
    }
    config_path = args.config or config_map[args.system]
    cfg = load_config(config_path)

    out_dir = args.output_dir or cfg.get("output_dir", f"results/{args.system}")
    os.makedirs(out_dir, exist_ok=True)

    model_path = args.model or os.path.join(out_dir, "pinn_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Run train_pinn.py first.")
        sys.exit(1)

    # ── System & Model ────────────────────────────────────────────────────────
    system = build_system(cfg)
    model = PINN.load(model_path)
    model.eval()
    print(f"\nLoaded PINN: {model}")

    # ── MPC ───────────────────────────────────────────────────────────────────
    mpc_cfg = build_mpc_config(cfg, system)
    clc = cfg["closed_loop"]
    x0 = np.array(clc["x0"])
    x_ref = np.array(clc["x_ref"])
    noise_std = args.noise if args.noise is not None else clc.get("noise_std", 0.0)
    n_steps = clc.get("n_steps", 150)

    controller = PINNMPC(model, mpc_cfg, x_ref=x_ref)

    print(f"\nRunning PINN-MPC on {system.name}...")
    print(f"  x0  = {x0}")
    print(f"  ref = {x_ref}")
    print(f"  steps = {n_steps}, noise_std = {noise_std:.4f}")

    # Seed measurement noise so a rerun of this exact command reproduces
    # the same trajectory/metrics (see closed_loop.seed in the config).
    np.random.seed(clc.get("seed", 123))

    result = controller.run_closed_loop(
        system=system,
        x0=x0,
        n_steps=n_steps,
        noise_std=noise_std,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "mpc_result.npz"),
        t=result["t"],
        x=result["x"],
        u=result["u"],
        cost=result["cost"],
        solve_time_ms=result["solve_time_ms"],
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(result, system, x_ref)
    print(f"\n{'='*55}")
    print(f"  PINN-MPC Performance on {system.name}")
    print(f"{'='*55}")
    print(f"  Tracking RMSE:       {metrics['rmse']:.4f}")
    print(f"  Settling time:       {metrics['settling_time_s']:.2f} s")
    print(f"  Control ISE:         {metrics['ise_u']:.4f}")
    print(f"  Avg solve time:      {metrics['avg_solve_time_ms']:.1f} ms")
    print(f"{'='*55}\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_closed_loop(
        result,
        system,
        x_ref=x_ref,
        title=f"PINN-MPC — {system.name}",
        save_path=os.path.join(out_dir, "mpc_closed_loop.png"),
    )

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
