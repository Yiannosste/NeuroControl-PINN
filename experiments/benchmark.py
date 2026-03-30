"""
Experiment 3 — Controller Benchmark Comparison.

Runs PINN-MPC, Classical MPC (oracle), and PID on the same closed-loop
problem and produces a comprehensive comparison table + figure.

Usage:
    python experiments/benchmark.py --system van_der_pol
    python experiments/benchmark.py --system cartpole
    python experiments/benchmark.py --system cstr
"""

import argparse
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.systems import VanDerPolSystem, CartPoleSystem, CSTRSystem
from src.models import PINN
from src.control import PINNMPC, ClassicalMPC, PIDController, MPCConfig
from src.utils import plot_benchmark_comparison
from src.utils.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_system(cfg: dict):
    name = cfg["system"]["name"].lower()
    if name == "vanderpol":
        return VanDerPolSystem(mu=cfg["system"].get("mu", 1.0))
    elif name == "cartpole":
        return CartPoleSystem()
    elif name == "cstr":
        return CSTRSystem()
    raise ValueError(name)


def build_mpc_config(cfg: dict, system) -> MPCConfig:
    mc = cfg["mpc"]
    n, m = system.state_dim, system.control_dim
    Q_diag = mc.get("Q", [1.0] * n)
    R_diag = mc.get("R", [0.01] * m)
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    P = mc.get("P_scale", 10.0) * Q
    u_min = np.array(mc["u_min"]) if mc.get("u_min") else None
    u_max = np.array(mc["u_max"]) if mc.get("u_max") else None
    return MPCConfig(
        horizon=mc.get("horizon", 15),
        dt=mc.get("dt", 0.05),
        integration=mc.get("integration", "rk4"),
        Q=Q, R=R, P=P,
        u_min=u_min, u_max=u_max,
        warm_start=mc.get("warm_start", True),
        max_iter=mc.get("max_iter", 150),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="van_der_pol",
                        choices=["van_der_pol", "cartpole", "cstr"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-pid", action="store_true")
    args = parser.parse_args()

    config_map = {
        "van_der_pol": "experiments/configs/van_der_pol.yaml",
        "cartpole": "experiments/configs/cartpole.yaml",
        "cstr": "experiments/configs/cstr.yaml",
    }
    cfg = load_config(args.config or config_map[args.system])
    out_dir = args.output_dir or cfg.get("output_dir", f"results/{args.system}")
    os.makedirs(out_dir, exist_ok=True)

    system = build_system(cfg)
    mpc_cfg = build_mpc_config(cfg, system)
    clc = cfg["closed_loop"]
    x0 = np.array(clc["x0"])
    x_ref = np.array(clc["x_ref"])
    n_steps = clc.get("n_steps", 150)
    noise_std = clc.get("noise_std", 0.0)

    results = []

    # ── 1. Classical MPC (oracle) ─────────────────────────────────────────────
    print("\n[1/3] Classical MPC (perfect model)...")
    classical = ClassicalMPC(system, mpc_cfg, x_ref=x_ref)
    res_classical = classical.run_closed_loop(system, x0, n_steps, noise_std)
    results.append(res_classical)

    # ── 2. PINN-MPC ───────────────────────────────────────────────────────────
    model_path = args.model or os.path.join(out_dir, "pinn_model.pt")
    if os.path.exists(model_path):
        print("\n[2/3] PINN-MPC...")
        model = PINN.load(model_path)
        model.eval()
        pinn_mpc = PINNMPC(model, mpc_cfg, x_ref=x_ref)
        res_pinn = pinn_mpc.run_closed_loop(system, x0, n_steps, noise_std)
        results.append(res_pinn)
    else:
        print(f"\n[2/3] Skipping PINN-MPC — model not found at {model_path}")

    # ── 3. PID ────────────────────────────────────────────────────────────────
    if not args.skip_pid:
        pid_cfg = cfg.get("pid", {})
        pid = PIDController(
            Kp=pid_cfg.get("Kp", 10.0),
            Ki=pid_cfg.get("Ki", 1.0),
            Kd=pid_cfg.get("Kd", 2.0),
            dt=mpc_cfg.dt,
            u_min=float(mpc_cfg.u_min[0]) if mpc_cfg.u_min is not None else None,
            u_max=float(mpc_cfg.u_max[0]) if mpc_cfg.u_max is not None else None,
            state_idx=pid_cfg.get("state_idx", 0),
        )
        x_ref_traj = np.tile(x_ref, (n_steps, 1))
        print("\n[3/3] PID...")
        res_pid = pid.run_closed_loop(system, x0, n_steps, noise_std, x_ref_traj)
        results.append(res_pid)

    # ── Metrics table ─────────────────────────────────────────────────────────
    all_metrics = [compute_metrics(r, system, x_ref) for r in results]

    col_w = 20
    headers = ["Controller", "RMSE", "Settle (s)", "ISE_u", "Solve (ms)"]
    print(f"\n{'='*80}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*80}")
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "-" * (col_w * len(headers) + len(headers) * 2))
    for m in all_metrics:
        row = [
            m["controller"],
            f"{m['rmse']:.4f}",
            f"{m['settling_time_s']:.2f}",
            f"{m['ise_u']:.2f}",
            f"{m['avg_solve_time_ms']:.1f}",
        ]
        print("  " + "  ".join(v.ljust(col_w) for v in row))
    print(f"{'='*80}\n")

    # ── Save metrics ──────────────────────────────────────────────────────────
    import json
    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_benchmark_comparison(
        results,
        system,
        x_ref=x_ref,
        title=f"Controller Benchmark — {system.name}",
        save_path=os.path.join(out_dir, "benchmark_comparison.png"),
    )

    print(f"Benchmark results saved to: {out_dir}")


if __name__ == "__main__":
    main()
