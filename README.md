# NeuroControl-PINN

### Physics-Informed Neural Network Surrogate Modeling for Nonlinear Model Predictive Control

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/SciML-PINN%20%2B%20MPC-8A2BE2" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## Overview

**NeuroControl-PINN** combines two ideas from modern engineering:

- **Physics-Informed Neural Networks (PINNs).** Deep learning models that put physical laws directly into the training loss through automatic differentiation. This makes system identification need less data and keeps the learned model physically consistent.

- **Model Predictive Control (MPC).** An optimisation-based control strategy. At every timestep it solves a finite-horizon optimal control problem using a forward model of the dynamics.

The core idea is simple: **replace the expensive or unavailable analytical model inside MPC with a PINN surrogate trained on noisy trajectory data.** The physics residuals act as regularisers, so the surrogate stays physically consistent even where there is little data.

```
    Noisy Trajectories
    {(x_i, u_i, ẋ_i)}         ┌─────────────────────────┐
          │                    │      PINN Training       │
          ▼                    │  L = λ_d·L_data          │
    ┌──────────┐               │      + λ_p·L_physics     │
    │  PINN    │◄──────────────│  (Adam + L-BFGS)         │
    │ f̂(x, u) │               └─────────────────────────┘
    └──────────┘
          │
          ▼
    ┌──────────────────────────────────────────────┐
    │          PINN-MPC (Receding Horizon)          │
    │                                              │
    │  min Σ [e_t'Qe_t + u_t'Ru_t] + e_N'Pe_N     │
    │  s.t. x_{t+1} = x_t + dt·f̂(x_t, u_t)       │
    │       u_min ≤ u_t ≤ u_max                   │
    └──────────────────────────────────────────────┘
          │
          ▼
    Closed-Loop Control  →  Benchmark vs Classical MPC + PID
```

---

## Benchmark Systems

| System | Dim | Nonlinearity | Real-World Application |
|--------|-----|-------------|----------------------|
| **Van der Pol Oscillator** | 2 | Limit cycle, cubic damping | Power electronics, cardiac rhythms |
| **Inverted Pendulum (Cart-Pole)** | 4 | Trigonometric, unstable | Robotics, aerospace attitude control |
| **CSTR Exothermic Reactor** | 2 | Arrhenius exponential, bistability | Chemical process control |

---

## Key Features

### PINN Architecture
- Flexible MLP with **residual skip connections** and **Xavier initialisation**. When the layer width changes, a linear projection is inserted so the skip connection still fits.
- Optional **Fourier Feature Embedding** to reduce spectral bias ([Tancik et al., 2020](https://arxiv.org/abs/2006.10739))
- **Learnable output scaling** so predictions respect the normalisation
- Deep-ensemble module for uncertainty quantification (`src/models/ensemble.py`). It is implemented and unit-tested but **not yet connected to the MPC loop**. See [Future Work](#future-work-achieving-real-time-inference--50ms).

### Physics-Informed Training
- **Composite loss** `L = λ_d·L_data + λ_p·L_physics` with adaptive weight balancing inspired by NTK analysis ([Wang et al., 2021](https://epubs.siam.org/doi/10.1137/20M1318043))
- **Two-phase optimiser**: Adam for fast convergence, then L-BFGS for fine-tuning. This is the standard PINN recipe from [Raissi et al., 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125).
- **Randomised collocation resampling** every epoch in state-control space (uniform sampling, not importance or adaptive sampling)
- The physics residual enforces only the **exact kinematic identities** that hold no matter what the unknown physical parameters are (for example "velocity is the derivative of position"). It does not enforce the full, uncertain governing equations. See [Method Details](#method-details).
- Early stopping with best-model checkpointing

### PINN-MPC
- **Single-shooting MPC** with SLSQP optimisation (scipy)
- Gradients computed through **PyTorch autograd** (exact, not finite differences)
- **RK4 integration** of the PINN dynamics over the prediction horizon
- **Warm starting** from the shifted previous solution
- Solve time logging and diagnostics

---

## Project Structure

```
NeuroControl-PINN/
│
├── src/
│   ├── systems/                  # Dynamical system simulators
│   │   ├── base.py               # Abstract DynamicalSystem (simulate, linearize)
│   │   ├── van_der_pol.py        # Van der Pol oscillator
│   │   ├── cartpole.py           # Inverted pendulum on cart
│   │   └── cstr.py               # Exothermic CSTR reactor
│   │
│   ├── models/                   # Neural network models
│   │   ├── pinn.py               # PINN (MLP + Fourier features + residual blocks)
│   │   └── ensemble.py           # Deep ensemble for uncertainty quantification
│   │
│   ├── training/                 # Training pipeline
│   │   ├── losses.py             # DataLoss, PhysicsResidualLoss, PINNLoss
│   │   ├── trainer.py            # PINNTrainer (Adam → L-BFGS)
│   │   └── physics.py            # Per-system physics_fn (kinematic-identity residuals)
│   │
│   ├── control/                  # Controllers
│   │   ├── pinn_mpc.py           # PINN-MPC (SLSQP + autograd gradients)
│   │   ├── classical_mpc.py      # Classical MPC (oracle / ground truth)
│   │   └── pid.py                # PID baseline with anti-windup
│   │
│   └── utils/                    # Utilities
│       ├── data_generation.py    # Dataset generation and normalisation
│       ├── metrics.py            # RMSE, R², tracking error, control effort
│       └── visualization.py      # Phase portraits, training curves, benchmarks
│
├── experiments/
│   ├── configs/                  # YAML configs for each system
│   │   ├── van_der_pol.yaml
│   │   ├── cartpole.yaml
│   │   └── cstr.yaml
│   ├── train_pinn.py             # Experiment 1: PINN training
│   ├── run_mpc.py                # Experiment 2: Closed-loop MPC
│   └── benchmark.py              # Experiment 3: Controller comparison
│
├── notebooks/
│   ├── 01_system_analysis.ipynb  # System dynamics & phase portraits
│   ├── 02_pinn_training.ipynb    # Training, ablation study
│   └── 03_mpc_control.ipynb      # MPC comparison & metrics
│
├── tests/
│   ├── test_systems.py
│   ├── test_models.py
│   ├── test_control.py
│   └── test_physics.py
│
├── docs/
│   └── methodology.md            # Full technical writeup
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/Yiannosste/NeuroControl-PINN.git
cd NeuroControl-PINN
pip install -r requirements.txt
```

### 1. Train a PINN

```bash
# Train on Van der Pol oscillator (default)
python experiments/train_pinn.py --system van_der_pol

# Train on cart-pole
python experiments/train_pinn.py --system cartpole

# Train on CSTR reactor
python experiments/train_pinn.py --system cstr

# With GPU acceleration
python experiments/train_pinn.py --system cartpole --device cuda
```

### 2. Run PINN-MPC

```bash
# Run closed-loop PINN-MPC (requires trained model)
python experiments/run_mpc.py --system van_der_pol

# With measurement noise
python experiments/run_mpc.py --system cartpole --noise 0.01
```

### 3. Benchmark All Controllers

```bash
# Compare PINN-MPC vs Classical MPC vs PID
python experiments/benchmark.py --system van_der_pol
```

### 4. Interactive Notebooks

```bash
jupyter lab notebooks/
```

Open the notebooks in order: `01_system_analysis`, then `02_pinn_training`, then `03_mpc_control`.

---

## Method Details

### Physics-Informed Loss

For a system `ẋ = f(x, u)`, the PINN is trained with:

$$
\mathcal{L} = \underbrace{\frac{\lambda_d}{N} \sum_{i=1}^{N} \|\hat{f}(\mathbf{x}_i, \mathbf{u}_i) - \dot{\mathbf{x}}_i\|^2}_{\text{Data loss}} + \underbrace{\frac{\lambda_p}{N_c} \sum_{j=1}^{N_c} \|\mathcal{R}(\mathbf{x}_j^c, \mathbf{u}_j^c)\|^2}_{\text{Physics residual}}
$$

`R` enforces only the part of the dynamics that is *exactly* known, whatever the uncertain physical parameters are. For Van der Pol this is `f̂₁(x, u) − x₂ = 0` (the derivative of position is velocity). The nonlinear, parameter-dependent term `μ(1−x₁²)x₂ − x₁ + u` is left for the network to learn from data. Cart-Pole gets the same kind of kinematic identities, one pair of them. The CSTR has no such identity between its two states (concentration and temperature), so it trains fully data-driven. See `src/training/physics.py` and [`docs/methodology.md`](docs/methodology.md#3-physics-informed-loss) for the full derivation, including how the residual handles the normalised training space.

The weights `λ_d`, `λ_p` are **rebalanced every 100 steps** so that both loss terms produce gradients of similar size:

$$
\lambda_k \leftarrow \alpha \lambda_k + (1 - \alpha) \frac{\bar{\sigma}}{\|\nabla_\theta \mathcal{L}_k\|_2}
$$

### MPC Formulation

At each timestep, PINN-MPC solves a nonlinear programme with a quadratic cost:

$$\min_{\mathbf{u}_{0:N-1}} \sum_{t=0}^{N-1} \left[ \mathbf{e}_t^\top Q \mathbf{e}_t + \mathbf{u}_t^\top R \mathbf{u}_t \right] + \mathbf{e}_N^\top P \mathbf{e}_N$$

subject to `x_{t+1} = x_t + Δt · f̂(x_t, u_t)` (RK4).

The gradients `∂J/∂u` come from PyTorch autograd through the unrolled PINN rollout. They are exact, which makes the SLSQP optimisation efficient.

---

## Benchmark Results

### Van der Pol Stabilisation (x₀ = [2.0, 0.0] → origin, 200 steps, Δt = 0.05 s, measurement noise σ = 0.02)

| Controller | RMSE | Settling (s) | ISE_u | Avg Solve (ms) |
|---|---|---|---|---|
| Classical MPC | 0.6956 | 2.95 | 15.77 | ~250–435 |
| PINN-MPC | 0.7054 | 2.65 | 19.31 | ~1800–2300 |
| PID | 1.0099 | 9.85 | 75.37 | 0.0 |

Reproduce with `python experiments/benchmark.py --system van_der_pol`. RMSE, settling time and ISE_u are exactly the same on every run, because the closed-loop measurement noise is seeded through `closed_loop.seed` in `experiments/configs/van_der_pol.yaml`. Solve time is wall-clock, so it changes with machine load. That is why it is given as a range.

### Cart-Pole and CSTR

Both systems train and run end-to-end through the full pipeline (`train_pinn.py`, then `run_mpc.py` or `benchmark.py`). They use the same physics-informed training and the same reproducibility setup described above. Cart-Pole gets its own kinematic-identity physics loss (test R² = 0.9998). Both were verified with real, seeded closed-loop runs of all three controllers (Classical MPC, PINN-MPC, PID). All metrics were finite and no solver failed. Cart-Pole's results were confirmed identical on a repeat run. CSTR's PINN-MPC solve is slow (tens of minutes per benchmark), so a repeat was not run, though it uses the same seeding that was verified elsewhere. These two systems confirm that the pipeline works beyond Van der Pol. They are not a second headline benchmark.

### Effect of the Physics Loss (ablation, `notebooks/02_pinn_training.ipynb`)

| Variant | λ_phys | Test RMSE (normalised) | R² |
|---|---|---|---|
| No Physics | 0.00 | 0.0566 | 0.9961 |
| Low Physics | 0.05 | 0.0506 | 0.9970 |
| High Physics | 1.00 | 0.0460 | 0.9976 |

Raising the physics-loss weight steadily improves accuracy on held-out data (about 19% lower RMSE from no physics to the strongest constraint). This happens even though the enforced relation (`ẋ₁ = x₂`) is only part of the dynamics. Enforcing it still regularises the shared hidden representation, which helps the harder, unconstrained output too.

---

## Future Work: Achieving Real-Time Inference (< 50ms)

The current implementation shows that the method is scientifically correct. Getting it ready for production needs the engineering steps below.

**Step 1: Model Export (removing the Python GIL)**

Export the PyTorch model with `TorchScript` (`torch.jit.script`) or `ONNX` (`torch.onnx.export`). This separates the network from the Python interpreter and removes the cost of crossing the C++/Python boundary. That crossing happens thousands of times during the solver iterations of a single MPC horizon rollout.

```python
# Example TorchScript export
scripted = torch.jit.script(trained_pinn)
scripted.save("pinn_surrogate.pt")
```

**Step 2: Industrial C++ Solvers**

Replace `scipy.optimize` with a compiled C/C++ solver built for nonlinear MPC. The main options are:

- [**Acados**](https://github.com/acados/acados): a real-time embedded NLP solver (RTI/SQP) with code generation. It reaches sub-millisecond solve times on embedded hardware.
- [**CasADi**](https://web.casadi.org/): a symbolic AD framework that generates efficient C code for the OCP, with IPOPT or qpOASES as the inner QP solver.

Both can take a C function exported from the ONNX/TorchScript model as the dynamics callback.

**Step 3: C++ Control Loop and Memory Management**

Rewrite the main MPC loop in C++. Pre-allocate the state and control trajectory buffers (the `(N+1) × n` state buffer and the `N × m` control buffer) so there is no dynamic memory allocation or garbage collection during the horizon rollout. In Python, even creating a simple tensor inside the solve loop adds tens of milliseconds of GC pressure per call.

**Step 4: Edge Hardware Acceleration**

Deploy the compiled ONNX/C++ stack on an embedded edge device with a tensor accelerator, for example the **Nvidia Jetson Orin** (up to 275 TOPS INT8) or a dedicated NPU. The dense matrix multiplications in the PINN forward pass parallelise easily on this hardware. Per-evaluation latency drops from about 1 ms to under 100 µs, and the per-step solve time drops from seconds to milliseconds.

---

## Scientific Background

This project sits at the intersection of:

1. **Physics-Informed Machine Learning.** Putting physical structure into neural networks for better generalisation and data efficiency.

2. **Data-Driven Model Predictive Control.** Learning system dynamics from data so that model-based control works without an explicit analytical model.

3. **Scientific Machine Learning (SciML).** Combining the expressiveness of deep learning with the inductive biases of physical laws.

### Core References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.* Journal of Computational Physics, 378, 686–707.

- Karniadakis, G.E., et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics, 3(6), 422–440.

- Wang, S., Teng, Y., & Perdikaris, P. (2021). *Understanding and mitigating gradient flow pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing, 43(5).

- Camacho, E.F., & Bordons, C. (2007). *Model Predictive Control.* Springer.

- Lakshminarayanan, B., et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
jupyter>=1.0.0
pytest>=7.0.0
```

---

## License

MIT License. See `LICENSE` for details.

---

*Built as part of a Scientific Machine Learning (SciML) portfolio project.*
