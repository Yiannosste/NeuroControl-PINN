# Methodology

## NeuroControl-PINN: Technical Overview

### 1. Problem Statement

Take a nonlinear, continuous-time dynamical system:

$$
\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}), \qquad \mathbf{x} \in \mathbb{R}^n, \qquad \mathbf{u} \in \mathbb{R}^m
$$

We observe **noisy trajectory data**:

$$
\mathcal{D} = \lbrace (\mathbf{x}_i, \mathbf{u}_i, \dot{\mathbf{x}}_i)\rbrace_{i=1}^N
$$

Each `x_i` is a sensor reading with additive Gaussian noise, `x_i = x_i^true + ε_i` with `ε_i ~ N(0, σ²I)`. The derivative target `ẋ_i` is computed analytically along the clean trajectory during data generation. This matches a common experimental setup: noisy state measurements, with derivatives coming from a higher-fidelity process (such as a state estimator) rather than raw finite differences.

We also assume **partial physics knowledge**. Not the full governing equations, but individual structural facts that hold regardless of any uncertain physical parameter (for example "velocity is the derivative of position").

**Goal**: learn a surrogate `f̂_θ(x, u) ≈ f(x, u)` that is:

1. Accurate on the training data
2. Consistent with the known partial physics
3. Usable as a forward model inside MPC

---

### 2. PINN Architecture

The PINN is a fully connected network:

$$
\hat{f}_{\theta}: \mathbb{R}^{n+m} \rightarrow \mathbb{R}^n
$$

with:

- **Input**: `[x, u]` (state + control, normalised)
- **Hidden layers**: a configurable stack of widths, tanh activations
- **Optional**: Fourier feature embedding, residual connections
- **Output**: `ẋ` prediction (normalised)

#### Residual Block

$$
h_{l+1} = \sigma\big(h_l + W_2\, \sigma(W_1 h_l + b_1) + b_2\big)
$$

This lets gradients flow through deep networks and keeps feature information. When consecutive layer widths differ (for example a `[128, 128, 64]` taper), a linear projection is inserted between blocks so the residual add-back always has the right shape.

#### Fourier Feature Embedding

$$
\phi(\mathbf{z}) = [\sin(\mathbf{Bz}),\ \cos(\mathbf{Bz})], \qquad B_{ij} \sim \mathcal{N}(0, \sigma_B^2)
$$

This reduces spectral bias, so the network can learn the high-frequency parts of the dynamics.

---

### 3. Physics-Informed Loss

$$
\mathcal{L}(\theta) = \lambda_d\, \mathcal{L}_{\text{data}} + \lambda_p\, \mathcal{L}_{\text{physics}}
$$

#### Data Loss

$$
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^N \big\| \hat{f}_{\theta}(\mathbf{x}_i, \mathbf{u}_i) - \dot{\mathbf{x}}_i \big\|^2
$$

#### Physics Residual Loss

For a known structural identity `g(x, u, f̂_θ) = 0`:

$$
\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \big\| g(\mathbf{x}_j^c, \mathbf{u}_j^c, \hat{f}_{\theta}) \big\|^2
$$

where `{(x_j^c, u_j^c)}` are **collocation points**, resampled uniformly at random in state-control space every training epoch (`src/training/trainer.py::_sample_collocation`).

**What is actually enforced.** We only constrain the relation that is *exactly* true regardless of unknown physical parameters. Enforcing anything parameter-dependent would just hand the network the answer it is supposed to learn from data.

| System | Constrained identity | Left to learn from data |
|---|---|---|
| Van der Pol | `f̂_θ,1(x, u) − x₂ = 0` (position's derivative is velocity) | The nonlinear restoring/damping term `μ(1−x₁²)x₂ − x₁ + u` |
| Cart-Pole | `f̂_θ,1 − ṗ = 0` and `f̂_θ,3 − θ̇ = 0` (both kinematic identities) | The coupled cart/pole torque-balance dynamics |
| CSTR | *(none, see below)* | Both `Ċ_A` and `Ṫ` in full |

The CSTR has no algebraic identity between its two states. Concentration and temperature are physically independent quantities with no position/velocity-style relationship. So it trains in pure data-driven mode and `lambda_phys` does nothing for this system.

Implementation detail: training happens in normalised (zero-mean, unit-variance) space, but the identity above is only exact in physical units. `src/training/physics.py` closes over the fitted normaliser. The residual unscales the normalised state, applies the identity in physical units, then rescales back before comparing against the model's normalised output.

#### Adaptive Weight Balancing

Inspired by Neural Tangent Kernel analysis:

$$
\lambda_k^{(t+1)} = \alpha\, \lambda_k^{(t)} + (1-\alpha) \frac{\bar{\sigma}}{\|\nabla_{\theta} \mathcal{L}_k\|_2}
$$

This keeps the gradient magnitudes of both loss terms comparable, so neither term takes over during training.

---

### 4. Two-Phase Training Strategy

**Phase 1: Adam** (stochastic gradient descent)

- Mini-batch SGD with batch size 512–1024
- Learning rate scheduling: step decay every `N_s` epochs
- Fast exploration of the loss landscape
- Typically 4000–8000 epochs

**Phase 2: L-BFGS** (quasi-Newton)

- Full-batch L-BFGS with Strong Wolfe line search
- Uses second-order curvature information for fast convergence
- Typically 200–500 epochs
- Reaches a much lower final loss than Adam alone

This is the standard approach in the PINN literature for smooth, low-noise objectives.

---

### 5. PINN-MPC Formulation

At each time `k`, solve the finite-horizon OCP:

$$
\min_{\mathbf{U}} J(\mathbf{U}) = \sum_{t=0}^{N-1} \Big[ \mathbf{e}_t^\top Q\, \mathbf{e}_t + \mathbf{u}_t^\top R\, \mathbf{u}_t \Big] + \mathbf{e}_N^\top P\, \mathbf{e}_N
$$

where `e_t = x_t − x_ref` and `U = [u_0, …, u_{N-1}]`.

Subject to:

- **PINN dynamics**: `x_{t+1} = x_t + Δt · f̂_θ(x_t, u_t)` (Euler or RK4)
- **Control bounds**: `u_min ≤ u_t ≤ u_max`

**Gradient computation.** Gradients of `J` with respect to `U` come from automatic differentiation through the unrolled PINN rollout. They are exact and cost one forward pass per SLSQP function evaluation, instead of the many passes finite differences would need.

**Warm starting**: the shifted previous solution

$$
\mathbf{U}^{(k+1)}_0 = [\mathbf{u}_1^{\ast}, \ldots, \mathbf{u}_{N-1}^{\ast}, \mathbf{u}_{N-1}^{\ast}]
$$

initialises the next MPC solve. This reduces the number of SLSQP iterations.

---

### 6. Uncertainty Quantification (Ensemble)

An ensemble of `M` independently trained PINNs:

$$
\bar{f}(\mathbf{x}, \mathbf{u}) = \frac{1}{M} \sum_{m=1}^M \hat{f}^{(m)}(\mathbf{x}, \mathbf{u})
$$

$$
\sigma^2(\mathbf{x}, \mathbf{u}) = \frac{1}{M-1} \sum_{m=1}^M \Big(\hat{f}^{(m)} - \bar{f}\Big)^2
$$

A high `σ²` signals an out-of-distribution state. The ensemble (`src/models/ensemble.py`) is implemented and unit-tested but is **not currently connected to the MPC controllers**. Using it for uncertainty-aware or robust MPC is future work, not a shipped feature.

---

### 7. Benchmarking Protocol

Controllers are compared under identical conditions:

- Same initial state `x_0`
- Same reference `x_ref`
- Same measurement-noise realisation `ε` (the RNG is seeded identically before each controller's closed-loop run, see `closed_loop.seed` in the experiment configs)
- Same cost matrices `Q`, `R`, `P`
- Same MPC horizon `N` and step `Δt`

**Metrics**:

- **RMSE**: `sqrt( (1/T) · Σ_{k=0}^T ‖x_k − x_ref‖² )`, over the full closed-loop trajectory (transient included)
- **Settling time**: first time `‖x_k − x_ref‖ / ‖x_0 − x_ref‖ < 0.05`, and stays below that threshold for the rest of the run
- **Control ISE**: `Σ_k ‖u_k‖² · Δt` (control energy)
- **Solve time**: wall-clock time per MPC solve (ms)

**Benchmark: Van der Pol Stabilisation** (`x_0 = [2.0, 0.0] →` origin, 200 steps, `Δt = 0.05` s, measurement noise `σ = 0.02`, seed `123`)

| Controller | RMSE | Settling (s) | ISE_u | Avg Solve (ms) |
|---|---|---|---|---|
| Classical MPC | 0.6956 | 2.95 | 15.77 | ~250–435 |
| PINN-MPC | 0.7054 | 2.65 | 19.31 | ~1800–2300 |
| PID | 1.0099 | 9.85 | 75.37 | 0.0 |

RMSE is computed over the *entire* trajectory, including the initial transient from `x_0=[2,0]` down to the reference. With a settling time around 3 s out of a 10 s run, the transient dominates the RMSE even for controllers that track the reference almost perfectly once settled. So a low settling time does not mean a near-zero RMSE. Read the two metrics together.

**On reproducibility.** RMSE, settling time and ISE_u are exactly the same on every rerun of `python experiments/benchmark.py --system van_der_pol`, because the closed-loop measurement noise is seeded (`closed_loop.seed` in the YAML config) identically before every controller's run. Solve time is wall-clock and not seeded. It varies with machine load, hence the range above.

An earlier draft of this document reported Classical MPC RMSE = 0.0125 next to the same settling time (3.15 s) and ISE_u (15.37) shown above. That RMSE was never reproducible from this codebase. A settling time of about 3 s over a 10 s trajectory starting at `‖x_0 − x_ref‖ = 2` cannot give an RMSE as low as 0.0125, because the transient alone forces a much larger value. Repeated, independent reruns of the fixed code (the `dt=0` bug removed) consistently give RMSE ≈ 0.70, which is the figure shown now. The conclusion is the same in spirit (Classical MPC and PINN-MPC track each other closely), but the number has been corrected to what the code actually measures.

**Cart-Pole and CSTR.** Both systems were also trained and run end-to-end through the same pipeline (`python experiments/train_pinn.py --system {cartpole,cstr}` then `benchmark.py`). They use the same physics-informed training, MPC formulation and seeded-reproducibility setup described above. Cart-Pole gets its own kinematic-identity physics loss and reaches test R² = 0.9998. Both produced finite closed-loop metrics for all three controllers with no solver failures. Cart-Pole's results were confirmed identical on a repeat run. CSTR's PINN-MPC solve is very slow (tens of minutes per benchmark), so a repeat run was not done. It uses the same seeding mechanism already verified on the other two systems, but was not independently re-confirmed.
