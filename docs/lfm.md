# LFM Stage (Liquid Foundation Model) in UCF

The LFM stage is a **first-class compute stage** in the UCF compute pipeline, executed after world model (JEPA), SAE, and SSM, and before final risk/confidence signal consumption.

## Purpose

`ToyLfmKernel` is the default deterministic, offline, bounded liquid-dynamics stub (`toy_lfm_liquid_dynamics_v0`).

`CandleLfmKernel` (`candle_lfm_liquid_dynamics_v1`) is available behind `--features lfm-candle` and uses CPU Candle tensors for the state update path while keeping deterministic reductions/digests in Rust.

`LnnOdeLfmKernel` (`lnn_ode_lfm_rk2_v1`) is available behind `--features lfm-lnn` and integrates a nonlinear neural ODE (Option A: `dx/dt = -α ⊙ x + tanh(Wx x + Wu*u + b)`) with fixed-step RK2 midpoint.

- `liquid_state_digest`
- `liquid_readout_digest`
- `uncertainty` in `[0,1]`
- `stability` in `[0,1]`
- `state_norm` in `[0,1]`

No model weights are downloaded at runtime. No tools/IO calls are triggered by LFM. State payload vectors are not persisted.

## Contract

### Input

`LfmInput` includes:

- tick and deterministic context (`t`, `context_digest`, `world_digest`, `seed`)
- world+SAE+SSM signals (`surprise`, `spikes_digest`, `spike_count`, `sae_energy`, `pressure`)
- optional context modifiers (`coherence`, `instability`, `hormone_stress`, `neuro_arousal`)

### Output

`LfmOutput` includes:

- liquid digests (`liquid_state_digest`, `liquid_readout_digest`)
- bounded scalars (`uncertainty`, `stability`, `state_norm`)
- stage quality and bounded notes (`quality`, `notes`)

### Kernel trait

The stage integrates via `LfmKernel`:

- `name()`
- `reset_session(seed)`
- `step(input, budget)`

## Dynamics

### v0 stub (toy)

Fixture file: `runtime/ucf-compute/fixtures/lfm_params_v1.json` (committed offline fixture loaded via `include_bytes!`).

State dimension is fixed (`N=32`), with deterministic per-tick update:

1. Compute liquid drive `u` from pressure/surprise/spike_count/energy and optional terms.
2. Update selected indices (derived from `spikes_digest`) using fixture arrays `A` and `B`; non-selected indices decay.
3. Readout is deterministic `Σ C[i]*x[i]`.
4. Derive metrics:
   - `state_norm = clamp(mean_abs(x)/scale)`
   - `uncertainty = clamp(0.6*u + 0.4*state_norm)`
   - `stability = 1 - uncertainty`
5. Compute digests from canonical float bits and context.

### v1 LNN ODE (feature `lfm-lnn`)

Fixture file: `runtime/ucf-compute/fixtures/lfm_lnn_params_v1.json` (offline fixture loaded through `include_bytes!`).

- bounded state size `N <= 64` (current fixture: `N=16`)
- fixed `dt` and RK2 midpoint solver (no adaptive steps, no RNG in stepping)
- derivative clamp and state clamp enforce bounded trajectories
- deterministic matmul iteration order (`i`, then `j`)
- digest drift resistance via quantized values:
  - state entries quantized as signed unit `i16`
  - scalar signals (`u`, `uncertainty`, `stability`) quantized as unit `u16`

Uncertainty/stability are computed as:
`uncertainty = clamp(0.5*u + 0.3*state_norm + 0.2*deriv_norm)` and `stability = 1 - uncertainty`.

## Budget and failure behavior

- LFM work is bounded and metered (`lfm_units` in compute budget profile).
- `BudgetExceeded` supports two policies:
  - `FailFast`: compute returns unavailable fallback.
  - `DegradeStages`: deterministic degraded LFM output (`uncertainty=1`, `stability=0`, marker digests).

## Evidence and persistence

- Evidence chain now includes `lfm_digest` (state digest).
- Risk contract verified signals require LFM evidence.
- Backend pack metadata and ESS backend-pack records include `lfm_backend`.
- Persisted compute summary carries LFM fields (`lfm_uncertainty`, `lfm_stability`, `lfm_digest`, and `lfm_quality`).
- No internal state vector (`x`) is persisted.

## Risk/decision integration

LFM currently modulates the fused signal monotonically and bounded:

- `risk += k * lfm_uncertainty` (clamped)
- `confidence *= lfm_stability` (clamped)

This makes decisions stricter under higher liquid uncertainty while preserving deterministic replay behavior.

## Path to real LFM backends

The `LfmKernel` trait allows later adapter implementations (e.g., Candle/Burn) while keeping:

- deterministic canonical evidence wiring,
- bounded summaries/digests,
- fixed failure policy semantics.


## Backend profiles

- `toy_v1`: toy LFM kernel
- `toy_lnn_v1`: LNN ODE LFM kernel (requires `--features lfm-lnn`)
- `candle_toy_v1`: candle LFM + candle LLM (LLM falls back to stub if llm-candle is disabled)
- `candle_liquid_v1`: candle LFM only; other components remain toy/stub

Selecting an LNN profile without `lfm-lnn` returns `BackendDisabled` safely.
Selecting a candle-LFM profile without `lfm-candle` returns `BackendDisabled` safely.
