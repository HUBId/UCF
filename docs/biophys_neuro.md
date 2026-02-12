# biophys_runtime v1: Neuro Micro-Engine (HH-lite / FHN-lite)

This module adds a deterministic, bounded neuron population layer on top of hormone dynamics.

## Model

We use a FitzHugh–Nagumo-lite system per neuron:

- `dv/dt = v - v^3/3 - w + I`
- `dw/dt = a*(v + b - c*w)`

with fixed RK2 integration (`dt=0.5`) and hard clamps:

- `v,w ∈ [-2,2]`
- `dv,dw ∈ [-1,1]`

Population size is fixed at `N=64`.

## Determinism and bounds

- No RNG is used.
- Per-neuron bias is deterministic from `neuron_id` hash and constant at runtime.
- Spike emission is bounded to max `32` spikes/tick.
- Overflow spikes are dropped deterministically by magnitude and neuron_id tie-breaker.
- If non-finite values are detected, state resets to defaults and degraded counter increases.

## Volume transmission (hormone -> neuron)

Hormone fields modulate excitability and gain:

- `excitability = clamp(1 - 0.6*cortisol + 0.3*drive)`
- `gain = clamp(0.5 + 0.5*drive - 0.2*cortisol)`

These factors modulate effective input current and spike threshold.

## Outputs and coupling

The neuron engine returns bounded summaries:

- `arousal` (0..1)
- `attention_gain` (0..1)
- `excitability` (0..1)
- `spike_rate` (0..1 proxy)

Runtime uses these deterministically in FEP/policy pathways:

- effective compute confidence is scaled by neuro attention gain
- exploration/attention terms receive arousal contribution
- action threshold/inhibit receives arousal/excitability deltas

No direct decision/action is emitted by the neuro engine.

## ESS persistence

A compact `NeuroRecord` is appended every 10 ticks:

- quantized summary scalars
- summary/evidence digests
- hormone link digest
- spike count + spike digest only (no full spike list)

This keeps storage bounded while allowing deterministic replay over short windows.

## Extension path

The current engine is designed to be replaced by richer HH-family variants later:

- swap `fhn_derivatives` with multi-gate HH-like conductance model
- keep fixed-step deterministic integration
- preserve bounded summaries and ESS schema for compatibility
