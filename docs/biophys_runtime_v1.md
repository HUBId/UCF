# biophys_runtime v1

`biophys_runtime` v1 implements a deterministic, bounded global chemistry layer (no HH neuron simulation in this layer).

## Hormones (operational)
All hormone values are maintained in `[0,1]` and persisted as quantized unit values (`*_q`).

- cortisol: stress axis output
- dopamine: salience/reward proxy
- norepinephrine: arousal proxy
- serotonin: stability proxy
- acetylcholine: attention proxy

## ODE update model
The runtime uses fixed-step RK2 with deterministic config:

- `dt` fixed (default `0.1`)
- `substeps` fixed (default `2`)
- per-axis time constants `tau_*` are clamped to safe limits

General form per hormone:

- `dH/dt = (target(signals,state) - H)/tau + input_drive(signals)`

Signals are derived from:

- `risk`, `pressure`, `surprise`, `confidence`
- optional `coherence`, `instability`

Safety constraints:

- clamp every substep into `[0,1]`
- derivative cap
- parameter cap for ODE constants
- saturation tracking (`saturation_ticks`) and deterministic input dampening after repeated saturation
- NaN/Inf fallback to safe default state and degraded flag

## Volume transmission modulators
Derived modulators:

- `attention_gain` from acetylcholine (bounded, tightening only in runtime hooks)
- `plasticity_gate` from dopamine + serotonin, additionally gated by stress
- `stress_gate` from stress index

Tightening-only policy:

- high stress can disable tool planning
- extreme stress can disable LLM generation
- stress gate reduces effective LLM max token budget
- emergency path forces maximal stress chemistry and strongest tightening behavior

## Records and explain integration
Runtime appends hormone records including:

- quantized hormone axes (`cortisol_q`, `dopamine_q`, `norepinephrine_q`, `serotonin_q`, `acetylcholine_q`)
- modulators (`attention_gain_q`, `plasticity_gate_q`, `stress_gate_q`)
- hormone/modulation digests

When hormone-driven tightening is applied, audit notes are added with deterministic reason labels.

## Tuning
Tune via `HormoneCfg`:

- `tau_*`
- `dt`, `substeps`
- modulation scales

All tuning remains bounded by runtime caps to prevent runaway behavior.
