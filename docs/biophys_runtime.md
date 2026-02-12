# biophys_runtime v0

`biophys_runtime v0` introduces a deterministic, offline hormone subsystem that couples homeostasis to FEP/policy gating without creating any new action/tool paths.

## Model equations (simplified HPA)

State (`0..1` bounded):
- `CRH`
- `ACTH`
- `Cortisol`
- `Drive`

Input stress drive:

- `S = clamp(0.6*pressure + 0.4*surprise + 0.5*risk - 0.3*confidence - 0.1*coherence + 0.2*instability, 0..1)`

Dynamics:

- `dCRH/dt = k1*S - k2*CRH - k_feedback*Cortisol*CRH`
- `dACTH/dt = k3*CRH - k4*ACTH`
- `dCortisol/dt = k5*ACTH - k6*Cortisol`
- `dDrive/dt = drive_recovery*(target_drive - drive) - drive_stress_coupling*S*drive`

where `target_drive = clamp((1 - Cortisol) * (1 - 0.25*S), 0..1)`.

## Determinism and bounds

- Fixed-step RK2 midpoint with `dt=1.0` tick.
- No RNG, no variable integration step.
- Derivative hard-cap `|dx| <= 0.2` per tick.
- Post-step clamping to `0..1` for every state component.
- NaN/Inf safety fallback resets to default safe state and increments degraded counter.

## Homeostasis/FEP/policy coupling

The hormone system only modulates internal scoring/gating parameters:

- `risk_penalty_scale = 1.0 + 1.5*stress_index`
- `action_threshold_delta = +0.3*stress_index`
- `exploration_bias_delta = -0.2*stress_index`
- `attention_gain = 1.0 + 0.5*drive`

These are injected into FEP scoring and inhibit thresholds; no direct action execution path is added.

## ESS persistence and replay

A bounded `HormoneRecord` is persisted every 10 ticks (windowed to avoid ESS bloat):

- `t`
- quantized `cortisol/drive/stress_index`
- `hormone_digest`
- `evidence_chain_digest`
- optional `modulation_digest`
- `schema_version`

Replayability is verified by deterministic digest equality over repeated runs with identical input signal sequences.

## Telemetry

- Gauges:
  - `ucf_hormone_cortisol`
  - `ucf_hormone_drive`
  - `ucf_hormone_stress_index`
- Counter:
  - `ucf_hormone_degraded_total`
- Trace span:
  - `biophys_runtime.step`

## Future extension path (HH/BlueBrain)

Interfaces are kept scalar and additive (`HormoneInput`, `HormoneStateSummary`, `GatingModulation`, `HormoneRecord`).
This allows swapping internals for richer ODE/compartmental or BlueBrain-aligned models later without breaking external orchestrator and ESS contracts.
