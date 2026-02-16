# SSM Selective-Scan v0

`runtime/ucf-compute` ships a deterministic, offline **ToySsmKernel** used as working-memory stage after SAE.

## Contract

### Input
`SsmInput` fields:
- `t: u64`
- `spikes_digest: [u8;32]`
- `spike_count: u16`
- `sae_energy: f32`
- `world_surprise: f32`
- `risk: f32`
- `seed: u64`
- `context_digest: [u8;32]`

### Output
`SsmOutput` fields:
- `pressure: f32` (0..1)
- `state_digest: [u8;32]`
- `readout_digest: [u8;32]`
- `state_norm: f32` (0..1)
- `readout: f32` (0..1)
- `quality: StageQuality`
- `notes: SmallNotes`

`SsmKernel::step` mutates only internal bounded state (`x:[f32;32]`) and returns canonical digests.

## v0 Selective scan rules

- Fixture parameters (`A`, `B`, `C`, `w1/w2/w3`, `kmax`) come from `runtime/ucf-compute/fixtures/ssm_toy_v1.json`.
- Input scalar:
  - `u = clamp(w1*(spike_count/kmax) + w2*sae_energy + w3*world_surprise, 0..1)`
- Deterministic selective set `S` is derived from `spikes_digest`, sorted ascending.
- For `i in S`:
  - `x[i] = clamp(A[i]*x[i] + B[i]*u, -1..1)`
- For non-selected indices:
  - `x[i] = clamp(0.98*x[i], -1..1)`
- Readout:
  - `r = sum_i C[i]*x[i]` in ascending `i`
  - `readout = clamp((r + 1)/2, 0..1)`

## Pressure metric

- `state_norm = clamp(mean_abs(x) / state_scale, 0..1)`
- `pressure = clamp(0.5*u + 0.5*state_norm, 0..1)`

This guarantees monotonic pressure response for higher spike/energy/surprise load via `u`.

## Budget/failure behavior

- Work is budgeted deterministically with fixed per-step unit accounting.
- On budget exceed:
  - pipeline `DegradeStages`: returns degraded marker digests and `pressure=1`.
  - pipeline `FailFast`: returns `Unavailable` compute summary.

## Persistence bounds

Only summary signals are persisted in compute outputs/evidence:
- pressure
- `ssm_digest` (`state_digest`)
- `ssm_quality`
- optional readout/state_norm in runtime signals

The internal state vector is **not persisted**.

## Telemetry

- `ucf_ssm_pressure` histogram
- `ucf_ssm_state_norm` gauge
- `ucf_ssm_degraded_total` counter
- tracing span: `ssm.step`

## Extension path

Future optimized kernels can implement the same `SsmKernel` trait while preserving:
- canonical deterministic digests
- bounded state contract
- pressure semantics for orchestration backpressure.
