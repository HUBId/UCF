# SAE v1.1 (Real) Contract Notes

## Unchanged external contract

SAE v1.1 keeps the external contract stable:

- output remains bounded top-k spikes (`feature_id`, quantized magnitude)
- `spikes_digest` is preserved as the integrity binding for downstream stages
- max emitted spikes per tick remains capped by `K <= 64` (runtime currently uses `SAE_TOP_K=32`)

## SAE architecture

Runtime-compatible SAE v1.1 shape:

- encoder: `z = ReLU(W_enc * x + b_enc)`
- sparsify: deterministic top-k over `z`
- optional decoder tensors may exist for shadow diagnostics:
  - `x_hat = W_dec * z + b_dec`

## WeightSpec tensors

For SAE slot (`ModelSlot::Sae`) the canonical tensors are:

- required
  - `sae.w_enc` with shape `[F, D]`, dtype `f32`
  - `sae.b_enc` with shape `[F]`, dtype `f32`
- optional
  - `sae.w_dec` with shape `[D, F]`, dtype `f32`
  - `sae.b_dec` with shape `[D]`, dtype `f32`

These names are enforced by `spec_for_slot(ModelSlot::Sae, ..)`.

## Determinism and tie-breaks

Top-k ordering requirements:

1. rank by magnitude descending
2. on ties: lower `feature_id` wins
3. emitted spike list must be stable and sorted by `feature_id` for canonicalization

This ensures stable `spikes_digest` across backend implementations.

## Safety and rollout

- SAE spikes are signal-only inputs for routing/attention and must not directly authorize tool escalation.
- rollout mode remains environment-governed via `UCF_SLOT_SAE_MODE=toy|shadow|active`.
- recommended sequence: shadow first, then active only after envelope checks pass (latency, rate stability, and drift alarms).

## Fixtures

Offline fixtures for SAE v1.1 are provided in `fixtures/weights/`:

- `sae_v1_small.safetensors.hex`
- `sae_v1_missing_tensor.safetensors.hex`
- `sae_v1_bad_shape.safetensors.hex`
