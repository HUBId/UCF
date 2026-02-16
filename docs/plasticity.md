# Liquid Plasticity v0

This repository implements **runtime-only plasticity** for the LNN/LFM kernel.

## Scope and safety model

- Plasticity never mutates fixture files.
- Base parameters remain immutable (`base_params` from fixture).
- Adaptation applies only to an in-memory runtime overlay (`alpha`, `wu`).
- Updates are deterministic and bounded.

## Governance gates

Plasticity is enabled only when all conditions hold:

- `governor_tier <= 1`
- `uncertainty <= 0.45`
- `coherence >= 0.6` (if present)

Otherwise, plasticity is disabled for the tick.

## Deterministic update rule

Per tick, a deterministic scalar direction is computed from internal runtime scalars:

- `prediction_error` / `surprise`
- `pressure`

A fixed learning rate and clamps are used:

- fixed `lr = 0.01`
- per-tick delta clamped to `[-0.02, 0.02]`
- deltas quantized to fixed resolution (`1e-3`) and persisted as `i16`
- max updated params per tick: `4` (hard cap overall deltas: `8`)
- alpha bounds enforced in `[0.1, 2.0]`
- additional session drift clamp to `base_alpha ± 0.5`

## Audit and replay

Each LFM step emits a bounded `PlasticityRecord` attached to compute signals:

- gate inputs (quantized)
- enabled flag
- quantized param deltas
- `delta_digest`
- `params_digest_after`

Digest evolution is deterministic for identical inputs and seeds, enabling replay checks.

## Telemetry

Added metrics:

- `ucf_plasticity_enabled_total`
- `ucf_plasticity_disabled_total{reason=...}`
- `ucf_lfm_alpha_mean`
- `ucf_plasticity_param_updates_total`
