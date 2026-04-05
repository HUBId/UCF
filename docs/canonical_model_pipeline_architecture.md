# Canonical Model Pipeline Architecture (runtime/ucf-compute)

Status: active for Phase-A follow-up hardening.

## Canonical entrypoint

The canonical runtime model path is `ComputePipelineBackend::compute_canonical` in:

- `runtime/ucf-compute/src/pipeline.rs`

Top-level request contract:

- `CanonicalPipelineRequest { input: ComputeInput, budget: ComputeBudget }`

This is the explicit Request -> Stage -> Result/Fault entry path for model inference in `runtime/ucf-compute`.

## Canonical stage order

The canonical core stage sequence is fixed to:

1. `World`
2. `SAE`
3. `SSM`
4. `LFM`

Encoded as `CANONICAL_STAGE_SEQUENCE` and emitted in `CanonicalPipelineResult.stage_order`.

NSR is intentionally **not** injected as a mandatory core stage in this step. NSR remains a separate/optional extension point and is not forced into this core inference path.

## Top-level result / failure semantics

Top-level result contract:

- `CanonicalPipelineResult`
  - carries `signals: ComputeSignals` (final compute output)
  - carries `validation_status` and `violation_reason_mask`
  - carries stage/backend provenance (`stage_order`, `route`)
  - carries canonical state (`ok`, `degraded`, `unavailable`)
  - carries structured failure detail (`CanonicalPipelineFailure`)

Structured failure taxonomy (`CanonicalFailureKind`):

- `invalid_input`
- `backend_disabled`
- `backend_contract_mismatch`
- `artifact_unavailable`
- `validation_degraded`
- `budget_exceeded`
- `execution_error`

Separation rules in canonical path:

- `degraded` = pipeline produced output but with degraded quality/validation.
- `unavailable` = output forced to safe unavailable envelope (`risk=1`, `confidence=0`) with structured failure.
- hard execution errors remain Rust `Err(ComputeError::...)` (e.g. invalid input).

## Backend/device routing role split

Routing remains in `backends.rs` + `backend_pack.rs`, while canonical execution is in `pipeline.rs`.

Role split for this hardening step:

- Burn: primary production-intent runtime backend lane.
- Candle: explicit backend/execution seam for parity and adapter isolation.

No implicit fallback from disabled Candle/Burn to hidden production defaults is introduced by this step; unsupported feature lanes still fail explicitly.

Stub/Toy paths remain explicit development/testing paths, not hidden production-normal routing.

## Provenance/diagnostics emitted

`CanonicalPipelineResult` makes visible:

- used stage order (`stage_order`)
- used backend route (`route` = pack + stage backend ids)
- degraded/unavailable state (`state`)
- validation status + violation mask
- structured failure (kind, stage, detail) when present

This keeps diagnostics focused on canonical inference execution without introducing broad telemetry/governance extensions.
