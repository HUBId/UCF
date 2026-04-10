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

NSR is intentionally **not** injected as a mandatory core stage in this step. NSR is wired as an optional **post-inference reasoning/evidence hook** after core `World -> SAE -> SSM -> LFM` execution, so the canonical core stage order remains unchanged.

Current NSR boundary in `runtime/ucf-compute`:

- request: `contracts::NsrRequest` (base risk/confidence + bounded runtime context)
- result: `contracts::NsrResult` (tightening-only risk/confidence + reason codes + digest)
- failure: `contracts::NsrFailureKind`
- runtime visibility: `CanonicalPipelineResult.nsr_stage` + `ComputeSignals.nsr_digest/nsr_status`

## Top-level result / failure semantics

Top-level result contract:

- `CanonicalPipelineResult`
  - carries `signals: ComputeSignals` (final compute output)
  - carries `validation_status` and `violation_reason_mask`
  - carries domain-scoped validation summary (`validation`)
  - carries load-bearing runtime diagnostics (`diagnostics`)
  - carries stage/backend provenance (`stage_order`, `route`)
  - carries canonical state (`ok`, `degraded`, `unavailable`)
  - carries structured failure detail (`CanonicalPipelineFailure`)

Structured failure taxonomy (`CanonicalFailureKind`):

- `invalid_input`
- `backend_disabled`
- `contract_mismatch`
- `stage_contract_mismatch`
- `artifact_unavailable`
- `artifact_verification_failed`
- `artifact_incompatible`
- `stage_unavailable`
- `degraded_fallback`
- `validation_degraded`
- `budget_exceeded`
- `timeout` (budget/timeout stage naming indicates timeout boundary)
- `execution_error`
- `nsr_disabled`
- `nsr_unavailable`
- `nsr_artifact_verification_failed`
- `nsr_contract_mismatch`
- `nsr_backend_unavailable`
- `nsr_execution_error`

Separation rules in canonical path:

- `degraded` = pipeline produced output but with degraded quality/validation.
- `unavailable` = output forced to safe unavailable envelope (`risk=1`, `confidence=0`) with structured failure.
- `invalid_input` is emitted as structured canonical `unavailable` failure, not as opaque runtime panic/error.

## Backend/device routing role split

Routing remains in `backends.rs` + `backend_pack.rs`, while canonical execution is in `pipeline.rs`.

Role split for this hardening step:

- Burn: primary production-intent runtime backend lane.
- Candle: explicit backend/execution seam bound to the same stage contracts and
  artifact/manifest compatibility checks as Burn.

No implicit fallback from disabled Candle/Burn to hidden production defaults is introduced by this step; unsupported feature lanes still fail explicitly.

Stub/Toy paths remain explicit development/testing paths, not hidden production-normal routing.

## Provenance/diagnostics emitted

`CanonicalPipelineResult` makes visible:

- used stage order (`stage_order`)
- used backend route (`route` = pack + stage backend ids)
- degraded/unavailable state (`state`)
- validation status + violation mask
- validation domains (`validation.input/stage/artifacts/output/evidence`)
- structured failure (kind, stage, detail) when present
- work/budget snapshot (`diagnostics.work`: per-stage remaining units + exceeded stage)
- timing snapshot (`diagnostics.timing`: total + per-stage latency hints)
- stage-level profiling snapshot (`diagnostics.stage_profiles`: stage id, state, duration, remaining work hints, detail)
- stage-cost attribution snapshot (`diagnostics.stage_cost_attribution`: measured timing share + derived work share + dominance/pattern classification + provenance)
- hotspot/bottleneck sketch (`diagnostics.hotspots`: slowest/dominant stage + degraded/skipped/unavailable/failed counts)
- evidence/provenance digest hook (`diagnostics.evidence_chain_digest_prefix`)

This keeps diagnostics focused on canonical inference execution without introducing broad telemetry/governance extensions.

## Minimal stage profiling semantics

Canonical stage profiling intentionally stays narrow and operational:

- states are limited to `success`, `slow_success`, `degraded`, `skipped`, `unavailable`, `failed`
- LFM disabled path is surfaced as explicit `skipped` (no fake execution time)
- budget/degradation paths remain aligned with existing `degraded fallback` semantics
- hard stage failures stay in the same `CanonicalPipelineFailure` taxonomy and are mirrored in stage profile state (`failed`/`unavailable`)

Hotspot derivation is also intentionally minimal:

- `slowest_stage` and `dominant_stage` are selected from observed stage durations
- `dominant_stage_share_bps` reports rough share of total runtime in basis points
- `dominant_work_stage` and `dominant_work_stage_share_bps` expose budget/meter-derived work concentration
- `degraded_stage` and `fallback_stage` make degraded-path vs skipped/fallback context explicit
- counts of degraded/skipped/unavailable/failed stages provide direct operational triage hints

Stage-cost attribution keeps provenance explicit without adding a tracing platform:

- timing shares are marked as `measured_timing`
- work shares are marked as `derived_from_budget_and_meter`
- per-stage `pattern` distinguishes `slow_but_healthy`, `dominant_cost_driver`, `degraded_path_driver`, `skipped_or_fallback`, `hard_failure`, `inactive`

## Job/Ops/History visibility

The same load-bearing profile is threaded into job/history views:

- `JobAccountingSummary.stage_profiles`
- `JobAccountingSummary.stage_cost_attribution`
- `JobAccountingSummary.hotspot_summary`
- persisted job history (`PersistedJobRecord.stage_profiles` + `stage_cost_attribution` + `hotspot_summary`)
- runtime ops snapshot repeated-hotspot hint (`RuntimeOpsSnapshot.repeated_hotspot_stage` + run count)

This keeps one canonical diagnostics surface across runtime execution, job accounting, and replay/history summaries.

## Runtime mode/profile consolidation

Runtime config semantics are consolidated in `runtime_profile` (`RuntimeProfile::from_env`) and documented in:

- `docs/runtime_modes_compute_v1.md`

Canonical `RuntimeOpsSnapshot` includes:

- `runtime_mode`
- `deployment_profile`
- `diagnostic_flags` (`compare_enabled`, `shadow_enabled`, `slot_shadow_enabled`)

This keeps production vs diagnostic/test behavior explicit and visible, while remaining within the existing runtime/ops surfaces.
