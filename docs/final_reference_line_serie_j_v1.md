# Final Reference Line — Serie J v1

Status: technical convergence note for the current productive core line in `runtime/ucf-compute`.

Source-of-truth constants live in:

- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_COMPUTE_REFERENCE_MAP`
  - `CANONICAL_FINAL_REFERENCE_LINE`

This file is intentionally short and load-bearing only.

## 1) Final reference line (canonical productive core + explicit extensions)

- Execution core:
  - `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- Rollout extension on top of same core:
  - `rollout diagnostics -> activation/fallback/rollback -> active production line`
- Replay extension on top of same core:
  - `replay_preflight -> replay_with_entry -> comparison/evidence on same result/fault/status core`
- Diagnostics/expert extension on top of same core:
  - `runtime snapshot/diagnostics + expert workflow surface -> same canonical core state`
- Cross-cutting production invariants on top of same core:
  - `blocked!=failed!=no_op; partial/stale/caveated/degraded remain distinct; rollout/replay/expert extend shared core`
- Non-canonical boundary (explicit):
  - `compatibility backends + internal/legacy worker/domain lanes are extension/internal only`

## 2) Canonical path scopes

Canonical production path remains:

1. `service_surface::CanonicalComputeEntryPoint::submit`
2. `pipeline::ComputePipelineBackend::compute_canonical`
3. rollout activation core in `enablement + model_store`

Canonical extensions (not second cores):

- expert runtime workflows: inspect/diagnose/act, replay-oriented, rollout-oriented
- diagnostics/evidence/history/replay comparability surfaces

Internal/non-canonical lanes remain explicit:

- compatibility constructor lane (`build_backend(kind=stub|candle)`)
- internal/legacy worker+domain compatibility boundary

## 3) Semantics convergence points (load-bearing)

- action/result/fault/status semantics are shared-core contracts (`contracts.rs`).
- `current|partial|stale|drift|unavailable` style runtime semantics stay normalized through canonical snapshot + diagnostics views.
- `active|candidate|compare|shadow|fallback|rollback|guarded` rollout semantics remain explicit in rollout lanes and do not redefine execution core semantics.
- replay/comparison/evidence remains anchored to the same canonical run contracts and snapshot evidence.
- replay caveats do not override hard blockers, rollout guardrails do not redefine healthy-active core status, and expert mutations without trustable state basis remain blocked.

## 4) Intentional boundaries

Not part of this convergence layer:

- no new orchestration/governance plane
- no broad repo reorganization
- no second reference language for code vs docs

The objective is one readable productive line with explicit extensions and explicit internal boundaries.
