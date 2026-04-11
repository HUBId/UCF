# Real Compute Reference Surface v1

Status: technical reference for the current `runtime/ucf-compute` kernel surface.

## 1) Canonical real-compute kernel (load-bearing)

The canonical productive path is the combination of:

1. service entry: `service_surface::CanonicalComputeEntryPoint`
2. bounded service/scheduler/placement: `compute_service::{InMemoryComputeService, MultiWorkerComputeService}`
3. canonical stage contract: `pipeline::ComputePipelineBackend::compute_canonical`
4. canonical onboarding backend: `backends::build_canonical_production_backend` (`BurnToyV1` pack)
5. slot/readiness/artifact foundation: `backend_pack`, `model_store`

This is the primary reference layer for production runtime behavior.

## 2) Extension paths (supported, non-primary)

Supported extension paths that stay attached to the same canonical contracts:

- multi-worker and worker IPC execution (`worker_backend`, `ipc`)
- optional remote execution support (`remote_compute`, feature-gated)
- rollout slot-path support (`active/candidate/compare/shadow`) in `enablement` and `model_store`
- history/recovery/replay persistence surfaces (`job_history`, replay/recovery methods in `service_surface`)

These paths extend placement/rollout/operability without introducing a second compute stage graph.

## 3) Diagnostic/test-near paths (explicitly non-primary)

Diagnostic and compatibility paths are intentionally separated from production defaults:

- compatibility backends `stub` and `candle` (`backends::build_backend`)
- compare/shadow diagnostic runtime modes and slot-level shadow diagnostics
- stage-level hotspot/diagnostic reporting surfaces
- test-only harness helpers (e.g. `test_env`)

They remain valid for validation and rollout diagnostics but are not the canonical productive default.

## 4) Contract and terminology alignment

Load-bearing terminology used consistently across service + pipeline + history:

- `request` = submission envelope
- `job` = admitted lifecycle entity
- `run` = execution attempt of a job
- `replay` = history-backed rerun/verification path

State and failure semantics are kept explicit:

- runtime/pipeline state: `ok`, `degraded`, `unavailable`
- lifecycle terminal classification: completed / degraded_completed / failed / timed_out / rejected_before_execution
- rollout path state: `active`, `candidate`, `compare`, `shadow`, `disabled`, `blocked`

## 5) Canonical vs side-entry guidance

- Primary production reference should use `build_canonical_production_backend` and the canonical service surface.
- `build_backend` remains a compatibility-oriented constructor and defaults to `stub` for safe/dev usage.
- compare/shadow paths are side paths and must not become implicit production defaults.

## 6) Deliberate boundaries that remain

This reference layer intentionally does **not** add:

- a separate orchestration/governance control plane in runtime
- a second parallel stage graph for diagnostics
- large repository reorganization

The goal is a coherent technical reference surface over the existing real-compute kernel.

## 7) Replay context bridge semantics (local vs remote)

Replay keeps a minimal execution-context bridge between source and replay runs, instead of
assuming local-only or remote-only replay semantics:

- source + replay context descriptors include:
  - execution mode/path (`Local` vs `RemoteWorkerIpc`)
  - lane/resource/capacity hints when available
  - backend-route availability
  - remote context completeness (`complete`, `partial`, `not_applicable`, `unavailable`)
- context transitions are explicitly classified as:
  - `local_to_local`
  - `local_to_remote`
  - `remote_to_local`
  - `remote_to_remote_same`
  - `remote_to_remote_changed`

Replay preflight and replay reports classify consistency across context boundaries as:

- `same_effective_execution_context`
- `changed_comparable_execution_context`
- `changed_context_with_fidelity_caveat`
- `not_meaningfully_comparable`

Preflight reuses the same replay path and explicitly marks context-bridge caveats/failures:

- `original_context_unavailable` (e.g. missing source remote context)
- `alternative_context_with_caveats` (e.g. local↔remote bridge required)
- `context_bridge_too_lossy` (bridge would be too lossy for meaningful replay)

Boundaries kept intentionally:

- no deterministic equivalence guarantee across local/remote boundaries
- no forensic diff platform or distributed reconciliation matrix
- only load-bearing mismatch summaries for replay diagnostics and ops/history views
