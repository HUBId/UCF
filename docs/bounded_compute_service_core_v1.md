# Bounded Compute Service Core v1 (Lifecycle + Accounting + Observability + Service Hardening)

Status: implemented as a **minimal in-memory bounded service** on top of the canonical runtime pipeline in `runtime/ucf-compute`.

## Scope in this step

The service core wraps the existing canonical `CanonicalPipelineRequest -> canonical pipeline result/fault` path and adds:

- job envelope (`JobId`, request, submission metadata),
- canonical job lifecycle states,
- technical admission before execution,
- structured admission rejection vs. post-admission execution failure separation,
- in-memory queue + lifecycle event log,
- bounded local scheduler cycle (`run_scheduler_cycle`) with configurable dispatch capacity,
- local canonical execution path and worker IPC-backed execution path (`new_worker`),
- optional multi-worker execution registry (`MultiWorkerComputeService`) for explicit local vs secondary-worker placement,
- minimal technical job accounting summary (`JobAccountingSummary`) attached to each `JobRecord`.

No distributed orchestration, persistence, billing/tenant policy, governance scoring, quota economy, or monitoring platform is introduced here.

## Scheduler model (bounded, local)

The scheduler keeps admitted jobs in a FIFO queue and executes them in deterministic cycles:

- `run_scheduler_cycle(max_jobs)` transitions queued jobs to running and then to a terminal state,
- dispatch per cycle is bounded by both:
  - `max_jobs` (caller-provided cycle bound),
  - `SchedulerConfig.max_concurrent_jobs` (service capacity bound),
- queue/running counts are exposed via `scheduler_snapshot`.
- in multi-worker mode, scheduling uses a minimal technical triage:
  - `run_now` (currently placeable),
  - `queue_required` (admissible but currently unavailable/saturated),
  - `not_placeable` (backend/device incompatibility).

There is no fairness/policy layer beyond FIFO and capacity bounds.

## Execution path binding

Execution always goes through `ComputePipelineBackend::compute_canonical`; the service does not define a second stage graph.

- `JobExecutionPath::LocalCanonical`: direct in-process canonical pipeline execution,
- `JobExecutionPath::WorkerIpc`: canonical pipeline backed by worker pack components (worker spawn + IPC transport in `worker_backend`).

Both paths preserve the same `CanonicalPipelineResult` / `CanonicalPipelineFailure` surface.

`MultiWorkerComputeService` extends this with a thin execution-unit registry:
- `ExecutionUnitKind::Local` and `ExecutionUnitKind::Worker`,
- stable registry identity fields (`ExecutionUnitId`, worker class, registry role),
- deterministic worker ids (`ExecutionUnitId`) and explicit availability (`available` / `unavailable`),
- optional per-job placement hint (`requested_unit`) with deterministic fallback to capability-aware placement.
- runtime worker lifecycle state (`known|ready|busy|saturated|degraded|unavailable|stale|unknown|unhealthy`)
  derived from availability, dispatch pressure, health-contact recency, and failure/cooldown streak.
- per-unit health timestamps (`last_health_contact_at_unix_ms`) with narrow cooldown/quarantine
  semantics to avoid immediate re-dispatch to intermittently failing workers.

## Backend capability matrix + execution placement (technical, narrow)

Placement now uses a small capability assessment for every execution unit before dispatch:

- lane classification from backend composition (`burn`, `candle`, `worker`, `toy`, `mixed`),
- execution device class classification (`cpu`, `worker`),
- technical admission compatibility (`ComputePipelineBackend::technical_admission`),
- explicit backend suitability state per candidate:
  - `suitable`
  - `incompatible`
  - `disabled`
  - `unavailable`
- explicit device suitability state per candidate:
  - `suitable`
  - `unsuitable`
  - `disabled`
  - `unavailable`

Selection rules stay intentionally simple:

1. requested unit: must be `suitable`, otherwise fail with structured placement failure.
2. automatic selection:
   - prefer `burn` suitable units (primary productive lane),
   - otherwise use `candle` suitable units as explicit degraded fallback (secondary seam),
   - otherwise any other suitable unit.
3. if no candidate is suitable right now:
   - transient unavailability is deferred (re-queued with bounded retry),
   - structural incompatibility is rejected as placement failure.

Placement candidate diagnostics now include the runtime worker health state used at dispatch time,
so skipped workers are visible with an explicit reason (for example `unit not dispatchable (Stale)` or
`unit not dispatchable (Degraded)`).

Warmup/readiness is now also emitted as a backend/device path context token
(`backend_device_readiness=<lane>:<device>:<state>`), where state is one of:
`cold`, `prepared`, `warm_ready`, `stale`, `blocked`.
This keeps warm-vs-cold decisions load-bearing per backend/device path instead of as a global flag.

The resulting `ExecutionPlacement` now carries provenance fields:

- selected lane + suitability,
- selected execution device class + device suitability,
- device preference/provenance for requested worker placement (`device_preference`, `device_preference_met`, fallback-from),
- `degraded_fallback` flag,
- selection reason,
- full candidate assessment list (`considered`) so non-selected backends/workers are diagnosable.

The worker side still executes the same canonical pipeline contract (`CanonicalPipelineRequest -> CanonicalPipelineResult`), including the same stage-order and failure-kind semantics.

## Technical accounting surface (job-level, non-billing)

Each `JobRecord` now carries `accounting: JobAccountingSummary` with load-bearing technical fields:

- `job_id`, lifecycle `status`, and `completion_class`,
- submitted/start/end timestamps,
- queue wait, execution duration, total duration,
- canonical failure class (if present),
- canonical work budget summary (`CanonicalWorkSummary`) when execution reached pipeline result,
- pipeline provenance mirror:
  - stage order + executed stages,
  - pipeline state (`ok` / `degraded` / `unavailable`),
  - model slot provenance list,
  - execution path (`local_canonical` / `worker_ipc`).

This is intentionally technical accounting only. No prices, no tenant quota policy, no invoicing schema.

## Result, completion, and failure mapping

Terminal mapping stays lifecycle-native and reuses canonical fault taxonomy:

- `rejected_before_execution`: admission rejection before run,
- `completed`: canonical result without failure and non-degraded state,
- `degraded_completed`: canonical completion with `CanonicalPipelineState::Degraded`,
- `failed_during_execution`: run started and ended in non-timeout failure,
- `timed_out`: canonical timeout failure, including backend budget/timeout execution errors mapped to `CanonicalFailureKind::Timeout`,
- `worker_ipc_failure`: worker/IPC execution-path failure classified as execution error.

Worker launch/IPC transport errors are surfaced as structured execution failures (`CanonicalFailureKind::ExecutionError`) and remain linked to execution-path provenance in lifecycle events.

Placement-level failures are additionally tagged as:

- `no_suitable_backend`
- `no_suitable_device`
- `backend_incompatible`
- `backend_device_incompatible`
- `device_unavailable`
- `backend_unavailable`
- `worker_placement_failed`
- `currently_unschedulable`

For multi-worker execution, dispatch outcome is classified with a single linked surface:
- `unavailable`,
- `deferred`,
- `dispatch_failure`,
- `transport_failure`,
- `execution_failure`,
- `timeout`,
- `completed`,
- `redispatched_local` (worker failed, job completed via explicit local fallback).

Additionally, each `MultiWorkerJobRecord` now carries `retry_summary` to make retry/redispatch
provenance explicit in ops/history-style diagnostics:
- `attempts`: terminal attempt count observed for the job execution path,
- `retries_exhausted`: bounded retry budget was consumed without recovery,
- `uncertain_prior_attempt_outcome`: set when transport-level mismatch means prior attempt
  completion cannot be proven from IPC correlation alone,
- `recovered_by`: recovery class (`retry_same_worker`, `redispatch_alternate_worker`, `local_fallback`) when recovery succeeded,
- `last_failure_kind`: worker-failure classification (`dispatch_failed_before_execution`,
  `transport_failure`, `worker_unavailable_or_stale`, `worker_execution_crashed`,
  `worker_structured_execution_failure`, `terminal_compute_execution_failure`).

`MultiWorkerJobRecord` now also carries a narrow `coordination` snapshot so distributed in-flight
state remains visible across worker/service/recovery boundaries without adding a workflow engine:
- canonical in-flight state: `queued`, `dispatching`, `running`, `awaiting_worker_outcome`,
  `retry_pending`, `redispatch_pending`, `uncertain`, `stale`,
- canonical terminal coordination state: `completed`, `failed`, `timed_out`,
- last in-flight state before terminalization (`last_in_flight_state`),
- last known owner (`owner`, `owner_kind`) + owner last status contact timestamp,
- freshness (`current`, `stale`, `uncertain`),
- stale/orphan diagnostics (`stale_worker_ownership`, `missing_worker_outcome`,
  `orphaned_in_flight_job`, `recovered_coordination_state`),
- recovery/dispatch signal (`safe_to_redispatch`, `unsafe_uncertain_prior_attempt`,
  `await_worker_outcome`, `recovery_decision_required`, `terminal`).

`MultiWorkerComputeService::in_flight_jobs()` exposes a compact runtime snapshot of currently
non-terminal coordination states (including queue-only jobs), so ops/recovery paths can
differentiate truly running jobs from stale/uncertain/orphaned in-flight residue.

This keeps the worker failure semantics explicit without adding a second failure taxonomy:
- no suitable healthy worker -> placement failure (`currently_unschedulable` or device/backend unavailable),
- selected worker became unavailable/degraded/stale -> worker unavailable outcome,
- dispatch transport/execute failure -> `transport_failure` / `execution_failure`,
- fallback to local worker -> `redispatched_local` with preserved provenance.

Worker-IPC internals now apply a **single bounded same-worker retry** only for transient worker/IPC
classes (`dispatch_failed_before_execution`, `transport_failure`,
`worker_unavailable_or_stale`, `worker_execution_crashed`). Structured compute failures reported by
the worker remain terminal and are not auto-retried.

When a non-requested worker fails at dispatch/execution and a local unit is still suitable, a
single minimal re-dispatch to local is attempted. Requested-worker submissions stay strict and do
not auto-fallback.

Recovery/retry coupling remains intentionally small:
- transport mismatch / missing correlated worker outcome -> `uncertain` + `missing_worker_outcome`
  + `unsafe_uncertain_prior_attempt`,
- stale ownership at dispatch boundary -> `stale` + `stale_worker_ownership`
  + `recovery_decision_required`,
- successful local redispatch after worker failure -> terminal record with
  `recovered_coordination_state`.

## Canonical job lifecycle states

- `submitted`
- `admitted`
- `rejected`
- `queued`
- `running`
- `completed`
- `failed`
- `timed_out`

`canceled` is intentionally not added in this step because there is no canonical cancellation execution path yet.

## Service-level observability guarantees

Lifecycle events now include:

- `job_id`,
- lifecycle `state`,
- failure kind (if any),
- execution-path detail,
- observed timestamp,
- completion class on terminal transitions.

Together with `JobAccountingSummary`, this gives a minimal but load-bearing per-job trace for admission decision, queueing, execution, and completion/failure.

## Fault domains + isolation boundaries (minimal, canonical)

The runtime now exposes a narrow fault-domain classification for canonical failures, without adding a parallel reliability control plane.

Canonical fault domains:

- `artifact_model`: model/artifact slot failures (unavailable, verification failed, incompatible),
- `stage`: stage-local contract/runtime degradation and stage unavailability,
- `backend`: backend capability/contract disablement or incompatibility,
- `worker_transport`: execution-path/transport execution errors,
- `placement_capacity`: budget/timeout/capacity pressure escalations,
- `runtime_service`: request/runtime service level faults.

Each canonical failure kind maps to one isolation disposition:

- `locally_isolated`: local containment with no job-level impact,
- `degraded_but_serviceable`: degraded but still serviceable output,
- `hard_escalation_job_failure`: non-isolatable escalation to job failure,
- `service_runtime_impact`: runtime/service-level impact.

Current canonical behavior intentionally stays minimal:

- stage-level degraded fallbacks are marked `degraded_but_serviceable`,
- artifact/backend/placement/worker faults are explicit and escalate honestly,
- failure-domain and isolation context are propagated into runtime notes and persisted history (`fault_domain`, `fault_isolation`, `fault_systemic`) for ops/diagnostics.

Deliberate limits in this step:

- no incident-management workflow,
- no orchestration/sandbox/container isolation framework,
- no automated global self-healing controller.

## Technical admission checks (pre-run)

Admission runs via `ComputePipelineBackend::technical_admission` and rejects before execution when any of these fail:

1. request validity:
   - `input.t != 0`
   - budget timing fields are non-zero
2. budget compatibility:
   - `max_micros <= hard_timeout_micros`
   - stage/global work-unit budgets are non-zero
3. artifact readiness:
   - required slot failures map to canonical artifact failure kinds
4. backend / contract compatibility:
   - disabled stage backends are rejected
   - stage contract compatibility (`StageContractVersion::V1`) must hold

All admission rejections reuse canonical pipeline failure kinds (`CanonicalFailureKind`) to avoid creating a second error taxonomy.

## Smoke + integration hardening reference coverage

`runtime/ucf-compute/src/compute_service.rs` includes service-focused tests covering:

- submit → admit → queue → run → complete terminal flow,
- submit → reject for invalid input, budget mismatch, artifact/backend admission issues,
- run-time failure mapping (timeout class and worker launch/IPC error mapping),
- accounting + provenance population for completed jobs,
- integration path: canonical onboarding reference backend wrapped through bounded service (when backend features are available in current build).

## What is deliberately not built yet

- scheduler policies beyond FIFO dispatch
- distributed queueing / crash-recoverable queue replay
- worker fleet orchestration and remote placement
- governance/billing/tenant policy layers
- service-level cancellation or preemption protocols
- external metrics/tracing/datastore platform

This keeps the core load-bearing and minimal while preserving a clean handoff to later scheduling/execution expansion.

## Canonical technical compute service surface (v1)

For service-level integration, the canonical technical entrypoint is now `CanonicalComputeEntryPoint` (`runtime/ucf-compute/src/service_surface.rs`).

This surface sits directly on top of `InMemoryComputeService` and keeps one coherent request/result world:

- request: `ComputeSubmitRequest` (`pipeline_request`, submit metadata, execution mode),
- invalid request: `ComputeSubmitOutcome::Invalid` (input envelope validation before bounded service submit),
- admitted/rejected split:
  - `ComputeSubmitOutcome::Rejected { status }` for canonical admission rejection,
  - `ComputeSubmitOutcome::Accepted { status, completion }` for admitted jobs.

Execution behavior is explicit and narrow:

- `ComputeExecutionMode::EnqueueOnly`: submit/admit/queue and return job handle + queued status,
- `ComputeExecutionMode::ExecuteInline`: submit then trigger one scheduler step and return optional terminal completion status.

Job and status surface is unified through:

- `ComputeJobHandle { job_id }`,
- `CanonicalComputeEntryPoint::status(handle)` for lifecycle/completion/fault/provenance snapshot,
- `CanonicalComputeEntryPoint::lifecycle(handle)` for per-job lifecycle events.

The status snapshot mirrors load-bearing service provenance already tracked by bounded compute service:

- admission failure vs execution failure,
- lifecycle state + completion class,
- failure kind taxonomy (`CanonicalFailureKind`),
- execution path (`local_canonical` / `worker_ipc`),
- pipeline state/work summary/model slot provenance.

Low-level `InMemoryComputeService::submit/run_next/run_scheduler_cycle` remains available as an internal primitive for tests and runtime internals, but service consumers should use `CanonicalComputeEntryPoint` as primary compute ingress.

## Minimal runtime operations surface (v1)

`CanonicalComputeEntryPoint` now also exposes a narrow operations-facing layer intended for
technical runtime control only (no second admin/control plane):

- `operations_snapshot() -> RuntimeOpsSnapshot`
- `run_operation(RuntimeOperation) -> RuntimeOperationOutcome`

### Runtime snapshot coverage

`RuntimeOpsSnapshot` provides the minimal load-bearing state for runtime operations:

- runtime state: `healthy_ready | degraded | partially_unavailable | unavailable`
- runtime signal quality: `known | unknown` (unknown when no job-derived signal exists yet)
- scheduler/queue envelope: queued, running, max-concurrency, execution path
- job summary counters (submitted/completed/failed/rejected/timed_out/degraded)
- latest slot provenance snapshot (required slots + runtime status)
- canonical job handles (`active_job` plus reserved `candidate/compare/shadow` fields)

The `candidate/compare/shadow` fields are intentionally present but currently `None` in the
in-memory service because no extra promotion/compare queue exists at this layer.

`RuntimeOpsSnapshot` now also carries `latest_baseline_comparison` when an explicit
candidate-vs-baseline check was executed through the canonical entry point. This keeps comparison
visibility in the same ops snapshot surface already used for runtime diagnosis and promotion prep.

`RuntimeOpsSnapshot.canonical` is the canonical expert diagnostics frame for the same data. It adds:

- `consistency`: `current | partial | stale | unavailable`
- `diagnostics_availability`: `available | partial | unavailable | blocked | internal_only`
- top-level caveats plus subsystem summaries for:
  worker, placement/capacity, rollout, warmup/capability, replay/history, specialization.

This keeps subsystem diagnostics tied to one load-bearing top-level semantic frame instead of
requiring separate interpretation paths.

### Runtime state semantics

State is derived from real service signals only:

- `unavailable`: missing required slot signals and no successful completion observed
- `partially_unavailable`: missing required slots or queued backlog still pending
- `degraded`: high terminal failure ratio and/or completed degraded pipeline outputs
- `healthy_ready`: none of the above conditions hold

No synthetic “always green” health outcome is emitted.

Snapshot consistency semantics are intentionally narrow:

- `current`: canonical view is coherent and replay/history fidelity has no active caveat.
- `partial`: canonical view exists, but required slot or snapshot fidelity caveats are present.
- `stale`: diagnostics rely on stale/incomplete history snapshot context.
- `unavailable`: runtime status itself is currently unavailable.

These are diagnostics semantics only; they do not replace compute/pipeline fault taxonomy.

## Minimal replay + reproducibility surface (v1)

`CanonicalComputeEntryPoint` now exposes a narrow replay trigger:

- `replay_preflight(ComputeJobHandle) -> ComputeReplayPreflight`
- `replay(ComputeJobHandle) -> ComputeReplayOutcome`

Replay is intentionally grounded in the same canonical compute path (`submit` + `run_next`) and
does not introduce a second replay pipeline.
`replay(...)` always runs the technical preflight first and only starts execution if the replayability
state is not blocked/insufficient.

### Replay record basis

Replay uses existing job/history persistence (`PersistedJobRecord`) and extends it with minimal
load-bearing replay fields:

- canonical request basis (`request identity` + canonical budget snapshot),
- executed path snapshot (`requested vs executed path`, lane/resource class, local/remote flag),
- backend route + model-slot summary from the effective run,
- rollout hint (`active` vs `guarded/candidate` vs `fallback_or_stale`) derived from slot status,
- top-level execution result summary (lifecycle/completion/failure/pipeline state),
- snapshot readiness class: `replay_ready`, `partial`, `insufficient`, `stale_or_incomplete`.
- backend/device readiness context and replay caveat fields when source run used
  cold/stale/blocked backend-device readiness.

This keeps replay tied to the same history layer used by normal jobs.

`replay_ready` means canonical request + effective execution context are available for bounded
replay checks. `partial` means replay may still run but only with limited fidelity checks.
`insufficient` and `stale_or_incomplete` are explicitly non-load-bearing and treated as
configuration-incomplete replay inputs.

### Replayability preflight classes (canonical)

Preflight classifies a source run before replay execution:

- `replay_ready`
- `replayable_with_caveats`
- `replayable_only_under_changed_context`
- `insufficient_for_replay`
- `blocked_for_replay`

Additionally, replay preflight/report now expose a constrained backend/device support class to keep
replayability caveats explicit instead of implicit:

- `fully_supported`
- `replayable_with_backend_device_caveat`
- `supported_only_under_guardrails`
- `not_meaningfully_comparable`
- `blocked_for_replay`

Both surfaces also carry `constrained_backend_device_context` (`source=...;current=...`) so
backend/device-path drift can be diagnosed as replay caveat vs replay blocker.

Preflight surfaces bounded reasons (`ReplayPreflightIssueCode`) for missing artifacts/slots,
changed backend/device/worker context, incomplete snapshots, changed rollout context, local/remote
path mismatch, and non-fidelity-equivalent caveats.

Replay preflight now also treats a changed backend/device readiness context
(e.g. source warm-ready path vs current cold/stale path) as a load-bearing
backend/device context shift rather than a generic warm/cold mismatch.

This remains technical preflight only (no certification/governance workflow).

### Consolidated replay mismatch view (v1)

Preflight and replay reports now share one compact `ReplayMismatchView` instead of separate
diagnostic worlds.

Canonical mismatch classes:

- `exact_or_close_replay_context`
- `context_changed_with_caveat`
- `meaningful_replay_but_mismatched_execution_context`
- `insufficiently_comparable`
- `blocked_by_missing_prerequisites`
- `replay_execution_diverged_technically`

Canonical mismatch categories/reasoning remain bounded to load-bearing replay context:

- snapshot completeness (`snapshot completeness mismatch`)
- artifact/slot state (`artifact/slot/hash mismatch` in effective slot provenance)
- backend/device/worker/placement context
- rollout/activation context
- local-vs-remote context bridge
- top-level result/fault divergence

`ReplayMismatchView.primary_reasons` is intentionally capped to the first 1-3 dominant reasons so
ops/history/replay views can explain why a replay is clean, caveated, mismatched, blocked, or
diverged without adding a full diff/audit platform.

### Replay outcome determinism classes

Replay outcomes are explicitly classified:

- `same_effective_configuration`: replay ran with matching execution path/lane/backend route/slot summary,
- `replayable_not_strictly_deterministic`: replay completed, but effective configuration changed,
- `not_replayable_under_current_runtime_state`: replay run failed under current runtime conditions.

No global strict determinism promise is made.

### Deterministic-subset classification (narrow and explicit)

Replayability and deterministic-subset semantics are intentionally separate:

- replayability answers: can replay run at all (possibly caveated)?
- deterministic-subset answers: does this run belong to the narrower stable-replay subset?

Canonical deterministic-subset classes:

- `deterministic_subset_candidate` (preflight-ready candidate)
- `stable_replay_subset` (completed replay under same effective context/config)
- `replayable_but_not_deterministic_subset`
- `excluded_from_deterministic_subset`

Canonical eligibility states shared by preflight and replay diagnostics:

- `stable_subset_eligible`
- `stable_subset_excluded_with_reason`
- `stable_subset_uncertain_due_to_missing_signal`

Bounded reason codes (non-exhaustive, load-bearing only):

- changed remote/worker context
- changed backend/device/runtime mode context
- rollout boundary relevance
- incomplete snapshot/context
- degraded/fallback/retry/redispatch execution context
- missing signal for stable classification
- replay outcome changed/diverged

This is a technical replay hardening layer only; it is not a determinism guarantee,
formal verification claim, or certification workflow.

### Structured replay failure semantics

The replay surface uses one bounded failure taxonomy:

- `record_missing`
- `configuration_incomplete`
- `required_artifact_unavailable`
- `backend_or_device_unavailable`
- `changed_runtime_context_incompatible`
- `replay_execution_failed`
- `replay_completed_with_changed_configuration`

This distinguishes unavailable historical records, incomplete replay inputs, runtime drift, and
execution failure without adding a second error world.

## Minimal baseline comparison surface (v1)

`CanonicalComputeEntryPoint` now exposes one narrow comparison trigger:

- `compare_against_baseline(candidate, baseline_ref) -> BaselineComparisonResult`

Baseline selection is intentionally bounded:

- explicit baseline by `job_id`,
- or `latest_by_request_identity` (same request identity + same canonical budget fingerprint +
  same lane/backend context).

No second benchmark database or independent comparison pipeline is introduced.

### Canonical comparison semantics

- `Compared(summary)` when candidate and baseline are terminal and configuration-compatible.
- `NotComparable` with bounded failure codes:
  - `no_baseline_available`
  - `baseline_incompatible`
  - `candidate_incompatible`
  - `comparison_execution_failed`
  - `not_meaningful_under_runtime_change`

This keeps non-comparability explicit and aligned with existing replay/runtime drift semantics.

### Load-bearing signals compared

The comparison layer is intentionally minimal and uses existing runtime summaries:

- terminal completion class delta (improved / equivalent / regressed ranking),
- failure kind change,
- degraded vs non-degraded pipeline state change,
- work-summary equality + remaining global work units snapshot,
- execution path/lane/backend route/model-slot compatibility checks.

No statistical suite, leaderboard, or governance signoff layer is added at this stage.

### Supported operations

- `RuntimeOperation::Snapshot`: explicit snapshot action (always applied)
- `RuntimeOperation::DrainScheduler { max_jobs }`: controlled execution of queued jobs
- `RuntimeOperation::RefreshRuntime`: currently returns structured `Unsupported` for
  `InMemoryComputeService` (no hidden refresh side-path)

### Bounded runtime recovery flow semantics (Serie G)

Runtime operations now expose an explicit bounded recovery interpretation instead of overloading
queue/recovery/replay signals implicitly:

- `refresh` (`RuntimeRecoveryFlow::RefreshState`): re-capture current runtime diagnostics/snapshot
  when stale basis is suspected.
- `resync` (`RuntimeRecoveryFlow::ResyncState`): reconcile diverged in-memory runtime/queue/worker
  views (bounded to scheduler/worker-readiness actions).
- `rehydrate` (`RuntimeRecoveryFlow::RehydrateState`): move persisted history/runtime intent back
  into active in-memory recovery state.
- `no-op` / `blocked` (`RuntimeRecoveryFlow::{NoOpRecoveryAction,BlockedRecoveryAction}`):
  explicitly represented rather than inferred from generic failure codes.

`RuntimeOpsSnapshot.bounded_recovery` provides a canonical recommendation driven by existing
signals:

- stale snapshot basis → refresh candidate,
- drift/inconsistency signals → resync candidate,
- orphaned/unreconciled persisted work → rehydrate candidate (or blocked when history is missing).

`RuntimeOperationOutcome` now returns:

- `recovery_flow` (refresh/resync/rehydrate/no-op/blocked),
- `recovery_state` (`state_refreshed | state_resynced | state_rehydrated | no_relevant_change |
  partial_recovery | blocked_unable_to_restore_trustable_state`),
- `trust_state_after` (`trustable | partial | blocked`) from the canonical post-operation snapshot.

This keeps resulting state change and remaining uncertainty diagnosable without adding an
orchestration/self-healing platform.

`RuntimeOperationOutcome` separates `Applied`, `Unsupported`, and `Failed` outcome classes to keep
operations failure semantics explicit and aligned with the canonical runtime failure world.

## Minimal job history persistence (v1)

`CanonicalComputeEntryPoint` can now be initialized with a file-backed history store:

- `CanonicalComputeEntryPoint::with_history_path(service, path)`
- `CanonicalComputeEntryPoint::with_history_store(service, store)`

The store is intentionally narrow (`runtime/ucf-compute/src/job_history.rs`):

- append-only JSONL records,
- one canonical load-bearing record per job id (last write wins in memory when reloaded),
- no DB, no event-sourcing layer, no analytics/query engine.

Persisted job record coverage is limited to the load-bearing fields:

- identity: `job_id`, submitter, request identity (`frame_id`, `t`, `context_digest`),
- lifecycle: last known state + terminal completion class,
- timing: submit/start/finish, queue wait, execution duration, total duration,
- failure/completion class: failure kind + pipeline state,
- accounting/provenance summary: work budget summary and model slot provenance summary.

History lookup is intentionally minimal and technical:

- `history_lookup(handle)` supports lookup by `job_id`,
- returns either `Found(record)` or `NotFound`,
- returns `StoreUnavailable` when no persistent history store is configured.

Persistence failure semantics stay separate from compute execution semantics:

- jobs can still complete/failed/timed_out/rejected in-memory even when history writes fail,
- persistence errors are surfaced through `history_status()` (`configured`, `available`, `last_error`),
- this keeps “job execution failed” distinct from “job succeeded but history persistence failed”.

## Minimal restart / recovery / resume semantics (v1)

`CanonicalComputeEntryPoint::with_history_store(...)` and `with_history_path(...)` now run a
minimal recovery pass over persisted history records during service startup.

Canonical behavior is intentionally narrow and explicit:

- terminal jobs (`completed` / `failed` / `timed_out` / `rejected`) stay
  `completed_before_restart` and are not replayed automatically,
- pre-execution jobs (`submitted` / `admitted` / `queued`) are only resumed when a
  `canonical_request` is present; resume is implemented as deterministic rehydration into the
  in-memory queue,
- `running` jobs from before restart are treated as uncertain runtime state and are surfaced as
  `running_state_uncertain_after_restart`,
- records without enough canonical request data are surfaced as
  `resume_unsupported` / `rerun_required`,
- hard restart recovery reconstruction failures are surfaced as `restart_recovery_failed`.

This is an honest boundary:

- **true in-process continuation of an interrupted execution is not implemented**,
- resume currently means “recover pending pre-execution intent from persisted canonical request”,
- interrupted or uncertain worker execution remains rerun-based, not checkpoint-resume.

Recovery context is carried through ops/history surfaces:

- `RuntimeOpsSnapshot.recovery` exposes aggregate recovery counts + per-job recovery records,
- recovered jobs include recovery provenance in job status/history (`recovery_status`,
  `recovery_source_job_id`, `recovery_note`),
- replay/rerun tooling can distinguish restarted-context jobs from regular submissions.
