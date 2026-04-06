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
- minimal technical job accounting summary (`JobAccountingSummary`) attached to each `JobRecord`.

No distributed orchestration, persistence, billing/tenant policy, governance scoring, quota economy, or monitoring platform is introduced here.

## Scheduler model (bounded, local)

The scheduler keeps admitted jobs in a FIFO queue and executes them in deterministic cycles:

- `run_scheduler_cycle(max_jobs)` transitions queued jobs to running and then to a terminal state,
- dispatch per cycle is bounded by both:
  - `max_jobs` (caller-provided cycle bound),
  - `SchedulerConfig.max_concurrent_jobs` (service capacity bound),
- queue/running counts are exposed via `scheduler_snapshot`.

There is no fairness/policy layer beyond FIFO and capacity bounds.

## Execution path binding

Execution always goes through `ComputePipelineBackend::compute_canonical`; the service does not define a second stage graph.

- `JobExecutionPath::LocalCanonical`: direct in-process canonical pipeline execution,
- `JobExecutionPath::WorkerIpc`: canonical pipeline backed by worker pack components (worker spawn + IPC transport in `worker_backend`).

Both paths preserve the same `CanonicalPipelineResult` / `CanonicalPipelineFailure` surface.

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
- distributed queueing / persistent job store
- worker fleet orchestration and remote placement
- governance/billing/tenant policy layers
- service-level cancellation or preemption protocols
- external metrics/tracing/datastore platform

This keeps the core load-bearing and minimal while preserving a clean handoff to later scheduling/execution expansion.
