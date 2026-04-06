# Bounded Compute Service Core v1 (Job Lifecycle + Scheduling + Worker Execution)

Status: implemented as a **minimal in-memory bounded service** on top of the canonical runtime pipeline in `runtime/ucf-compute`.

## Scope in this step

The service core wraps the existing canonical `CanonicalPipelineRequest -> canonical pipeline result/fault` path and adds:

- job envelope (`JobId`, request, submission metadata),
- canonical job lifecycle states,
- technical admission before execution,
- structured admission rejection vs. post-admission execution failure separation,
- in-memory queue + lifecycle event log,
- bounded local scheduler cycle (`run_scheduler_cycle`) with configurable dispatch capacity,
- local canonical execution path and worker IPC-backed execution path (`new_worker`).

No distributed orchestration, persistence, billing/tenant policy, governance scoring, or quota economy is introduced here.

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

## Result and failure mapping

Terminal state mapping is lifecycle-native while preserving canonical fault taxonomy:

- `completed`: canonical result without failure,
- `failed`: canonical result with non-timeout failure or execution error from backend call,
- `timed_out`: canonical timeout failure, including backend budget/timeout execution errors mapped to `CanonicalFailureKind::Timeout`.

Worker launch/IPC transport errors are surfaced as structured execution failures (`CanonicalFailureKind::ExecutionError`) and recorded in lifecycle details with execution-path provenance.

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

## What is deliberately not built yet

- scheduler policies beyond FIFO dispatch
- distributed queueing / persistent job store
- worker fleet orchestration and remote placement
- governance/billing/tenant policy layers
- service-level cancellation or preemption protocols

This keeps the core load-bearing and minimal while preserving a clean handoff to later scheduling/execution expansion.
