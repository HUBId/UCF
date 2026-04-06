# Bounded Compute Service Core v1 (Job Lifecycle + Technical Admission)

Status: implemented as a **minimal in-memory service core** on top of the canonical runtime pipeline in `runtime/ucf-compute`.

## Scope in this step

The service core wraps the existing canonical `CanonicalPipelineRequest -> canonical pipeline result/fault` path and adds:

- job envelope (`JobId`, request, submission metadata),
- canonical job lifecycle states,
- technical admission before execution,
- structured admission rejection vs. post-admission execution failure separation,
- in-memory queue + lifecycle event log.
- minimal job-level observability fields (`job_id`, lifecycle state transitions, admission route, failure kind/detail, evidence-chain digest prefix after execution).

No distributed orchestration, persistence, billing/tenant policy, governance scoring, or quota economy is introduced here.

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
   - unavailable stage backends are rejected
   - stage contract compatibility (`StageContractVersion::V1`) must hold

All admission rejections reuse canonical pipeline failure kinds (`CanonicalFailureKind`) to avoid creating a second error taxonomy.

## What is deliberately not built yet

- scheduler policies beyond FIFO `run_next`
- distributed queueing / persistent job store
- worker fleet orchestration and remote placement
- governance/billing/tenant policy layers
- service-level cancellation protocols

This keeps the core load-bearing and minimal while preserving a clean handoff to later scheduling/execution expansion.
