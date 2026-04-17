# Real Compute Transition Checkpoint

This file defines the repo-based transition point between **Real Compute Onboarding** and **Real Compute Stack** work.

## Canonical role in docs/status/readiness surfaces

This file is the transition decision surface and must align with:

- technical reference surface: `docs/real_compute_reference_surface_v1.md`
- code-level lane authority: `runtime/ucf-compute/src/reference_map.rs`
- readiness classification surface: `docs/real_compute_readiness_sweep_v26.md`
- status companion: `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`

It records transition criteria and scope boundaries; it does not redefine runtime contracts.

## 1) Repo truth: onboarding target areas vs current state

| Area | Repo-based status | Evidence anchor (code/tests/docs) |
|---|---|---|
| Canonical model pipeline (`runtime/ucf-compute`) | **Done (minimal honest path)** | Canonical backend builder + fixed stage sequence (`World -> Sae -> Ssm -> Lfm`) are present and exercised through service integration. |
| Artifact / manifest / compatibility | **Done (usable in runtime path)** | Canonical default manifest path is lowercase (`models/manifest.toml`), model-slot verification/provenance is part of canonical results, admission rejects artifact failures. |
| Burn as primary runtime path | **Done (for onboarding scope)** | Burn onboarding route is first-class in canonical backend selection and canonical pipeline runtime path. |
| Candle as backend seam | **Done as seam, partial as production lane** | Candle path is feature-gated and validated in canonical runtime contracts; parity/availability remains environment-dependent by design. |
| JEPA readiness | **Done for onboarding reference path; production-blocked overall** | World stage is in canonical pipeline with typed readiness/state/provenance and structured failures; not all production semantics are complete. |
| NSR readiness | **Done as canonical stage contract path; production-blocked overall** | NSR stage wiring/readiness/failure taxonomy exists in canonical pipeline and contributes structured state/provenance. |
| LFM readiness | **Done for onboarding reference path; production-blocked overall** | LFM is in canonical stage order with validation and slot provenance; Burn path is still a bounded minimal runtime. |
| Canonical E2E reference path | **Done** | `build_onboarding_reference_backend` + `InMemoryComputeService` integration test preserves canonical pipeline surface and accounting linkage. |
| Validation / structured failures / provenance | **Done (load-bearing)** | Canonical admission + execution emit typed failures, validation summary/state, executed stages, and model-slot provenance. |
| Bounded compute service (lifecycle/admission/scheduling/worker/accounting/observability) | **Done (minimal service scope)** | Submit/admit/reject/queue/run lifecycle, bounded scheduler, worker IPC path, accounting summary, lifecycle event stream and smoke/integration tests are in place. |

## 2) Technical completion criteria for "Real Compute Onboarding complete"

Onboarding is considered complete **only if all criteria below hold in-repo**:

1. **Canonical pipeline path exists and is runnable** via `runtime/ucf-compute` with fixed stage ordering and canonical request/result contract.
2. **Artifact/manifest/compatibility path is operational**: canonical manifest default is `models/manifest.toml`; slot verification and compatibility failures are surfaced as typed failures.
3. **A real E2E onboarding reference path exists** and is executed through tests using `build_onboarding_reference_backend` and the compute service wrapper.
4. **Primary runtime path is explicit**: Burn onboarding path is first-class in canonical backend routing (not delegated to legacy compatibility wrappers).
5. **Validation + structured failure + provenance semantics are present** in canonical admission/execution outputs.
6. **Bounded compute service is minimally functional**: lifecycle, admission, bounded scheduling, execution (local + worker IPC path), accounting, and observability events are present and covered by tests.

## 3) Hard onboarding blockers (only if load-bearing)

**Current result:** no remaining hard blocker against the criteria above.

The following items remain blockers for broader production-scale compute, but **not** blockers for onboarding completion:

1. No durable queue/recovery semantics across process restart.
2. No distributed multi-worker fleet orchestration/placement.
3. No external operator telemetry/alerting integration.

## 4) Transition decision: onboarding -> real compute stack

**Decision:** transition point is reached now.

**Technical definition of the transition point:**
- All onboarding completion criteria in section 2 are satisfied in the current repository state, and
- Remaining gaps are now stack-expansion concerns (durability, distributed execution, platform integration), not canonical pipeline onboarding gaps.

Therefore, next work should be tracked as **Real Compute Stack expansion**, not further onboarding.

## 5) First expansion path after onboarding

**Selected first path:** **remote/multi-worker execution with durable scheduling state**.

Why this path is the best immediate continuation from current repo state:
- The repo already has a bounded service abstraction with queue + scheduler + worker IPC execution mode.
- The highest load-bearing gap is loss of queue/running-job state on restart.
- Multi-worker placement builds directly on existing `JobExecutionPath` / scheduler concepts without introducing a second compute graph.

Scope boundary for the first expansion slice:
1. durable queue + job state persistence and crash-safe recovery,
2. worker pool registration + deterministic placement policy,
3. replay-safe lifecycle/accounting continuity after restart.

## 6) Practical use of this checkpoint

Use this checkpoint as the technical starting contract for the next prompt series:
- treat onboarding as complete for the canonical bounded reference scope,
- treat remaining work as real compute stack build-out,
- avoid reopening onboarding architecture decisions unless new repo evidence contradicts section 1.
