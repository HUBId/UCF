# ucf-compute v0 pipeline

Deterministic offline compute pipeline used by the runtime.

## Real-compute reference layer

For a compact, repo-based reference map of the canonical kernel, extension
paths, and diagnostic/test-near seams, see
`docs/real_compute_reference_surface_v1.md`.

## Capability model

The top-level runtime contract stays `AiComputeBackend`, but concrete backends are now composed from stable subtraits in `src/capabilities.rs`:

- `WorldModelPredictor` (JEPA/world prediction)
- `FeatureExtractor` (SAE/sparse spike extraction)
- `WorkingMemoryModel` (SSM/selective scan memory)
- `LlmInference` + `LlmOutput` placeholder (defined for future policy/model integration)

`ComputePipelineBackend` orchestrates these capabilities with bounded deterministic degradation.

## Profile mapping (factory)

`build_backend` wires `ComputeBackendKind` into runtime packs:

- `stub` (**compat/dev lane**): `BackendPackKind::ToyV1`
- `candle` (**compat seam**, `--features compute-candle`): `BackendPackKind::CandleToyV1`
- `burn` (**canonical production lane**, `--features compute-burn,backend-burn`): `BackendPackKind::BurnToyV1`
- `worker` (**internal execution lane**): `BackendPackKind::WorkerV1`

Legacy aliases (`cpu_stub`, `candle_dummy`, `burn_dummy`, `worker_v1`) are no longer accepted in
`UCF_COMPUTE_BACKEND`; use canonical names only.

### Canonical onboarding lane (single reference path)

- The canonical onboarding entrypoint is now pinned to Burn via
  `build_onboarding_reference_backend`
  / `build_canonical_production_backend` (`CANONICAL_ONBOARDING_PACK = BurnToyV1`).
- Canonical request/result/failure contract stays:
  `CanonicalPipelineRequest -> ComputePipelineBackend::compute_canonical -> CanonicalPipelineResult|CanonicalPipelineFailure`.
- Canonical stage sequence is fixed as `World -> SAE -> SSM -> LFM`
  (`CANONICAL_STAGE_SEQUENCE`), with honest runtime state per stage:
  - required productive core: `World`, `SAE`, `SSM`;
  - `LFM` runs when Burn LFM runtime is enabled (`lfm-burn`), otherwise backend init is
    explicitly blocked (`BackendDisabled`) — no silent fallback lane.
- `NSR` remains an optional attachment and is surfaced explicitly in `CanonicalPipelineResult.nsr_stage`
  (`disabled`, `used`, `contract_mismatch`, `verification_failed`, etc.).
- `Candle` remains a compatibility seam and is **not** a second onboarding default path.

See also `docs/compute_onboarding_reference_path.md` for the compact readiness matrix.
For the hard, repo-based closure matrix of Distributed Execution Hardening (Serie A), see
`docs/distributed_execution_serie_a_closure_v1.md`.

### Expert runtime entry contracts (Serie F)

The service surface keeps one canonical submit path and now models high-trust usage
as explicit entry contracts instead of side-entry behavior:

- `RuntimeEntryClass::standard_canonical`
  - canonical submit + snapshot-safe ops.
  - does **not** support replay or scheduler-drain operations.
- `RuntimeEntryClass::expert_high_trust`
  - technical replay and runtime-ops entry (`replay_with_entry`, `run_operation_with_entry`).
  - guarantees explicit contract metadata in outcomes (`entry_class`, `contract_shape`,
    `contract_safety`).
- `RuntimeEntryClass::internal_dev_test`
  - internal control/test surface on the same contract model.

No auth/role/tenant platform is introduced here. The contract distinction is runtime-technical:
standard-safe vs high-trust vs internal-only semantics with explicit unsupported outcomes.

Shared surface core (drift guard, Serie F Prompt 6):

- shared core status terms:
  - snapshot consistency: `current | partial | stale | drift_affected | unavailable`
  - diagnostics availability core: `available | partial | unavailable | blocked`
  - stale/drift runtime view:
    - freshness: `current | partial | stale`
    - drift: `none | drift_suspected | inconsistent_needs_refresh`
    - source-scoped drift signals are surfaced from worker/placement, warmup/readiness,
      rollout-vs-snapshot, and replay-basis checks.
- explicit extension seam:
  - `internal_only` stays an internal extension (`ExpertDiagnosticsAvailability::InternalOnly`)
    and is intentionally not folded into productive core diagnostics states.
- shared entry mapping:
  - `RuntimeEntryClass` now owns canonical mapping for replay/runtime-ops shape + safety
    (`replay_contract_shape`, `runtime_ops_contract_shape`, `contract_safety`), so standard,
    expert, and internal paths reuse one mapping source.
- shared action-result alignment:
  - runtime-ops outcome code and mutation result are checked against a single core semantic
    compatibility rule (`accepted/completed/no_op/blocked/failed/unsupported` vs mutation result),
    reducing silent contract drift across entry paths.

For the hard, repo-based closure matrix of Expert Runtime Surface / API Hardening (Serie F), see
`docs/ops/serie_f_expert_runtime_surface_closure.md`.

### Expert ops actions / controlled runtime interventions (Serie F Prompt 2)

`CanonicalComputeEntryPoint::run_operation_with_entry` now keeps a narrow expert-ops surface with
explicit action class + scope + result semantics:

- action classes:
  - `read_only`
  - `controlled_mutating`
  - `internal_dev_test_only`
- scopes:
  - `runtime_status`
  - `worker_readiness`
  - `replay_history`
- result codes:
  - `accepted`
  - `completed`
  - `no_op`
  - `blocked`
  - `failed`
  - `unsupported` (runtime mode/context cannot provide this action)

Canonical actions in this layer:

- `snapshot` (read-only runtime status)
- `drain_scheduler` (controlled mutating worker/readiness intervention; blocked on
  `standard_canonical` entry)
- `rehydrate_history` (controlled mutating replay/history rehydrate trigger; blocked without
  history store)
- `refresh_runtime` (explicitly unsupported for the in-memory bounded runtime)
- `internal_clear_replay_regression` (internal/dev/test-only operation)

Ops provenance stays in the runtime surface itself (`RuntimeOpsSnapshot.recent_operations`) so
interventions are visible without introducing an admin/control-plane subsystem.

Stale/drift guardrails for expert mutations:

- mutating `rehydrate_history` operations are now blocked whenever stale/drift view requires
  refresh (`needs_refresh=true`), not only for stale replay snapshots.
- block details explicitly include freshness, drift class, and primary source to separate:
  - stale/partial diagnostic basis
  - drift/inconsistency context
  - normal runtime failure paths.

## Execution-device classes (bounded service placement)

`MultiWorkerComputeService` now keeps a narrow execution-device layer for placement:

- `cpu`: in-process/local execution units.
- `worker`: isolated worker execution units (`worker_ipc` path).

Per candidate, placement tracks backend suitability and device suitability separately:

- backend: `suitable|incompatible|disabled|unavailable`
- device: `suitable|unsuitable|disabled|unavailable`

Serie E tightens this with a narrow capability-contract view per assessed path:

- support: `supported|supported_with_constraints|unsupported`
- stage/path view (per `world|sae|ssm|lfm` segment):
  `supported|supported_with_constraints|degraded_only|fallback_only|unsupported`
- constraints (minimal, load-bearing only):
  - `only_local` / `only_remote_worker`
  - `warm_ready_preferred`
  - `guarded_degraded_usage`
  - `capacity_or_cold_start_caveat`

The contract is intentionally path-bound and technical:

- it is evaluated on the effective execution path (`local_canonical` vs `worker_ipc`);
- it now also records the dominant stage/path caveat per candidate, so "path valid but
  constrained by one stage" is explicit in placement/replay provenance;
- it uses stage/contract admission failures as hard blockers (`unsupported`) instead of
  optimistic backend claims;
- it keeps constrained vs blocked distinct so placement can prefer full support, still execute
  constrained paths when serviceable, and reject blocked paths deterministically.

This is intentionally technical and minimal. The repo does **not** introduce GPU vendor/driver
inventory or hardware orchestration in this layer.

Worker snapshots now expose a narrow registry/health signal set:
- registry identity: worker id + class (`local_primary|remote_secondary`) + role (`primary|secondary`);
- runtime health status:
  `known|ready|busy|saturated|degraded|unavailable|stale|unknown|unhealthy`;
- last health-contact timestamp, optional cooldown/quarantine-until timestamp, and
  last dispatch/error metadata.

Dispatch candidacy is tied to those health states: `degraded`, `stale`, `unknown`,
`unavailable`, `saturated`, and `unhealthy` units are explicitly skipped and reflected in
placement candidate diagnostics.

Multi-worker scheduling remains intentionally compact: jobs are either placed immediately, kept
queued as currently-unschedulable (capacity/device temporarily unavailable), or rejected when no
technical backend/device placement is possible.

Selection/ranking now uses this same contract:

- `supported` paths rank ahead of `supported_with_constraints`;
- constrained paths remain eligible and explicit in decision provenance;
- blocked paths stay outside admissible placement.

### Warmup/readiness-aware placement (Serie C Prompt 3)

Placement candidates now carry a narrow warmup/readiness context derived from required model-slot
rollout details:

- `warm_ready`
- `prepared`
- `cold_runnable`
- `stale_prepared`
- `blocked_unavailable`

The scheduler keeps this coupling intentionally small:

- `blocked_unavailable` candidates are treated as not currently placeable and can be deferred.
- Placeable candidates are ranked by warmup state first (warm > prepared > cold > stale), then by
  lane preference.
- Candidate diagnostics include a deterministic `cold_start_penalty_units` hint so runtime
  decisions can explain why a cold path was still selected (for example due to current
  capacity/readiness constraints).

This is **not** a global warmup/caching orchestration system; it is a local placement signal that
feeds existing admission/capacity/ops diagnostics.

### Backend/device degradation + fallback semantics (Serie E Prompt 5)

Placement now keeps a narrow backend/device degradation view per candidate and selected path:

- `healthy_support`
- `constrained_serviceable`
- `degraded_path`
- `fallback_preferred`
- `degraded_fallback_used`
- `blocked_unusable`
- `generic_compute_failure`

This view is intentionally technical and derived from existing signals only:

- stage-path support (`supported_with_constraints|degraded_only|fallback_only|unsupported`),
- warmup/readiness state (`warm_ready|prepared|cold|stale|blocked`),
- backend lane pressure (`burn` vs guarded `candle` fallback lane),
- explicit admission blockers vs generic semantic compute failures.

Fallback semantics now stay tied to that backend/device view:

- candle-only serviceable paths are marked `fallback_preferred`,
- once selected/used they become `degraded_fallback_used`,
- worker->local redispatch fallback is also marked `degraded_fallback_used`,
- hard backend/device incompatibility remains `blocked_unusable`.

Placement, rollout-context snapshots and replay caveats continue to reuse the same compact
provenance surfaces (`backend_device_readiness_context`, degraded/fallback stage counters, and
reason strings) instead of introducing a separate reliability/incident platform.

### Distributed admission / placement consistency (Serie A Prompt 5)

Admission and placement now share one worker-crossing diagnostic summary
(`ExecutionPlacement.distributed`) derived from the same candidate assessments for local and
remote units.

Canonical distributed states:

- `admissible_and_placeable`: request is technically admissible and currently placeable.
- `admissible_placeable_on_subset`: admissible/placeable only on a subset of units.
- `admissible_but_currently_unschedulable`: admissible in theory, blocked by readiness/capacity now.
- `admissible_degraded_only`: placeable only through degraded fallback lane (for example candle fallback).
- `blocked_incompatible`: no admissible worker/backend/device combination.

The summary also reports:

- locality scope (`none|local_only|remote_only|local_and_remote`);
- admissible unit set (principally eligible units);
- currently placeable unit set;
- whether degraded fallback is currently possible.

This keeps admission-vs-placement mismatches explicit (for example, admitted but no currently
placeable worker) without introducing a global optimization scheduler.

### Distributed degradation / recovery semantics (Serie A Prompt 7)

`MultiWorkerComputeService` now also exposes a narrow `distributed_recovery_snapshot()` with one
canonical degradation state for the active runtime:

- `healthy`
- `partially_degraded`
- `constrained_but_serviceable`
- `recovery_in_progress`
- `unrecoverable_unavailable`

Interpretation is intentionally worker+service scoped (not cluster-orchestrator scope):

- Worker snapshots expose placement eligibility (`placement_eligible`) and a normalized degradation
  state (`degradation_state`) alongside runtime status.
- Unstable workers are excluded from placement candidacy while constrained-but-serviceable workers
  remain dispatchable.
- Recovery becomes visible when a previously degraded/unavailable worker returns and is marked with
  `recovered_at_unix_ms`; the service reports `recovery_in_progress` during this bounded window.
- Snapshot counters include uncertain and recovery-required in-flight jobs so redispatch/wait/orphan
  pressure remains visible without adding a reconciliation platform.

### Remote execution consistency / provenance / replay fidelity (Serie A hardening)

Remote (`worker_ipc`) execution stays on the same canonical top-level contract as local execution:
`CanonicalPipelineRequest -> compute_canonical -> CanonicalPipelineResult|CanonicalPipelineFailure`.
No parallel remote-only result/fault model is introduced.

Load-bearing remote context is now carried in history/replay surfaces as a bounded summary:
- execution path (`LocalCanonical` vs `WorkerIpc`);
- execution lane;
- backend route + model-slot summary;
- resource class + capacity pressure context.

Replay now classifies remote-context reproducibility explicitly:
- `exact`: remote replay kept equivalent worker-context signals;
- `partial`: replay completed but remote-context signals drifted;
- `missing`: remote fidelity cannot be established under current runtime context;
- `not_applicable_local`: source run was local.

When historical remote records lack required context fields, replay is blocked with a structured
`missing_remote_execution_context` failure instead of silently claiming equivalence.

### Rollout-aware replay/comparison context (Serie D hardening)

Replay and baseline-compare now carry a narrow rollout context view so before/after activation
checks do not claim blind equivalence:

- rollout context class:
  - `active_or_warm`
  - `guarded_or_candidate`
  - `fallback_or_rollback`
  - `mixed_or_unknown`
  - `unavailable`
- rollout comparability class:
  - `comparable_across_rollout_boundary`
  - `comparable_with_rollout_caveat`
  - `not_meaningfully_comparable_across_rollout_boundary`
  - `blocked_insufficient_rollout_context`
  - `blocked_changed_execution_context_beyond_useful_comparison`

The context is populated from persisted execution snapshots when available and from live slot
warmup summaries for in-memory records. This keeps rollout-aware replay tied to existing
history/replay/runtime surfaces and avoids separate experiment/release analytics planes.

## Resource classes and capacity accounting (runtime scope)

Capacity is modeled as a narrow runtime signal (not a cluster manager):

- Resource classes: `light`, `standard`, `heavy` (derived from canonical `global_work_units`).
- Class weights: `1`, `2`, `3` capacity units respectively.
- Each execution unit exposes `max_parallel_jobs * 2` capacity units.

Scheduler/admission behavior uses these signals to distinguish:

- admitted + queued due to capacity pressure,
- deferred due to transient capacity saturation,
- rejected as currently not supportable under class/capacity constraints,
- placement fallback/degradation decisions under capacity pressure.

Runtime/job provenance now includes resource class, queue/reject capacity disposition, and
capacity pressure (`nominal|saturated|overloaded`) so ops/history can separate scheduling-capacity
decisions from execution failures.

### Consolidated work/cost runtime signals (Serie C)

Runtime now keeps one narrow `ConsolidatedWorkCostSummary` across scheduling/accounting/history:

- **provenance-aware** (`estimated_from_budget` vs `runtime_measured`) so consumers can distinguish
  admission-time estimates from measured run summaries;
- **job-level summary** (estimated work, consumed/remaining work when available, pressure,
  queue/disposition);
- **stage/hotspot hook** (dominant stage + share, degraded stage count) without introducing a new
  profiling platform;
- **stage-cost attribution hook** (`diagnostics.stage_cost_attribution`) with explicit provenance
  (`measured_timing` vs `derived_from_budget_and_meter`), dominant timing/work flags, and narrow
  runtime pattern classes (`slow_but_healthy`, `dominant_cost_driver`, `degraded_path_driver`,
  `skipped_or_fallback`, `hard_failure`);
- **failure/degradation tension semantics** (`expensive_but_successful`,
  `expensive_and_degraded`, `retried_with_additional_cost`, `low_cost_but_blocked`) for load-bearing
  diagnostics.

Scheduling/placement uses the same signals for queue/defer/reject and fallback outcomes, and
pressure snapshots now expose queued `light|standard|heavy` counts so capacity pressure can be tied
to job-class work mix.

### Runtime optimization feedback loop (Serie C Prompt 5)

`MultiWorkerComputeService` now derives one **narrow technical feedback view** from recent
runtime outcomes (bounded lookback, deterministic reduction, no adaptive scorer):

- repeated cold path penalties (`cold_runnable|stale_prepared` and cold-start units);
- repeated degraded placement/fallback outcomes;
- repeated worker/path pressure (`constrained|saturated|backpressured|temporarily_unschedulable`);
- repeated retry/redispatch cost (`attempts > 1` / local redispatch);
- repeated dominant hotspot stage from runtime-measured work summaries.

The feedback is intentionally typed as:
`strong|weak|stale|contradicted|insufficient`.

Scheduling/placement can use it **only as a bounded hint**:

- prefer a proven warm unit under similar current suitability;
- avoid a repeatedly degraded unit when an equivalent suitable alternative exists;
- surface cold-start repetition as prewarm signal context (without forcing a gate);
- keep normal fallback behavior whenever feedback is weak/stale/contradicted/insufficient.

Important boundary:
feedback does **not** replace admission/readiness/capability/compatibility gates. It only acts
inside already admissible placement space, and decision provenance remains visible in placement
decisive signals plus per-job feedback view.

### Cold-path minimization for productive reference paths (Serie C Prompt 7)

Cold-path handling remains a **narrow runtime hint layer** and is now extended with explicit
reference-path context:

- each placement candidate now carries:
  - `effective_reference_path` (`execution_kind:lane`, deterministic),
  - `reference_path_class` (`active_production|guarded_active|candidate|compare_shadow|unknown`),
  - `cold_start_sensitive` (cold/stale/blocked warmup context);
- feedback marks repeated cold behavior not only per unit, but also per effective reference path,
  plus repeated cold on `candidate|guarded_active` paths;
- placement ranking keeps existing suitability gates, but now biases active production reference
  paths before candidate/compare paths when technically equivalent.

Decision provenance for ops/history stays explicit via `placement.decisive_signals`, including:

- `reference_path=...`
- `reference_path_class=...`
- `cold_path_decision=warm_path_preferred_and_used|warm_path_preferred_but_unavailable|cold_path_unavoidable|preparation_warmup_insufficient|cold_start_penalty_accepted_due_to_stronger_constraints`

This is intentionally **not** a warmup/caching platform: no global prewarming, no dynamic policy
engine, no separate optimization control-plane.

### Specialization-aware placement refinement (Serie E Prompt 7)

Placement keeps the existing bounded candidate model, but now applies a small support-class pass
before tie-breakers:

- `fully_supported`: suitable candidate with healthy support and no stage/path caveat pressure.
- `constrained_acceptable`: suitable candidate with warmup/readiness or stage/path constraints.
- `degraded_fallback`: technically suitable but only via degraded/fallback specialization path.
- `blocked`: unsuitable/non-placeable candidate.

Canonical heuristics stay narrow and deterministic:

- prefer `fully_supported` over `constrained_acceptable` when otherwise equivalent;
- prefer `constrained_acceptable` over `degraded_fallback` when a full path is unavailable;
- use degraded fallback only when no supported path remains viable;
- keep blocked paths outside acceptable placement.

Stage/path caveats are operational (not just diagnostic): selection now emits explicit provenance for
stage-constrained alternatives and support-class-driven acceptance, e.g.
`selected_path_over_stage_constrained_alternative=true`,
`constrained_accepted_due_to_missing_supported_path=true`, and
`degraded_accepted_due_to_missing_supported_path=true`.

Boundaries remain unchanged: no global optimization engine, no hardware scheduler, no autoscaling
control-plane.

For the hard, repo-based closure matrix of Device/Backend Specialization Hardening (Serie E), see
`docs/ops/serie_e_device_backend_specialization_closure.md`.

## Backend selection (runtime)

The orchestrator can be bootstrapped from env config via `RuntimeOrchestrator::try_new_from_env`.

- `UCF_COMPUTE_BACKEND=stub|candle|burn|worker`
- `UCF_COMPUTE_SEED=<u64>`
- `UCF_COMPUTE_MAX_MICROS=<u64>`
- `UCF_COMPUTE_HARD_TIMEOUT_MICROS=<u64>`

Default remains `stub` when env vars are unset (compatibility/dev-safe default).
Production callers should set `UCF_COMPUTE_BACKEND=burn` explicitly or call
`build_canonical_production_backend`.

## Candle feature extractor v0 (offline dummy weights)

`compute-candle` enables `CandleFeatureExtractor`, which performs a deterministic forward pass (`32 -> 64`) on CPU-only candle tensors using inline dummy weights.

- No HTTP, no model downloads, no external fixture pulls.
- Input vector is derived from `ComputeInput.context_digest` + world prediction digest.
- Reductions (`top-k`, sparsity, energy) are done in Rust over `Vec<f32>` for deterministic ordering.

## Offline fixture policy and constraints

- No network and no model-weight download.
- Output deterministic from `(context_digest, seed, t)`.
- Bounded outputs: capped spikes/notes and digest-only persistence for large vectors/state.

## Model manifest source

- Canonical manifest path: `models/manifest.toml`.
- Override path only via `UCF_MODEL_MANIFEST` when explicit compatibility behavior is required.

## Rollout path semantics (candidate/compare/shadow/promotion)

Runtime rollout diagnostics use a narrow canonical state set per slot path:

- `active`: selected primary hash/path for the slot.
- `candidate`: staged hash under `UCF_MODEL_CANDIDATE_<SLOT>`.
- `compare`: side-by-side compare hash under `UCF_MODEL_COMPARE_<SLOT>`.
- `shadow`: observational shadow hash under `UCF_MODEL_SHADOW_<SLOT>`.
- `disabled`: slot not configured or disabled by manifest/env.
- `blocked`: required rollout path is configured but cannot be verified.

`ModelStore::slot_path_statuses` verifies these paths against promoted artifacts and surfaces
whether a candidate/compare/shadow path is technically comparable (`verified + comparable`) or
blocked with explicit reason text.

`BackendPack` slot provenance carries this rollout digest in lifecycle details so ops/history views
can distinguish:

- active path reference,
- candidate/compare/shadow side paths,
- compare/shadow availability failures that block activation for required slots,
- compare/shadow context (`same_effective_config`, `with_caveats`, `not_comparable`, `blocked`),
- compare/shadow outcome (`compared|shadowed|inconclusive|blocked|failed_technically|not_comparable`)
  and promotion disposition (`candidate_remains_blocked|candidate_more_promotable|
  candidate_comparison_inconclusive|active_path_remains_preferred`).
- activation scope (`not_active|compare_shadow_only|guarded_active|fully_active|blocked|reverted`)
  with explicit guardrail reasons and resulting post-fallback/rollback scope.
- rollout recovery classification:
  - problem kind (`none|activation_unstable|activation_induced_degradation|
    candidate_rejected_after_activation_attempt|general_runtime_failure_no_rollout_meaning`)
  - recovery outcome (`not_needed|guarded_active|fallback_to_prior_active|rollback_completed|
    candidate_blocked|incomplete_or_blocked`).
  These keep bad activation / unstable candidate handling separate from generic runtime failures and
  show whether recovery held candidate in guarded-active, fell back, rolled back, or stayed blocked.

Recovery boundaries (narrow and intentional):
- guardrail prevention (`GuardrailPreventedWiderActivation`) stays distinct from post-activation
  instability (`ActivationBecameUnstableAfterGoingActive`);
- fallback stabilization (`FallbackStabilizedService`) stays distinct from rollback restoration
  (`RollbackRestoredPriorActive`);
- candidate-blocking remains explicit (`CandidateRemainsBlockedAfterRecovery`);
- no incident automation or release-orchestration loop is introduced.

Boundaries (intentional):

- no approval workflow/governance engine in runtime;
- no experiment/statistics suite in rollout paths;
- promotion still uses existing compatibility gates and artifact verification, now with richer
  blocked diagnostics.

## Adding future backends

To add a real backend later without refactoring orchestrator/frame contracts:

1. Implement one or more capability traits (`WorldModelPredictor`, `FeatureExtractor`, `WorkingMemoryModel`).
2. Register capability wiring in `build_backend` for a profile.
3. Keep `AiComputeBackend` entrypoint unchanged by returning `ComputePipelineBackend`.
