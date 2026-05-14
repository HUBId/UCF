# AI Model Pipeline Status (Phase A canonicalization)

## Scope and decision

This status file fixes the canonical architecture for Real Compute Onboarding based on repository code paths.

1. Canonical runtime model pipeline: `runtime/ucf-compute`
2. Compatibility/ABI layer (non-canonical runtime path):
   - `domains/ai`
   - `domains/ai-host-abi`
   - `domains/ai-backends`
3. Canonical manifest path for runtime model loading: `models/manifest.toml`

## Canonical surface role of this file

This file is the status/transition surface for the current repo state.

It is intentionally paired with:

- technical reference + lane map:
  - `docs/real_compute_reference_surface_v1.md`
  - `runtime/ucf-compute/src/reference_map.rs`
- readiness classification:
  - `docs/real_compute_readiness_sweep_v26.md`

This file reports repo-truth status and blockers, but does not redefine runtime contracts.

### Compute surface status (canonical vs legacy/internal)

- **Canonical compute package path (not a production claim)**: `runtime/ucf-compute` via the bounded Burn onboarding lane
  (`build_onboarding_reference_backend` / `build_canonical_production_backend` names are historical/API names, not current production-readiness evidence).
- **Canonical reference-map authority**: `runtime/ucf-compute/src/reference_map.rs`
  (`CANONICAL_COMPUTE_REFERENCE_MAP`) classifies historical/API lane names, expert, diagnostics/evidence,
  and internal/legacy lanes as one code-pinned map; those names must be read through the Prompt 23 overclaim guard.
- **Compatibility/dev lanes**: `build_backend(kind=stub|candle)` and `domains/ai*` adapter crates.
- **Internal-only lane**: `build_backend(kind=worker)` for process-isolated worker execution wiring.
- **Removed legacy entry aliases**: `cpu_stub`, `candle_dummy`, `burn_dummy`, `worker_v1` are no longer accepted backend names.

### Shared-core invariant anchor

Across production, expert/runtime-control, diagnostics/evidence, and internal/legacy lanes, one
invariant remains fixed: canonical request/job/run contracts and canonical result/failure semantics
stay authoritative and are not redefined by compatibility or diagnostic lanes.

### Prompt 23 overclaim guard

This status file is retained as a current/planning status surface, but the compute claim boundary is now the Prompt 23 taxonomy: stub fixture, toy golden, optional-real compile-only, remote/external compile-only, optional-real runtime deferred, and production claim forbidden. Names such as Burn, Candle, LFM, and LLM identify feature/backend families; they do not by themselves prove runtime inference or production readiness. Compile gates prove compilation only. Minimal Spine v1.x remains independent of compute.

## Inventory and gap matrix (repo-truth only)

| Area | real implementiert | scaffolded / placeholder | dokumentiert, aber nicht implementiert |
|---|---|---|---|
| `core/crates/ucf-jepa` | Deterministic `WorldModel` trait + `JepaCore` tick/state/prediction/surprise commit flow; `MockWorldModel` exists. | None in this crate. | None identified in this crate. |
| `core/crates/ucf-nsr` | `NsrCore`, rule/fact/trace plumbing, backend config types, policy ecology exports, mock reasoner. | Optional `nsr_datalog`/`nsr_smt` backends are feature-gated (not default runtime path). | None asserted here. |
| `core/crates/ucf-ssm` | Deterministic fixed-point-style selective scan state/input/output flow with bounded params and commits. | None in this crate. | None asserted here. |
| `runtime/ucf-compute` | Runtime pipeline surface: world model, SAE extractor, SSM, LFM, orchestration, model store (allowlist/hash/max_bytes/pinning), capability wiring, stage contracts. Candle/Burn feature paths exist in the runtime crate, but current claims remain stub fixture, toy golden, or optional-real compile-only unless a local artifact-backed runtime fixture proves otherwise. Minimal bounded compute service layer is wired on top with lifecycle, admission, scheduler, worker-path binding, accounting summary, and service-level observability. | Several paths are intentionally bounded/degraded fixtures and controlled stubs (e.g., burn v0 skeleton behavior and fixture-driven kernels). | Production-grade compute stack and production compute claims remain downstream and forbidden for current lanes. |
| `domains/ai-host-abi` | Host ABI structs, bounded output contract, commit functions, `AiBackend` trait, `MockBackend`. | Real tensor-model backend logic is not in this crate. | None beyond ABI-level docs. |
| `domains/ai` | Host runtime wrapper (`AiHostRuntime`) around ABI backend trait; tests for mock coherence behavior. | Depends on mock/adapter behavior for actual inference. | None beyond wrapper scope. |
| `domains/ai-backends` | Feature-gated module seams for `ai-candle` / `ai-burn`. | Candle/Burn adapters currently TODO placeholders returning empty bounded outputs. | Backend roadmap requests tensor I/O + hooks that are not yet implemented in this crate. |
| `models/manifest.toml` | Populated slot-oriented manifest used as canonical source after this Phase A change. | Slots are mostly disabled by default (expected for offline-safe default). | None. |
| `models/MANIFEST.toml` | Legacy alternate filename present in repo. | Effectively non-canonical for runtime default after this phase. | Historical docs mention mixed casing; canonical path is now fixed to lowercase. |
| `docs/roadmap/AI_STACK.md` + `docs/roadmap/AI_BACKENDS.md` | Updated to reflect canonical runtime path + compatibility-layer role. | Prior wording had ambiguity about primary path. | Future milestones remain roadmap-only until implemented. |

## Canonical Burn runtime status (repo-truth, v1 narrow path)

- **Burn family package path (bounded, not production-ready)**: `UCF_COMPUTE_BACKEND=burn` resolves to `BackendPackKind::BurnToyV1` inside `runtime/ucf-compute`, not to `domains/ai-backends`.
- **Current bounded E2E evidence**: `World -> SAE -> SSM` is covered as a bounded package path with Burn-named components and verified slots where available; this is not a production runtime-inference claim.
- **LFM in Burn lane**: Burn pack slot validation and the `burn_lfm_liquid_scalar_v1` path are bounded feature behavior; treat as optional-real runtime deferred unless a pinned local artifact-backed fixture and deterministic runtime golden test prove the lane.
- **Failure semantics**:
  - artifact missing/verification/incompatible are classified as typed canonical failures before execution;
  - stage backend disabled and stage execution errors are distinguished and returned as structured canonical failures;
  - degraded core results are explicit and include stage/route/provenance.
- **Visibility/provenance**:
  - canonical response includes both configured stage order and actually executed stages;
  - backend route and model slot provenance remain attached in every canonical result.

## Immediate follow-up tasks (next, concrete)

1. **Canonical model pipeline architecture**
   - Keep all runtime model pipeline expansion in `runtime/ucf-compute` capability/stage modules.
   - Restrict `domains/ai*` changes to ABI/compatibility concerns.
2. **Artifact resolution / compatibility**
   - Standardize tooling/docs to `models/manifest.toml` as single canonical path.
   - Keep env override `UCF_MODEL_MANIFEST` for explicit compatibility only.
3. **Burn LFM hardening**
   - Keep Burn LFM on the canonical path (`runtime/ucf-compute`) and expand from scalar minimal runtime toward full tensor parity without re-introducing silent toy fallback paths.
4. **Candle as backend seam**
   - Keep candle runtime seam parity with the same artifact validation + structured failure model.
5. **Bounded compute service (minimal)**
   - ✅ Implemented as a thin service over canonical runtime pipeline with technical accounting/provenance and smoke/integration hardening.
   - Remaining transition work is limited to real-compute-stack concerns (durable queueing, distributed orchestration, operator platform wiring), not a second execution graph.

## Bounded compute service readiness checkpoint (repo-truth)

- **Canonical model pipeline onboarding-complete**: yes for the narrow reference path (`build_onboarding_reference_backend` + canonical stage order) with structured result/failure/provenance.
- **Bounded compute service minimally functional**: yes. Jobs can be submitted, admitted/rejected, queued, executed, and terminated with service-level lifecycle + accounting summary attached.
- **Load-bearing guarantees now present**:
  - technical work/budget/timing accounting at job level,
  - service lifecycle observability with completion/failure class,
  - pipeline provenance mirrored on job completion without pipeline duplication,
  - smoke and integration tests over service + canonical pipeline path.
- **Concrete blockers before transition to real compute stack**:
  1. durable/persistent queue + recovery semantics are not implemented,
  2. distributed worker/fleet orchestration and placement are not implemented,
  3. external operator telemetry/alerting platform integration is not implemented.

## JEPA world-stage readiness (repo-truth as of this change)

This section records JEPA against the canonical readiness ladder:

| readiness stage | current JEPA status | repo-truth notes |
|---|---|---|
| `scaffolded` | ✅ complete | `core/crates/ucf-jepa` already had deterministic JEPA structs/commit logic and a world boundary trait. |
| `contract-ready` | ✅ complete | `runtime/ucf-compute` world stage stays on canonical stage contracts (`StageContractVersion::V1`) and now carries explicit `previous_state_digest` input at the world boundary. |
| `artifact-ready` | ✅ complete | `world_jepa` slot is validated through canonical model store + compatibility checks and appears in `model_slots` provenance with explicit status/code (`used`, `disabled`, `unavailable`, `verification_failed`, `incompatible`). |
| `runtime-path-ready` | ✅ minimal honest path complete | Canonical `compute_canonical` now records world-stage status (`world_stage`) including predictor, slot, runtime usage, and readiness; Burn/Candle JEPA predictors are routed through the canonical world stage and consume/emit world state digest continuity. |
| `production-blocked` | ⚠️ still true overall | JEPA path is honest/minimal but not yet production-complete; blockers below remain. |

### Current JEPA blockers (3–5 concrete)

1. **No full JEPA tensor-model semantics yet for all lanes**: Burn/Candle world paths are deterministic runtime implementations, but still represent a constrained minimal JEPA lane rather than full production model parity.
2. **Cross-cycle persistence is session-local**: world state continuity is explicit in-process (`previous_state_digest`), but there is no durable/recoverable world-state persistence contract for process restarts.
3. **Candle availability remains environment/feature dependent**: Candle JEPA path is a compatible seam, but can still be unavailable when feature/runtime prerequisites are missing.
4. **Failure handling is now typed but still pipeline-local**: JEPA-specific failure classes are represented in canonical failure/provenance, but downstream operator automation and runbook wiring are not fully specialized yet.

## LFM readiness (repo-truth as of this change)

| readiness stage | current LFM status | repo-truth notes |
|---|---|---|
| `scaffolded` | ✅ complete | Canonical LFM stage structs/contracts exist (`LfmInput`, `LfmOutput`, `LfmValidatorV1`) and are part of the fixed stage sequence. |
| `contract-ready` | ✅ complete | LFM uses the same canonical `StageContractVersion::V1` boundary as world/sae/ssm and is validated in the canonical request→stage→result path. |
| `artifact-ready` | ✅ minimal honest path complete | Burn pack treats `lfm` as required slot and maps slot state into structured runtime/provenance (`used`, `disabled`, `unavailable`, `verification_failed`, `incompatible`). |
| `runtime-path-ready` | ✅ minimal honest path complete | Burn pack runs `World -> Sae -> Ssm -> Lfm` with `BurnLfmV1` route metadata and emits explicit LFM stage diagnostics (`state`, `readiness`, `runtime`, slot provenance). |
| `production-blocked` | ⚠️ still true overall | Runtime path is real and typed, but still a constrained minimal burn scalar path (not full production tensor parity). |

### Current LFM blockers (3–5 concrete)

1. **Burn LFM runtime is minimal scalar path**: current Burn kernel is a deterministic scalar runtime path, not yet full tensor-runtime parity with broader model expectations.
2. **No dedicated burn-typed LFM weight loader yet**: LFM slot is hash/manifest verified, but there is no dedicated Burn LFM tensor schema loader equivalent to mature world/sae/ssm weight spec flows.
3. **Candle/Burn LFM parity remains partial**: Candle and LNN lanes exist, but backend-level runtime parity and compatibility diagnostics are still uneven across lanes.
4. **Operator-grade failure automation is pending**: canonical failures are typed, but runbook automation for LFM-specific remediation (artifact class → operator action) is not fully wired.

## Transition checkpoint

The canonical onboarding->stack transition decision is recorded in `docs/roadmap/REAL_COMPUTE_TRANSITION.md`.

## Historical naming note

`Phase A canonicalization` in the title is historical naming. The active interpretation is current
real-compute stack status and transition framing as documented above.
