# AI Model Pipeline Status (Phase A canonicalization)

## Scope and decision

This status file fixes the canonical architecture for Real Compute Onboarding based on repository code paths.

1. Canonical runtime model pipeline: `runtime/ucf-compute`
2. Compatibility/ABI layer (non-canonical runtime path):
   - `domains/ai`
   - `domains/ai-host-abi`
   - `domains/ai-backends`
3. Canonical manifest path for runtime model loading: `models/manifest.toml`

## Inventory and gap matrix (repo-truth only)

| Area | real implementiert | scaffolded / placeholder | dokumentiert, aber nicht implementiert |
|---|---|---|---|
| `core/crates/ucf-jepa` | Deterministic `WorldModel` trait + `JepaCore` tick/state/prediction/surprise commit flow; `MockWorldModel` exists. | None in this crate. | None identified in this crate. |
| `core/crates/ucf-nsr` | `NsrCore`, rule/fact/trace plumbing, backend config types, policy ecology exports, mock reasoner. | Optional `nsr_datalog`/`nsr_smt` backends are feature-gated (not default runtime path). | None asserted here. |
| `core/crates/ucf-ssm` | Deterministic fixed-point-style selective scan state/input/output flow with bounded params and commits. | None in this crate. | None asserted here. |
| `runtime/ucf-compute` | Full runtime pipeline surface: world model, SAE extractor, SSM, LFM, orchestration, model store (allowlist/hash/max_bytes/pinning), capability wiring, stage contracts. Candle/Burn compute paths exist in runtime crate. | Several paths are intentionally bounded/degraded fixtures and controlled stubs (e.g., burn v0 skeleton behavior and fixture-driven kernels). | Bounded compute service rollout is downstream work (not completed in this phase). |
| `domains/ai-host-abi` | Host ABI structs, bounded output contract, commit functions, `AiBackend` trait, `MockBackend`. | Real tensor-model backend logic is not in this crate. | None beyond ABI-level docs. |
| `domains/ai` | Host runtime wrapper (`AiHostRuntime`) around ABI backend trait; tests for mock coherence behavior. | Depends on mock/adapter behavior for actual inference. | None beyond wrapper scope. |
| `domains/ai-backends` | Feature-gated module seams for `ai-candle` / `ai-burn`. | Candle/Burn adapters currently TODO placeholders returning empty bounded outputs. | Backend roadmap requests tensor I/O + hooks that are not yet implemented in this crate. |
| `models/manifest.toml` | Populated slot-oriented manifest used as canonical source after this Phase A change. | Slots are mostly disabled by default (expected for offline-safe default). | None. |
| `models/MANIFEST.toml` | Legacy alternate filename present in repo. | Effectively non-canonical for runtime default after this phase. | Historical docs mention mixed casing; canonical path is now fixed to lowercase. |
| `docs/roadmap/AI_STACK.md` + `docs/roadmap/AI_BACKENDS.md` | Updated to reflect canonical runtime path + compatibility-layer role. | Prior wording had ambiguity about primary path. | Future milestones remain roadmap-only until implemented. |

## Canonical Burn runtime status (repo-truth, v1 narrow path)

- **Primary runtime path**: `UCF_COMPUTE_BACKEND=burn` resolves to `BackendPackKind::BurnToyV1` inside `runtime/ucf-compute`, not to `domains/ai-backends`.
- **Honest minimal E2E path (real today)**: `World -> SAE -> SSM` runs with Burn components and verified model slots (`world_jepa`, `sae`, `ssm`).
- **LFM in Burn lane**: Burn pack now requires a verified `lfm` slot and routes LFM through a dedicated Burn runtime kernel (`burn_lfm_liquid_scalar_v1`) in the canonical pipeline path.
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
5. **Bounded compute service afterwards**
   - After runtime stage parity and artifact path stabilization, wire bounded service surface on top of canonical runtime pipeline.

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
