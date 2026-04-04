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

## Honest status by requested components

- **JEPA**: implemented deterministic core and runtime integration path; currently lightweight/fixture-style rather than production learned model weights.
- **NSR**: implemented core and policy reasoning plumbing in dedicated crate; not the bottleneck for model artifact onboarding.
- **LFM**: implemented runtime kernel and contracts in `runtime/ucf-compute`, with fixture-driven and gated paths.
- **Burn**: runtime-side seam exists with deterministic implementations/skeleton behavior; host-ABI-side adapter in `domains/ai-backends` is placeholder.
- **Candle**: runtime-side seam exists with deterministic implementations and model-store integration; host-ABI-side adapter in `domains/ai-backends` is placeholder.

## Immediate follow-up tasks (next, concrete)

1. **Canonical model pipeline architecture**
   - Keep all runtime model pipeline expansion in `runtime/ucf-compute` capability/stage modules.
   - Restrict `domains/ai*` changes to ABI/compatibility concerns.
2. **Artifact resolution / compatibility**
   - Standardize tooling/docs to `models/manifest.toml` as single canonical path.
   - Keep env override `UCF_MODEL_MANIFEST` for explicit compatibility only.
3. **Burn as primary runtime path (incremental)**
   - Promote burn runtime stages from fixture/skeleton behavior toward verified slot-backed execution under existing safety bounds.
4. **Candle as backend seam**
   - Keep candle runtime seam parity with burn on model-store contracts and deterministic degradation semantics.
5. **Bounded compute service afterwards**
   - After runtime stage parity and artifact path stabilization, wire bounded service surface on top of canonical runtime pipeline.
