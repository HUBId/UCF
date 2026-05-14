# UCF Real Compute Optional Lane Inventory

## 0. Purpose

- This is an inventory for the optional Real Compute lane.
- It is not an implementation.
- It is not a production-readiness claim.
- Minimal Spine v1.x remains independent of Real Compute.
- The inventory separates stub, toy, mock, partial, optional-real, remote, and unknown lanes so later prompts can reduce naming and claim ambiguity without changing runtime behavior.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `4c4c574d29531023a1657e72719d77855de7a1da` |
| HEAD short | `4c4c574d` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Post-freeze roadmap present | yes |
| Freeze doc present | yes |
| `runtime/ucf-compute` present | yes |
| `domains/ai-backends` present | yes |
| `core/crates/ucf-ai-port` present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -15`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and file/directory presence checks for the post-freeze roadmap, freeze doc, `runtime/ucf-compute`, `domains/ai-backends`, and `core/crates/ucf-ai-port`.

Required companion documents:

- [`docs/roadmap/post_freeze_roadmap_selection.md`](post_freeze_roadmap_selection.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)
- [`docs/roadmap/compute_backend_naming_boundary_plan.md`](compute_backend_naming_boundary_plan.md)

## 2. Compute Module Inventory

| Module/path | Purpose | Current maturity | Backend/lane relevance | Tests | Risk |
|---|---|---|---|---|---|
| `runtime/ucf-compute` | Canonical compute runtime crate with backend packs, pipeline contracts, model store, bounded service, fixtures, and optional feature lanes. | mixed | Defines default stub/toy pack behavior plus Candle, Burn, LFM/LNN, worker, remote, and LLM seams. | Many crate-local unit tests in `src/*`; no `runtime/ucf-compute/tests/` directory was found; compute-specific package tests are runnable with `cargo test -p ucf-compute --all-targets`. | Names and docs include canonical/production/real wording while several executable paths are fixture, toy, stub, or compile-gated. |
| `runtime/ucf-compute/src/backend_pack.rs` | Backend pack registry, pack parsing, fixture digests, model-slot provenance, pack construction, and remote pack gating. | mixed | Contains `stub_v0`, `toy_v1`, `candle_toy_v1`, `candle_liquid_v1`, `burn_toy_v1`, `toy_lnn_v1`, `worker_v1`, and feature-gated `remote_v1`. | Unit tests cover fixture digest stability, pack parsing, feature gating, provenance, and selected feature-gated pack cases. | `burn_toy_v1` and `candle_toy_v1` names can be mistaken for real production backends unless documented as toy/fixture/optional-real-compile lanes. |
| `runtime/ucf-compute/src/backends.rs` | Service/backend builder for `stub`, `candle`, `burn`, and `worker` backend kinds. | partial | Default kind is `Stub`; `Burn` is named as canonical onboarding kind, but the pack constant is `BurnToyV1`. | Covered through crate tests and workspace tests. | Environment variable activation (`UCF_COMPUTE_BACKEND`) must remain explicit and non-default for non-stub lanes. |
| `runtime/ucf-compute/src/backends/burn_backend.rs` | Burn world/SAE/SSM runtime wrappers and deterministic stage behavior under `compute-burn`. | functional-prototype | Optional Burn compile/runtime seam, but not an external tensor-engine proof by itself. | Feature-matrix CI has non-blocking `burn-cpu`; package tests can be run with Burn features. | Overclaim risk if called production real compute without verified model artifacts and fixture-backed tests. |
| `runtime/ucf-compute/src/backends/candle_backend.rs` | Candle world/SAE/SSM wrappers under `compute-candle`. | optional-real-compile | Uses Candle dependency and model-store verification path, with fallback/hash behavior for disabled slots. | Feature-matrix CI has non-blocking `candle-cpu`; package tests can be run with Candle features. | Needs clear separation between Candle compile support, toy fixture execution, and verified real model execution. |
| `runtime/ucf-compute/src/capabilities/*llm*` | LLM capability seams for stub, toy, Candle, and Burn. | mixed | `LlmStubBackend` and toy weights are default/offline; Candle can load safetensors/tokenizer; Burn LLM currently returns `NotImplemented`. | Unit tests cover toy fixture loading; Candle/Burn paths require features. | `llm-burn` is compile-gated but not a real inference implementation; `llm-candle` can be real only with verified artifacts. |
| `runtime/ucf-compute/src/lfm.rs` | LFM toy, Candle, Burn, LNN/ODE, and plasticity-adjacent kernels. | mixed | `ToyLfmKernel` is offline deterministic; `lfm-candle`, `lfm-burn`, `lfm-lnn`, and `plasticity` add optional lanes. | Unit tests in the crate and feature-gated checks. | LFM feature names can imply realness while some lanes are toy kernels or require model-store artifacts. |
| `runtime/ucf-compute/fixtures/*` | Embedded deterministic fixture inputs and tiny weights. | toy | Supports offline toy/stub tests for LLM, JEPA, SAE, SSM, LFM, and LNN parameters. | Used by crate unit tests and package tests. | Fixtures are not production model artifacts. |
| `domains/ai-backends` | Compatibility-only adapter seams for `domains/ai-host-abi`. | stub | Feature-gated `ai-candle` and `ai-burn` modules currently return empty bounded outputs with TODO comments. | Workspace compile/tests only; no dedicated tests found. | High overclaim risk if confused with canonical runtime compute or real tensor I/O. |
| `core/crates/ucf-ai-port` | Broad AI port aggregation crate exposing optional AI runtime, bus, NSR, ODE, and digitalbrain dependencies. | partial | Feature flags are integration/port toggles, not compute backend implementations. | Workspace tests and crate compile coverage. | Features named `burn` and `candle` are empty toggles; `digitalbrain` and NSR/ODE features must not expand Real Compute scope. |
| `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs` | Runtime E2E scenario using compute summaries and deterministic evidence-chain assertions. | functional-prototype | Tests runtime integration around compute summaries, not necessarily external real model inference. | Integration test file has multiple `#[test]` cases. | Test name contains `real_compute`; Prompt 15 should classify whether the tested lane is stub/toy/runtime-integration rather than real model execution. |
| `.github/workflows/ci.yml` | CI workflow with feature-matrix lanes. | partial | Defines default, `candle-cpu`, `burn-cpu`, `stage-isolation`, `gpu-cuda`, and `ebm-train`. | Default lane is blocking; optional feature lanes use `continue-on-error` for non-blocking entries. | Optional lanes are useful compile/test signals but cannot be treated as production real-compute gates. |
| `.github/workflows/nightly_verify.yml` | Nightly workspace, docs, spec, goldens, readiness, adversarial checks. | functional-prototype | Exercises default workspace and readiness/goldens, not optional Burn/Candle by default. | Nightly Linux and Windows jobs. | Nightly default coverage is offline deterministic but does not prove optional real backend artifacts. |
| `docs/feature_matrix.md` and `README.md` | Document supported feature lanes. | docs-only | Mention default toy, Candle CPU, Burn CPU, stage isolation, and tools lane. | Docs lint only. | README labels the feature matrix as production; should be softened or clarified in Prompt 15. |
| `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md` | Model pipeline status and canonical runtime-path narrative. | docs-only | Claims canonical runtime path and describes Burn/Candle/model slots. | Docs lint only. | Several phrases can overclaim canonical production readiness relative to toy/fixture code evidence. |
| `docs/backend_burn_world_v0.md`, `docs/backend_candle_*.md`, `docs/backends*.md`, `docs/real_compute_*` | Historical/backend-specific docs and readiness/dossier material. | docs-only | Useful for prior intent and architecture, but must be reconciled with current code and tests. | Docs lint only. | Historical readiness language can be mistaken for current production proof. |

## 3. Feature Flag Matrix

| Crate | Feature | Claims / Intended lane | Actual dependencies | Default? | Offline deterministic? | External model needed? | Risk |
|---|---|---|---|---:|---:|---:|---|
| `ucf-compute` | `backend-stub` | Stub/default compatibility lane. | none | yes | yes | no | low; label is honest. |
| `ucf-compute` | `backend-toy` | Toy default lane and pack validation for `toy_v1` and worker-related toy surfaces. | none | yes | yes | no | medium; default toy can be mistaken for real compute if docs use production wording. |
| `ucf-compute` | `backend-candle` | Enables Candle stage module export/contract surfaces. | none directly; `compute-candle` pulls `candle-core` | no | yes for compile/fixtures | yes for verified model execution | medium; feature name lacks toy/real distinction. |
| `ucf-compute` | `compute-stub` | Alias enabling `backend-stub`. | `backend-stub` | no | yes | no | low; alias should be documented. |
| `ucf-compute` | `compute-candle` | Candle compute lane. | `candle-core`, `backend-candle` | no | yes for compile/fixtures | yes for real verified slots | medium/high; can be compile/fixture-only without model artifacts. |
| `ucf-compute` | `compute-burn` | Burn compute lane. | none in Cargo | no | yes | yes for real verified slots | high; Burn naming implies backend but Cargo has no Burn engine dependency. |
| `ucf-compute` | `backend-burn` | Burn backend pack validation and stage module. | `compute-burn` | no | yes | yes for real verified slots | high; `burn_toy_v1` must not be called real by default. |
| `ucf-compute` | `llm-candle` | Candle LLM lane. | `candle-core` | no | yes for tiny fixture | yes for verified safetensors/tokenizer path | medium; real only with artifacts. |
| `ucf-compute` | `llm-burn` | Burn LLM lane. | none | no | compile-only / not implemented | yes for future real path | high; Burn LLM backend returns `NotImplemented`. |
| `ucf-compute` | `lfm-candle` | Candle LFM lane and pack validation for Candle packs. | `candle-core` | no | yes for toy/fixture | yes for verified real path | medium; required for `candle_toy_v1` even though name says LFM. |
| `ucf-compute` | `lfm-burn` | Burn LFM lane. | none | no | yes for fixture/degraded path | yes for verified `lfm` slot | high; requires model-store clarity before real claim. |
| `ucf-compute` | `lfm-lnn` | LNN/ODE toy LFM pack. | none | no | yes | no | medium; LNN can sound real/scientific but is a toy/ODE kernel here. |
| `ucf-compute` | `plasticity` | Optional plasticity-related surfaces. | none | no | unknown from inventory | unknown | medium; out of scope for Real Compute prompt series unless documented as non-compute. |
| `ucf-compute` | `replay` | Optional replay-facing surfaces. | none | no | yes if used with fixtures | no by itself | medium; must not create replay scope creep. |
| `ucf-compute` | `ops-explain` | Optional ops explanation surfaces. | none | no | yes | no | low/medium; explainability docs should not assert real inference. |
| `ucf-compute` | `remote-compute` | Remote proxy lane. | none | no | no for remote service; local construction is gated | yes, plus allowlist/policy/env | high; must remain disabled by default and never hidden-activate. |
| `ucf-runtime` | `compute-candle` | Propagates Candle compute to `ucf-compute`. | `ucf-compute/compute-candle` | no | yes for compile/fixtures | yes for real slots | medium. |
| `ucf-runtime` | `compute-burn` | Propagates Burn compute to `ucf-compute`. | `ucf-compute/compute-burn` | no | yes for compile | yes for real slots | high. |
| `ucf-runtime` | `llm-candle` | Propagates Candle LLM. | `ucf-compute/llm-candle` | no | yes for fixture | yes for real slots | medium. |
| `ucf-runtime` | `llm-burn` | Propagates Burn LLM. | `ucf-compute/llm-burn` | no | compile-only / not implemented | yes for future | high. |
| `ucf-runtime` | `lfm-candle` | Propagates Candle LFM. | `ucf-compute/lfm-candle` | no | yes for fixture | yes for real slots | medium. |
| `ucf-runtime` | `lfm-burn` | Propagates Burn LFM. | `ucf-compute/lfm-burn` | no | yes for fixture/degraded path | yes for real slots | high. |
| `ucf-runtime` | `gpu-cuda` / `gpu-metal` | Optional GPU backend package. | optional `ucf-backends-gpu` features | no | unknown | hardware/toolchain likely | high; not a default CI-safe real-compute lane. |
| `ucf-ai-backends` | `ai-candle` | Compatibility Candle adapter module. | none | no | yes | no real tensor model path in crate | high; TODO returns empty bounded output. |
| `ucf-ai-backends` | `ai-burn` | Compatibility Burn adapter module. | none | no | yes | no real tensor model path in crate | high; TODO returns empty bounded output. |
| `ucf-ai-port` | `bus` | Optional bus integration. | `ucf-bus` | no | yes | no | low; not a compute backend. |
| `ucf-ai-port` | `ai-runtime` | Optional AI runtime port. | `ucf-ai-runtime/ai-runtime` | no | unknown | unknown | medium; integration toggle, not real backend proof. |
| `ucf-ai-port` | `ode` | Optional ODE port. | `ucf-ode-port` | no | yes/unknown | no | medium; avoid LNN/ODE scope creep. |
| `ucf-ai-port` | `burn` | Placeholder/marker. | none | no | yes | no implementation | high; empty feature name can overclaim. |
| `ucf-ai-port` | `candle` | Placeholder/marker. | none | no | yes | no implementation | high; empty feature name can overclaim. |
| `ucf-ai-port` | `digitalbrain` | Digitalbrain/bus-related integration. | `ucf-bus` | no | unknown | no by itself | high for scope creep; not part of Real Compute lane. |
| `ucf-ai-port` | `nsr-smt` | Optional NSR SMT port. | `ucf-nsr-smt` | no | unknown | no by itself | medium; NSR is not compute backend evidence. |
| `ucf-ai-port` | `nsr-datalog` | Optional NSR datalog port. | `ucf-nsr-datalog` | no | unknown | no by itself | medium; NSR is out of compute scope here. |

## 4. Backend / Lane Classification

| Lane | Code paths | Feature flags | Classification | Deterministic? | Offline? | Tests | Can be CI default? | Overclaim risk |
|---|---|---|---|---:|---:|---|---:|---|
| Default backend kind `stub` | `runtime/ucf-compute/src/backends.rs` | default `backend-stub`, `backend-toy` | stub | yes | yes | workspace/default and `ucf-compute` package tests | yes | low if called stub, high if called real. |
| `stub_v0` pack | `runtime/ucf-compute/src/backend_pack.rs` | none required beyond crate compile | stub | yes | yes | pack parsing/build tests | yes | low. |
| `toy_v1` pack | `backend_pack.rs`, fixture files, toy kernels | default `backend-toy` | toy | yes | yes | fixture digest, package tests, workspace default | yes | medium if treated as real compute. |
| `candle_toy_v1` pack | `backend_pack.rs`, `backends/candle_backend.rs`, `stage_v1_candle.rs`, Candle LLM code | `compute-candle`, `llm-candle`, `lfm-candle`, `backend-candle` where applicable | optional-real-compile with toy/fixture fallback | yes for fixtures | yes for compile/fixture; no for external artifacts | non-blocking feature-matrix `candle-cpu`; package feature tests | no | high unless every claim says compile/fixture or verified artifact. |
| `candle_liquid_v1` pack | `backend_pack.rs`, `lfm.rs` | `lfm-candle` plus `backend-toy` validation | experimental/toy | yes | yes | feature-gated pack tests | no | medium. |
| `burn_toy_v1` pack | `backend_pack.rs`, `backends/burn_backend.rs`, `stage_v1_burn.rs`, Burn LFM/LLM seams | `compute-burn`, `backend-burn`, `lfm-burn`, `llm-burn` depending stage | optional-real-compile / toy-named runtime prototype | yes for fixture/degraded path | yes for compile; real path needs artifacts | non-blocking feature-matrix `burn-cpu`; package feature tests | no | high because docs call Burn canonical/production while pack is `BurnToyV1`. |
| Burn LLM | `capabilities/burn_llm_backend.rs` | `llm-burn` or `compute-burn` | stub/compile-only | yes | yes | compile coverage only | no | high; returns `NotImplemented`. |
| Candle LLM fixture | `capabilities/candle_llm_backend.rs`, `capabilities/llm_toy.rs`, fixtures | `llm-candle` or `compute-candle` | toy / optional-real-runtime with safetensors slot | yes for fixture | yes for fixture; real needs local artifact | unit tests and feature-lane compile/tests | no | medium. |
| `toy_lnn_v1` pack | `backend_pack.rs`, `lfm.rs` | `lfm-lnn` plus `backend-toy` or `lfm-candle` | toy/experimental | yes | yes | feature-gated pack tests | no | medium. |
| `worker_v1` pack | `worker_backend.rs`, `backend_pack.rs` | `backend-toy` | internal worker/mock-ish execution lane | yes locally | yes | package tests where enabled | no | medium; worker is internal, not real backend proof. |
| `remote_v1` pack | `remote_compute.rs`, `backend_pack.rs` | `remote-compute` plus env/policy allowlist | remote/external | no as a backend service | no | feature-gated construction tests only | no | critical if hidden-activated; must remain disabled by default. |
| `domains/ai-backends` `ai-candle` | `domains/ai-backends/src/candle_backend.rs` | `ai-candle` | stub | yes | yes | compile/workspace only | no | high; returns empty bounded output. |
| `domains/ai-backends` `ai-burn` | `domains/ai-backends/src/burn_backend.rs` | `ai-burn` | stub | yes | yes | compile/workspace only | no | high; returns empty bounded output. |
| GPU backend lane | `runtime/ucf-backends-gpu` and `ucf-runtime` GPU features | `gpu-cuda`, `gpu-metal` | unknown/experimental | unknown | no/unknown | non-blocking `gpu-cuda` CI lane | no | high; hardware/toolchain not part of Minimal Spine or default compute lane. |

Clear classification statements:

- The default lane is `ComputeBackendKind::Stub` with default features `backend-stub` and `backend-toy`.
- The only CI-default-safe compute behavior is stub/toy, because it is deterministic and offline.
- No lane should be called production Real Compute without a verified local model artifact path, deterministic fixture/golden coverage, and an enabled gate proving that exact path.
- `stub_v0`, `toy_v1`, `toy_lnn_v1`, `worker_v1`, and `domains/ai-backends` adapters must be called stub, toy, internal, or compatibility lanes, not real.
- `candle_toy_v1` and `burn_toy_v1` are optional compile/runtime prototypes; they can be compile-checked and fixture-tested, but realness depends on verified model slots.
- `remote_v1` needs environment activation, policy allowlist approval, and external service semantics; it must never be default or hidden-activated.

## 5. Test and Fixture Inventory

| Test/fixture | Path | Lane covered | Deterministic? | Offline? | Current claim | Gap |
|---|---|---|---:|---:|---|---|
| Package tests | `cargo test -p ucf-compute --all-targets` | Default stub/toy plus unit-tested internals. | yes | yes | Compute crate compiles and default tests pass. | Does not prove optional Burn/Candle real model artifacts. |
| Workspace tests | `cargo test --workspace` / CI default lane | Default stub/toy across workspace. | yes | yes | Blocking CI baseline. | Optional feature lanes are not default. |
| Feature-matrix `candle-cpu` | `.github/workflows/ci.yml` | Candle compile/fixture lane. | likely for fixtures | yes unless artifacts are configured | Non-blocking CI feature lane. | Does not by itself prove real external model inference. |
| Feature-matrix `burn-cpu` | `.github/workflows/ci.yml` | Burn compile/prototype lane. | likely | yes unless artifacts are configured | Non-blocking CI feature lane. | Cargo has no Burn engine dependency; Burn LLM is not implemented. |
| Nightly workspace | `.github/workflows/nightly_verify.yml` | Default workspace, docs, goldens, readiness, adversarial. | yes | yes | Nightly default validation. | No optional Burn/Candle real lane by default. |
| Runtime onboarding E2E | `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs` | Runtime integration with compute summaries/evidence chain. | yes | yes | Test name says real compute onboarding. | Must be relabeled or documented as runtime onboarding unless it proves real model execution. |
| Embedded compute fixtures | `runtime/ucf-compute/fixtures/*.json` | Toy LLM, JEPA, SAE, SSM, LFM/LNN parameter fixtures. | yes | yes | Deterministic tiny fixtures. | Not production model artifacts. |
| Model manifest | `models/manifest.toml` | Optional model-store slot configuration. | yes if local files stable | yes if files are local | Canonical local manifest path in model pipeline docs. | Slots may be disabled or require external/local artifacts; inventory did not prove all real slots. |
| `ucf-ai-backends` compile | `domains/ai-backends/src/*.rs` | Compatibility Burn/Candle adapters. | yes | yes | Feature-gated adapter seams. | No tensor I/O; TODO returns empty bounded outputs. |
| Ops model lifecycle tests | `runtime/ucf-ops/src/models_lifecycle.rs` and related ops modules | Model governance/evidence reports and backend resolution artifacts. | yes | yes for repo-local reports | Operational evidence surfaces. | Governance reports are not backend inference tests. |

## 6. Docs vs Code Drift

| Claim | Source doc | Code/test evidence | Status | Risk | Recommended correction |
|---|---|---|---|---|---|
| `runtime/ucf-compute` is the canonical runtime model pipeline. | `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md` | Code exports canonical compute surfaces and backend builders. | partially implemented | medium | Keep, but qualify that canonical path includes stub/toy/prototype lanes and not production readiness by default. |
| Canonical production compute path is Burn onboarding. | `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md` and compute crate docs | `CANONICAL_ONBOARDING_PACK` is `BurnToyV1`; Burn feature lanes are optional/non-default. | partially implemented / overclaim-prone | high | Rename or annotate as optional Burn onboarding prototype until real artifacts and blocking tests exist. |
| README feature matrix is production. | `README.md` | Default lane is toy; optional lanes are non-blocking in CI. | contradicted by lane maturity | high | Prompt 15 should clarify production wording without changing behavior. |
| `candle-cpu` supported lane. | `README.md`, `docs/feature_matrix.md`, CI | Non-blocking CI lane with Candle features. | implemented as optional compile/test lane | medium | Call it optional Candle CPU compile/fixture lane, not production real compute. |
| `burn-cpu` supported lane. | `README.md`, `docs/feature_matrix.md`, CI | Non-blocking CI lane with Burn features; Burn LLM not implemented; Burn engine dependency absent. | partially implemented / compile-only in places | high | Call it optional Burn CPU prototype/compile lane until real model path is proven. |
| `domains/ai-backends` provides Burn/Candle adapters. | `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md` and compatibility docs | Adapter modules exist but TODO and return empty bounded outputs. | stub-only | high | Mark compatibility-only stub adapters. |
| Real compute onboarding E2E proves real compute. | Test filename and historical docs | Test covers deterministic runtime integration and compute summaries; inventory did not identify external model artifact execution. | unclear / likely toy-runtime integration | high | Prompt 15 should classify test claims and names in docs; no code rename yet. |
| Remote compute available. | `docs/remote_compute_deferred.md` and feature flag | Remote pack is feature-gated and env/policy gated. | documented but disabled/deferred | critical if misread | Continue to state disabled by default and policy-gated. |
| Blue-Brain/HH/microcircuit compute readiness. | Historical `docs/blue_brain_*` | Out of scope for Real Compute Optional Lane; current index treats as advisory/deferred in many areas. | documented but out-of-scope | critical for scope creep | Do not integrate in prompts 15-24 unless a separate scope explicitly permits it. |
| Production-ready full compute stack. | Historical final/readiness docs | Current code has bounded service and prototypes, but not fleet/persistent/external full stack proof. | documented but missing as production claim | high | Replace broad production wording with exact tested lane and artifact requirements. |
| No external model needed. | Toy/default docs and fixture behavior | True only for stub/toy/fixtures; not true for verified real model slots. | partially implemented | medium | State per-lane: default is offline; real verified slots need local artifacts. |

## 7. Safety Assessment

| Risk | Evidence | Severity | Required mitigation |
|---|---|---|---|
| Minimal Spine accidentally depends on Real Compute. | Freeze docs keep v1.x independent; compute is a separate optional lane. | high | Keep compute optional, do not add required dependency or runtime activation to Minimal Spine v1.x. |
| Default feature set overclaims real compute. | `ucf-compute` defaults are `backend-stub` and `backend-toy`. | medium | Always label default as stub/toy and offline deterministic. |
| Burn naming overclaims production. | `BurnToyV1`, `llm-burn` returns `NotImplemented`, and `compute-burn` has no Burn engine dependency. | high | Prompt 15 should define naming taxonomy and warnings around Burn prototype/compile lanes. |
| Candle naming hides fixture vs verified-artifact boundary. | Candle LLM has fixture mode and verified safetensors mode; Candle pack can fall back or reject slots depending status. | medium | Document exact activation requirements and separate fixture tests from artifact tests. |
| Compatibility adapters overclaim backend maturity. | `domains/ai-backends` TODO adapters return empty bounded outputs. | high | Mark as compatibility stubs in docs and do not route production claims through them. |
| Remote compute hidden activation. | `remote-compute` feature plus `UCF_REMOTE_ENABLE=1` and allowlist/policy requirements. | critical | Keep disabled by default; require explicit feature, env, policy allowlist, and tests before use. |
| Runtime E2E test name implies real model execution. | `e2e_real_compute_onboarding.rs` does deterministic runtime integration. | high | Prompt 15 should classify/rename documentation labels or add explanatory docs; no test rename in this prompt. |
| CI optional lanes misread as blocking production proof. | CI uses `continue-on-error` for non-default feature lanes. | medium | Matrix docs should distinguish blocking default from non-blocking optional compile/test lanes. |
| Docs imply production readiness. | README and model pipeline status use production/canonical wording. | high | Prompt 15 should create claim taxonomy and proposed docs/API cleanup plan. |
| Blue-Brain/HH/microcircuit/DBM scope creep. | Compute crate exports diagnostic Blue-Brain modules, but prompt guardrails forbid integration. | critical | Keep Real Compute prompts restricted to compute inventory/naming/tests; no integration or authority changes. |

Safety answers:

- Compute can remain optional if feature defaults and docs keep stub/toy as the only default behavior.
- No hidden Minimal Spine dependency was introduced by this inventory.
- Default features are safe for offline deterministic CI but too broad in wording if called real.
- Feature names such as `backend-burn`, `compute-burn`, `llm-burn`, `lfm-burn`, `compute-candle`, `llm-candle`, and `lfm-candle` need doc-level realness boundaries.
- Tests with `real_compute` in the name should be treated as onboarding/integration tests until they prove verified model execution.
- Docs suggesting production readiness should be corrected or annotated before first implementation work.
- CI lanes should keep optional Burn/Candle non-default until artifact-backed tests exist.
- Before first implementation, Prompt 15 should define taxonomy, ambiguous aliases, and precise docs/API cleanup targets.

## 8. Required Guardrails

- Real Compute remains optional only.
- Minimal Spine v1.x must not gain a required Real Compute dependency.
- Stub, toy, mock, partial, optional-real, and real labels are required at every lane boundary.
- No production claim is allowed without a real model fixture/artifact path and tests for that exact path.
- No external service is enabled by default.
- Deterministic fixtures are required for CI-safe lanes.
- No hidden activation through defaults, environment fallbacks, or broad feature aliases.
- No policy/output override authority is granted to compute lanes.
- No Gateway write integration is part of this lane inventory.
- No Evidence or Archive authority changes are part of this lane inventory.
- No Blue-Brain, HH, microcircuit, DBM, Geist, Replay, Gateway, or Capability scope creep is allowed.

## 9. Prompt 15 Naming Boundary Status

Prompt 15 is complete. The naming and boundary cleanup plan is available at [`docs/roadmap/compute_backend_naming_boundary_plan.md`](compute_backend_naming_boundary_plan.md).

Prompt 15 remains analysis and documentation only. It:

1. Defines a naming taxonomy for `stub`, `toy`, `mock`, `optional-real-compile`, `optional-real-runtime`, `remote/external`, `experimental`, `deferred`, and `forbidden-for-now`.
2. Identifies ambiguous names and aliases such as `real_compute`, `canonical production`, `BurnToyV1`, `CandleToyV1`, `compute-burn`, `llm-burn`, and `ai-backends` adapter names.
3. Proposes documentation/API cleanup without behavior changes, feature renames, new backends, or runtime activation.
4. Marks which docs should be softened from production claims to exact lane claims.
5. Preserves Minimal Spine v1.x independence and all guardrails above.

Suggested prompt series after this inventory and the Prompt 15 plan:

| Prompt | Focus | Behavior changes? |
|---|---|---:|
| 15 | Stub/Toy/Real backend naming and boundary cleanup plan. | no; complete |
| 16 | Compute Backend Trait Contract Hardening with explicit backend identity/classification metadata. | labels/metadata only |
| 17 | Test-name/coverage map and proposed fixture taxonomy. | no |
| 18 | Compile-gate design for one optional real backend lane. | no implementation by default |
| 19 | Local model artifact contract and deterministic fixture requirements. | no runtime activation |
| 20 | CI lane policy for optional real compile/tests. | no default activation |
| 21 | Candle lane proof plan. | no default activation |
| 22 | Burn lane proof plan. | no default activation |
| 23 | Remote compute disabled-boundary audit. | no activation |
| 24 | Real Compute optional-lane readiness gate proposal. | no Minimal Spine dependency |

## 10. Open Questions

- Which backend should be the first real compile gate: Candle, Burn, or another already-present optional lane?
- Are current Burn/Candle feature combinations buildable across Linux and Windows at current HEAD with no external artifacts?
- Which local model fixtures or artifacts exist beyond embedded toy JSON fixtures and `models/manifest.toml` slots?
- Which tests are actually stub/toy/runtime-integration despite names containing `real_compute`?
- Which feature flags should be renamed later versus documented as legacy aliases?
- Should `ucf-ai-port` empty `burn`/`candle` features remain as markers or be documented as non-backend toggles?
- What exact evidence artifact should prove a future production-real model path without changing Evidence/Archive authority?

## 11. Prompt 17 Stub Deterministic Fixture Lane Note

Prompt 17 hardens the CPU stub path as a deterministic fixture lane only. It does not activate real compute and does not change the optional status of any Real Compute lane.

Current stub-lane facts after Prompt 17:

- The stub lane is classified as `BackendClass::Stub` through backend identity metadata.
- Stub fixture output carries explicit fixture provenance: backend name `stub`, fixture id `stub_compute_fixture_v1`, no-real-inference true, external-service-required false, runtime-inference-supported false, and production-claim false.
- The stub pack reports `StubV0` component identifiers and uses a zero external model-hash digest for deterministic fixture output, so it does not require external model artifacts.
- Stub fixture tests compare repeated-output equality, stable digest, provenance metadata, offline/no-external-artifact properties, and no real runtime inference claim in `runtime/ucf-compute/tests/stub_compute_fixture.rs`.
- This is not a production compute claim, not optional-real-runtime evidence, not Gateway integration, not Evidence/Archive authority, not policy/output override authority, and not a Minimal Spine v1.x dependency.

Remaining real-compute inventory gaps are unchanged: toy golden coverage, optional-real compile gates, artifact-backed local runtime proof, compute-output linkage, feature CI matrix hardening, and overclaim cleanup still require later prompts.

## 12. Prompt 18 Toy Deterministic Golden Lane Note

Prompt 18 hardens the default `toy_v1` pack as a deterministic local golden lane only. It does not activate real compute, does not add a real backend, and does not change the optional status of any Real Compute lane.

Current toy-lane facts after Prompt 18:

- The toy lane is classified as `BackendClass::Toy` through backend identity metadata and is distinct from `BackendClass::Stub` and `BackendClass::OptionalRealRuntime`.
- Toy golden output carries explicit provenance: backend name `toy_v1`, fixture/golden id `toy_compute_golden_v1`, toy-not-real true, no-real-inference true, external-service-required false, runtime-inference-supported false, Minimal Spine authority false, and production-claim false.
- The toy golden lane is deterministic and offline for the golden fixture; tests compare repeated output equality and the pinned digest `0e73835b8059fb173668d8e8afbc8bc10c2e8f684194777399a2630d9ab5b7de`.
- The toy pack reports `ToyV1` component identifiers and has no required external model slots for the golden fixture; it does not require model paths, network configuration, or external services.
- Toy may support local toy inference semantics through embedded local fixtures and small deterministic kernels, but this is not production inference and not real model inference.
- Toy has no Minimal Spine authority, no Gateway write authority, no Evidence/Archive authority change, and no policy/output override authority.

Remaining real-compute inventory gaps are now: optional-real compile gates, artifact-backed local runtime proof, compute-output linkage, compute evidence/audit records, feature CI matrix hardening, docs overclaim cleanup, and readiness-gate/prod-profile stability.

## Prompt 19 Optional-Real Compile Gate Results

Prompt 19 inventories optional-real lanes as compile/check gates only. It intentionally does not activate real compute, does not load models, does not add external services, does not integrate a gateway path, and does not make Minimal Spine v1.x depend on compute.

### Feature inventory

| Feature / lane | Path(s) | Current purpose | Current `BackendClass` | Default? | Build dependency | Artifact/model dependency | Current tests | Gap |
|---|---|---|---|---:|---|---|---|---|
| `backend-candle` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs` | Candle optional compile/backend seam | `OptionalRealCompile` for Candle backend/pack identities | no | `candle-core` | Runtime artifacts deferred; no local runtime fixture asserted | `backend_identity_contract`, `optional_real_compile_gate` | Needs artifact-backed fixture before any runtime claim. |
| `compute-candle` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Alias-like Candle compute lane enabling `backend-candle` | `OptionalRealCompile` through Candle identities | no | `candle-core` | Runtime artifacts deferred | compile probe | CI should treat as compile-only. |
| `llm-candle` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/capabilities/candle_llm_backend.rs` | Candle LLM compile candidate | No separate machine-readable identity; optional-real compile candidate by feature class | no | `candle-core` | Tiny/local fixture references exist in code paths, but no runtime inference claim is made here | compile probe | Add explicit identity only when a concrete wrapper exposes it. |
| `lfm-candle` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/lfm.rs` | Candle LFM compile candidate | No separate machine-readable identity; optional-real compile candidate by feature class | no | `candle-core` | Model/artifact runtime deferred | compile probe | Add explicit identity only when a concrete wrapper exposes it. |
| `backend-burn` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs` | Burn optional compile/backend seam; aliases `compute-burn` | `OptionalRealCompile` for Burn backend/pack identities | no | none beyond workspace deps | Runtime artifacts deferred; no local runtime fixture asserted | `backend_identity_contract`, `optional_real_compile_gate` | Needs artifact-backed fixture before any runtime claim. |
| `compute-burn` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Burn compute compile lane | `OptionalRealCompile` through Burn identities | no | none beyond workspace deps | Runtime artifacts deferred | compile probe | CI should treat as compile-only. |
| `llm-burn` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/capabilities/burn_llm_backend.rs` | Burn LLM compile candidate | No separate machine-readable identity; optional-real compile candidate by feature class | no | none beyond workspace deps | Model/artifact runtime deferred | compile probe | Add explicit identity only when a concrete wrapper exposes it. |
| `lfm-burn` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/lfm.rs` | Burn LFM compile candidate | No separate machine-readable identity; optional-real compile candidate by feature class | no | none beyond workspace deps | Model/artifact runtime deferred | compile probe | Add explicit identity only when a concrete wrapper exposes it. |
| `lfm-lnn` | `runtime/ucf-compute/Cargo.toml`, `BackendPackKind::ToyLnnV1` | LNN-adjacent/toy compile lane | `Toy` for `ToyLnnV1` pack identity | no | none beyond workspace deps | none | compile probe | Not a real backend proof. |
| `remote-compute` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/remote_compute.rs`, `BackendPackKind::RemoteV1` | Remote/external compile lane | `RemoteExternal` when compiled | no | none beyond workspace deps | External service required for runtime use | `optional_real_compile_gate` under feature | Keep excluded from default/offline CI runtime paths. |
| `burn` | `core/crates/ucf-ai-port/Cargo.toml` | Compatibility feature flag | none in this crate | no | none beyond workspace deps | none | compile probe | Docs-only/compatibility unless a concrete identity is added later. |
| `candle` | `core/crates/ucf-ai-port/Cargo.toml` | Compatibility feature flag | none in this crate | no | none beyond workspace deps | none | compile probe | Docs-only/compatibility unless a concrete identity is added later. |
| `ai-runtime` | `core/crates/ucf-ai-port/Cargo.toml` | Optional dependency seam to `ucf-ai-runtime` | none in this crate | no | `ucf-ai-runtime` path dependency | none | compile probe | Compile proof only; not inference proof. |
| `ai-burn` | `domains/ai-backends/Cargo.toml`, `domains/ai-backends/src/burn_backend.rs` | Non-canonical adapter module | none; adapter seam only | no | none beyond host ABI | none; returns bounded empty output | compile probe | Add identity only if this adapter becomes an explicit backend inventory item. |
| `ai-candle` | `domains/ai-backends/Cargo.toml`, `domains/ai-backends/src/candle_backend.rs` | Non-canonical adapter module | none; adapter seam only | no | none beyond host ABI | none; returns bounded empty output | compile probe | Add identity only if this adapter becomes an explicit backend inventory item. |

### Compile gate probe results

| Command | Result | Meaning | Follow-up |
|---|---|---|---|
| `cargo check -p ucf-compute --no-default-features` | PASS | `ucf-compute` compiles without default stub/toy features. | Keep as default safety probe. |
| `cargo check -p ucf-compute --features backend-stub` | PASS | Stub lane compiles. | Continue fixture tests. |
| `cargo check -p ucf-compute --features backend-toy` | PASS | Toy lane compiles. | Continue golden tests. |
| `cargo check -p ucf-compute --features backend-burn` | PASS | Burn backend alias compiles. | Treat as compile-only optional-real gate. |
| `cargo check -p ucf-compute --features backend-candle` | PASS | Candle backend compiles with `candle-core`. | Treat as compile-only optional-real gate. |
| `cargo check -p ucf-compute --features compute-burn` | PASS | Burn compute feature compiles. | Treat as compile-only optional-real gate. |
| `cargo check -p ucf-compute --features compute-candle` | PASS | Candle compute feature compiles. | Treat as compile-only optional-real gate. |
| `cargo check -p ucf-compute --features llm-burn` | PASS | Burn LLM feature compiles. | No runtime claim until fixture/artifact test exists. |
| `cargo check -p ucf-compute --features llm-candle` | PASS | Candle LLM feature compiles. | No runtime claim until fixture/artifact test exists. |
| `cargo check -p ucf-compute --features lfm-burn` | PASS | Burn LFM feature compiles. | No runtime claim until fixture/artifact test exists. |
| `cargo check -p ucf-compute --features lfm-candle` | PASS | Candle LFM feature compiles. | No runtime claim until fixture/artifact test exists. |
| `cargo check -p ucf-compute --features lfm-lnn` | PASS | LNN-adjacent feature compiles. | Keep non-real/toy wording. |
| `cargo check -p ucf-compute --features remote-compute` | PASS | Remote lane compiles. | Keep explicit external-service classification. |
| `cargo check -p ucf-ai-port --features burn` | PASS | Compatibility flag compiles. | No backend identity or runtime claim. |
| `cargo check -p ucf-ai-port --features candle` | PASS | Compatibility flag compiles. | No backend identity or runtime claim. |
| `cargo check -p ucf-ai-port --features ai-runtime` | PASS | Optional runtime dependency seam compiles. | Not a backend inference proof. |
| `cargo check -p ucf-ai-backends --all-targets` | PASS | Non-canonical adapter crate compiles by default. | Keep default no-real. |
| `cargo check -p ucf-ai-backends --features ai-burn --all-targets` | PASS | Burn adapter module compiles. | Adapter still makes no production/runtime claim. |
| `cargo check -p ucf-ai-backends --features ai-candle --all-targets` | PASS | Candle adapter module compiles. | Adapter still makes no production/runtime claim. |

A PASS above means compile/check success only. It is not proof of runtime inference, model loading, production readiness, or external service availability.

### Identity findings

| Lane | `BackendClass` | Runtime inference supported | Production claim | Notes |
|---|---|---:|---:|---|
| `ComputeBackendKind::Candle` | `OptionalRealCompile` | false | false | Machine-readable compile identity exists. |
| `ComputeBackendKind::Burn` | `OptionalRealCompile` | false | false | Machine-readable compile identity exists. |
| `BackendPackKind::CandleToyV1` | `OptionalRealCompile` | false | false | Name remains risky; docs must not imply real runtime. |
| `BackendPackKind::CandleLiquidV1` | `OptionalRealCompile` | false | false | Compile identity only. |
| `BackendPackKind::BurnToyV1` | `OptionalRealCompile` | false | false | Compile identity only despite Burn naming. |
| `BackendPackKind::RemoteV1` | `RemoteExternal` | false | false | Only present with `remote-compute`; external service required and offline false. |
| `llm-*` / `lfm-*` feature-specific wrappers | none separate | false by absence | false by absence | Existing feature paths compile; add identities only when concrete wrappers expose them. |

### CI recommendation

| Lane | Recommended CI treatment | Reason |
|---|---|---|
| Default no-real-compute | Required blocking lane | Ensures no optional-real feature is default. |
| Stub fixture | Required blocking lane | Deterministic fixture safety. |
| Toy golden | Required blocking lane | Portable local golden safety; unrelated to real inference. |
| Burn/Candle optional-real compile | Explicit feature-matrix compile lane | Catches compile regressions without runtime claims. |
| LLM/LFM optional feature probes | Explicit compile-only lane if cost stays acceptable | Feature-specific coverage without model/artifact claims. |
| Remote/external | Excluded from default; optional separate compile lane only | Requires external-service semantics for runtime use. |
| Artifact-backed runtime fixture | Deferred until pinned local artifact exists | Required before any `OptionalRealRuntime` claim. |

Docs that mention Burn, Candle, LLM, LFM, remote compute, or AI runtime compatibility must keep the boundary explicit: compile support is not runtime inference support; optional-real-compile is not optional-real-runtime; no production claim exists; no external service is used by default; Minimal Spine v1.x does not depend on compute.
