# UCF Compute Backend Naming and Boundary Plan

## 0. Purpose

- Defines the canonical stub/toy/mock/optional-real/remote taxonomy for compute backend language.
- Planning only: this document does not change backend behavior, feature flags, runtime paths, policy authority, evidence authority, or output authority.
- Minimal Spine v1.x remains independent of all Real Compute optional lanes.
- Stub, toy, mock, optional-real-compile, optional-real-runtime, remote/external, experimental, deferred, and forbidden-for-now labels are intentionally stricter than prior wording so later prompts can harden APIs and tests without overclaiming.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `ff5d244db4abbd8f543491f94bd3149174daf44a` |
| HEAD short | `ff5d244d` |
| Dirty state | clean |
| Workspace package count | 192 |
| Real compute inventory present | yes |
| Freeze doc present | yes |
| `runtime/ucf-compute` present | yes |
| `domains/ai-backends` present | yes |
| `core/crates/ucf-ai-port` present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -15`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and required path presence checks.

Required companion documents:

- [`docs/roadmap/real_compute_lane_inventory.md`](real_compute_lane_inventory.md)
- [`docs/roadmap/post_freeze_roadmap_selection.md`](post_freeze_roadmap_selection.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)

## 2. Existing Naming Inventory

| Name / Term | Path(s) | Current usage | Clear or ambiguous? | Risk | Suggested classification |
|---|---|---|---|---|---|
| `backend-stub` | `runtime/ucf-compute/Cargo.toml` | Default feature enabling the stub backend lane. | Clear | Low if it keeps explicit stub wording. | stub |
| `compute-stub` | `runtime/ucf-compute/Cargo.toml` | Alias-like feature forwarding to `backend-stub`. | Mostly clear | Medium: `compute-*` aliases can be confused with richer backend lanes. | stub |
| `backend-toy` | `runtime/ucf-compute/Cargo.toml` | Default feature for deterministic toy-local behavior. | Clear | Low if documented as toy and not real inference. | toy |
| `ToyV1`, `toy_v1`, `ToyLfmKernel`, `ToySsmKernel`, `ToySaeExtractor` | `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/src/lfm.rs`, `runtime/ucf-compute/src/ssm.rs`, `runtime/ucf-compute/src/feature_extractor.rs` | Deterministic local fixture and small-kernel implementation names. | Clear | Low: must not be promoted to real/production claims. | toy |
| `MockJepaPredictor`, `LensMock`, `SaeMock` | `runtime/ucf-compute/src/backend_pack.rs`, `core/crates/ucf-ai-port/src/lib.rs` | Test-double or placeholder-facing components. | Clear | Medium if mocks appear in user-facing backend claims. | mock |
| `backend-candle` | `runtime/ucf-compute/Cargo.toml` | Optional feature adding Candle dependency for backend code. | Ambiguous | High: dependency presence can be misread as runtime proof. | optional-real-compile |
| `compute-candle` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-runtime/Cargo.toml`, `.github/workflows/ci.yml` | Optional Candle compute lane and CI matrix feature. | Ambiguous | High: name does not distinguish compile support, toy fixture, and verified artifact runtime. | optional-real-compile |
| `llm-candle` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Optional Candle LLM lane with tiny fixture/tokenizer paths and artifact hooks. | Ambiguous | High: may imply real LLM readiness before verified local artifact tests. | optional-real-compile |
| `lfm-candle` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Optional Candle LFM lane. | Ambiguous | Medium: backend label lacks runtime fixture/artifact class. | optional-real-compile |
| `CandleToyV1`, `candle_toy_v1` | `runtime/ucf-compute/src/backend_pack.rs` | Candle-named toy fixture pack. | Ambiguous | High: contains both a real backend family name and a toy class. | toy |
| `CandleJepaV1`, `CandleSaeV1`, `CandleSsmV1`, `CandleEbmV1`, `CandleLfmV1`, `CandleVljepaV1` | `runtime/ucf-compute/src/backend_pack.rs`, backend docs | Candle component names for optional backend slots. | Ambiguous | High unless metadata states compile/runtime/artifact status. | optional-real-compile |
| `backend-burn` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Optional Burn backend alias via `compute-burn`. | Ambiguous | High: can be interpreted as real Burn inference readiness. | optional-real-compile |
| `compute-burn` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-runtime/Cargo.toml`, `.github/workflows/ci.yml` | Optional Burn compute lane. | Ambiguous | High: current docs describe prototype/onboarding semantics, not production proof. | optional-real-compile |
| `llm-burn` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Optional Burn LLM lane. | Ambiguous | High: current implementation status should not be claimed as real LLM inference. | deferred |
| `lfm-burn` | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml` | Optional Burn LFM lane. | Ambiguous | Medium: needs exact compile/runtime status in backend identity. | optional-real-compile |
| `BurnToyV1`, `burn_toy_v1` | `runtime/ucf-compute/src/backend_pack.rs` | Burn-named toy fixture pack. | Ambiguous | High: easiest name to overclaim as real Burn backend. | toy |
| `BurnJepaV1`, `BurnSaeV1`, `BurnSsmV1`, `BurnLfmV1` | `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/src/backends/burn_backend.rs` | Burn component names under optional features. | Ambiguous | High unless classified as compile/prototype/runtime-verified per slot. | optional-real-compile |
| `remote-compute`, `RemoteProxyV1`, `remote_v1` | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/src/remote_compute.rs` | Explicitly gated remote lane requiring feature/env/policy checks. | Clear but sensitive | Critical: network/service use must never be default or hidden. | remote/external |
| `worker_v1` | `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/src/worker_backend.rs` | Worker adapter lane for bounded compute job execution. | Ambiguous | Medium: worker does not itself prove real backend runtime. | experimental |
| `ai-candle`, `ai-burn` | `domains/ai-backends/Cargo.toml`, `domains/ai-backends/src/*` | Compatibility adapter features returning bounded empty outputs/TODO seams. | Ambiguous | High: domain adapter names can overclaim canonical runtime backend status. | stub |
| `ai-runtime` | `core/crates/ucf-ai-port/Cargo.toml` | Optional AI runtime dependency feature for the AI port aggregation crate. | Ambiguous | Medium: integration/port feature, not a backend realness label. | experimental |
| `burn` | `core/crates/ucf-ai-port/Cargo.toml`, docs | Empty/marker feature in AI port plus backend family word elsewhere. | Ambiguous | High: overloaded marker and backend-family term. | ambiguous |
| `candle` | `core/crates/ucf-ai-port/Cargo.toml`, docs | Empty/marker feature in AI port plus backend family word elsewhere. | Ambiguous | High: overloaded marker and backend-family term. | ambiguous |
| `digitalbrain` | `core/crates/ucf-ai-port/Cargo.toml`, digitalbrain docs | Optional bridge/port integration. | Clear outside compute | Critical in this lane: must not be activated by compute prompts. | forbidden-for-now |
| `nsr-datalog`, `nsr-smt` | `core/crates/ucf-ai-port/Cargo.toml` | Optional NSR solver integrations. | Clear outside compute | Medium: solver features are not compute backend realness. | experimental |
| `ode` | `core/crates/ucf-ai-port/Cargo.toml` | Optional ODE port dependency. | Ambiguous | Medium: local math/integration feature, not Real Compute proof. | experimental |
| `lfm-lnn` | `runtime/ucf-compute/Cargo.toml` | Optional LNN/ODE local LFM feature. | Ambiguous | Medium: can be toy/nontrivial local algorithm but not production real compute by name alone. | toy |
| `plasticity` | `runtime/ucf-compute/Cargo.toml` | Optional feature associated with plasticity-adjacent compute records. | Ambiguous | Critical for scope creep; not Real Compute proof. | forbidden-for-now |
| `replay` | `runtime/ucf-compute/Cargo.toml`, runtime replay docs/tests | Optional replay-related feature/terms. | Clear outside compute | Critical for scope creep; compute naming must not alter Replay authority. | forbidden-for-now |
| `ops-explain` | `runtime/ucf-compute/Cargo.toml` | Optional operational explanation feature. | Ambiguous | Medium: explanation lane is not backend realness. | experimental |
| `Real Compute Onboarding v0` | `README.md`, `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs`, docs | Historic/onboarding wording for deterministic runtime integration. | Ambiguous | High: file/section names can imply proven real inference. | docs-only |
| `canonical feature matrix (production)` | `README.md` | Feature matrix heading names default toy and optional Candle/Burn lanes under production wording. | Ambiguous | High: heading overclaims default toy and optional compile lanes. | forbidden-for-now |
| `production`, `prod-ready`, `production-ready`, `ready`, `onboarded`, `complete`, `full pipeline` | `README.md`, backend docs, roadmap docs, generated reports | Broad maturity claims across docs. | Ambiguous | Critical when attached to stub/toy/optional-real lanes without evidence. | forbidden-for-now |

## 3. Canonical Compute Taxonomy

| Canonical class | Allowed meaning | Required label | CI default allowed? | Production claim allowed? | Required tests |
|---|---|---|---:|---:|---|
| stub | Deterministic, intentionally fake or simple backend; may produce fixed outputs; never real inference. | `class=stub`, `real_inference=false`, `deterministic=true`, `offline=true` | yes | no | Default compile/unit tests proving stable fixed outputs and identity not real. |
| toy | Deterministic small local algorithm/model; may compute nontrivial values; offline and CI-safe; not production real compute. | `class=toy`, `real_inference=false`, `deterministic=true`, `offline=true` | yes | no | Unit/golden tests with bounded fixtures and assertions that toy is not real. |
| mock | Test double only; not a user-facing lane and not a backend claim. | `class=mock`, `test_double=true`, `user_lane=false` | tests only | no | Test-only assertions or crate-local tests; no product docs claim. |
| optional-real-compile | Code compiles against a real backend dependency or feature; runtime inference is not proven by the compile gate. | `class=optional-real-compile`, `runtime_proven=false`, `external_default=false` | no, except non-default matrix jobs | no | Feature compile tests and metadata assertions that runtime real inference is not claimed. |
| optional-real-runtime | Optional local real inference with fixture/model artifacts; requires deterministic fixture or golden; no external service by default. | `class=optional-real-runtime`, `runtime_proven=true`, `local_artifact_required=true`, `external_default=false` | no | no by default; only exact bounded claim allowed | Explicit feature tests with local artifact checks, deterministic fixture/golden, identity metadata, and no policy/output/evidence authority changes. |
| remote/external | Requires network, service, API, remote worker, or nonlocal dependency. | `class=remote-external`, `external=true`, `default_enabled=false` | no | no | Disabled-by-default tests, explicit env/policy/allowlist tests, and denial tests for default CI. |
| experimental | Research, unstable, prototype, or diagnostic lane with no production claim. | `class=experimental`, `stable=false`, `production_claim=false` | only if offline deterministic and not real | no | Compile/unit tests documenting caveats and no hidden activation. |
| deferred | Documented or named but intentionally not active or not wired. | `class=deferred`, `active=false`, `runtime_proven=false` | no | no | Compile/documentation tests or explicit not-implemented assertions. |
| forbidden-for-now | Names or claims that must not be used without stronger evidence, for example `production-ready real compute`, `native real backend`, or `full pipeline ready`. | `class=forbidden-for-now` or no label; wording must be replaced | no | no | Docs lint/checklist in future prompt; no runtime test because use is disallowed. |

## 4. Existing Lane Mapping

| Existing lane / feature / backend | Current name | Canonical class | Keep name? | Rename later? | Doc clarification needed? | Reason |
|---|---|---|---:|---:|---:|---|
| Stub backend feature | `backend-stub` | stub | yes | no | yes | Clear default stub lane; docs should state never real inference. |
| Stub compute alias | `compute-stub` | stub | yes | maybe | yes | Alias is harmless but should be documented as stub-only. |
| Toy backend feature | `backend-toy` | toy | yes | no | yes | CI-safe default local toy lane. |
| Candle backend feature | `backend-candle` | optional-real-compile | yes | maybe | yes | Dependency/feature availability is not runtime proof. |
| Candle compute feature | `compute-candle` | optional-real-compile | yes | maybe | yes | Should be described as optional compile/prototype lane until artifact tests prove runtime. |
| Burn backend feature | `backend-burn` | optional-real-compile | yes | maybe | yes | Current name lacks compile/runtime distinction. |
| Burn compute feature | `compute-burn` | optional-real-compile | yes | maybe | yes | Must not imply real Burn production inference. |
| Candle LFM | `lfm-candle` | optional-real-compile | yes | maybe | yes | Needs backend identity and local fixture/artifact distinction. |
| Burn LFM | `lfm-burn` | optional-real-compile | yes | maybe | yes | Needs backend identity and local fixture/artifact distinction. |
| LNN LFM | `lfm-lnn` | toy | yes | maybe | yes | Deterministic local algorithm lane, not production real compute. |
| Candle LLM | `llm-candle` | optional-real-compile | yes | maybe | yes | May become optional-real-runtime only with verified local artifact/golden tests. |
| Burn LLM | `llm-burn` | deferred | yes | maybe | yes | Should remain explicitly not runtime-proven until implemented and tested. |
| Remote compute | `remote-compute` | remote/external | yes | no | yes | Clear name; requires strict disabled-by-default docs and tests. |
| AI runtime port | `ai-runtime` | experimental | yes | no | yes | Port/integration toggle, not compute backend realness. |
| AI port Burn marker | `burn` | ambiguous | yes | maybe | yes | Overloaded marker; should not be used as backend readiness label. |
| AI port Candle marker | `candle` | ambiguous | yes | maybe | yes | Overloaded marker; should not be used as backend readiness label. |
| Digital brain bridge | `digitalbrain` | forbidden-for-now | yes | no | yes | Explicitly out of scope for Real Compute optional lane prompts. |
| NSR datalog | `nsr-datalog` | experimental | yes | no | yes | Solver feature, not compute backend realness. |
| NSR SMT | `nsr-smt` | experimental | yes | no | yes | Solver feature, not compute backend realness. |
| ODE port | `ode` | experimental | yes | no | yes | Math/port feature, not Real Compute proof. |
| Replay | `replay` | forbidden-for-now | yes | no | yes | Replay authority must not be altered by compute lane cleanup. |
| Plasticity | `plasticity` | forbidden-for-now | yes | no | yes | Scope-creep risk; keep out of backend activation. |
| Ops explanation | `ops-explain` | experimental | yes | no | yes | Diagnostics/explanation, not backend realness. |
| Compatibility adapters | `ai-candle`, `ai-burn` | stub | yes | maybe | yes | Current adapter seams are not canonical real runtime backends. |
| Component pack | `stub_v0` / `StubV0` | stub | yes | no | yes | Clear if identity metadata says stub. |
| Component pack | `toy_v1` / `ToyV1` | toy | yes | no | yes | Clear if identity metadata says toy. |
| Component pack | `candle_toy_v1` / `CandleToyV1` | toy | yes | maybe | yes | Misleading compound name; likely alias/deprecate later rather than immediate rename. |
| Component pack | `burn_toy_v1` / `BurnToyV1` | toy | yes | maybe | yes | Misleading compound name; likely alias/deprecate later rather than immediate rename. |
| Component pack | `remote_v1` / `RemoteProxyV1` | remote/external | yes | no | yes | Keep but enforce explicit non-default activation. |

## 5. Overclaim / Forbidden Wording Audit

| Phrase / Claim | Path | Why risky | Replacement wording | Priority |
|---|---|---|---|---|
| `Canonical feature matrix (production)` | `README.md` | Places default toy and optional compile lanes under production wording. | `Canonical feature matrix (default and optional lanes; no production inference claim)` | P0 must fix before implementation prompts |
| `default (toy)` under production heading | `README.md` | Toy default is CI-safe but not production real compute. | `default deterministic toy/stub lane` | P0 must fix before implementation prompts |
| `candle-cpu` / `burn-cpu` matrix labels without caveat | `README.md`, `.github/workflows/ci.yml` | CPU labels can be read as runtime-proven real inference. | `optional-real-compile Candle CPU lane` / `optional-real-compile Burn CPU lane` | P1 fix during naming cleanup |
| `Real Compute Onboarding v0 Quick Start` | `README.md` | Historic onboarding language can imply proven real inference. | `Optional compute onboarding quick start; real runtime not yet proven by this section` | P0 must fix before implementation prompts |
| `e2e_real_compute_onboarding.rs` | `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs` | Test file name implies real compute, but should be classified as onboarding/integration unless artifact-backed. | Keep file for now; docs call it `deterministic optional-compute onboarding integration test` | P1 fix during naming cleanup |
| `canonical production backend` | `runtime/ucf-compute/src/lib.rs`, `runtime/ucf-compute/src/backends.rs` | Production wording overstates current backend proof level. | `canonical optional backend constructor` or `optional-real-compile constructor` | P0 must fix before implementation prompts |
| `production invariants` near compute runtime docs | `runtime/ucf-compute/src/lib.rs` | Invariants may be real, but wording can attach production maturity to backend lanes. | `runtime safety invariants; no production inference claim` | P1 fix during naming cleanup |
| `Burn backend ready`-style wording | `docs/backend_burn_world_v0.md`, roadmap docs | Burn lane is not automatically runtime-verified real inference. | `Burn optional-real-compile lane available; real runtime not yet proven` | P0 must fix before implementation prompts |
| `Candle backend ready`-style wording | `docs/backend_candle_*.md`, `docs/backends*.md` | Candle dependency/slot support is not equal to production real inference. | `Candle optional-real-compile lane; optional-real-runtime requires local artifact and golden` | P0 must fix before implementation prompts |
| `model pipeline ready` / `full pipeline` | `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`, `docs/canonical_model_pipeline_architecture.md` | Pipeline wording can imply end-to-end real model readiness. | `model pipeline architecture/status; no production inference claim` | P1 fix during naming cleanup |
| `onboarded` | README, roadmap and backend docs | Onboarding can be confused with production readiness. | `registered`, `compiled`, `fixture-tested`, or `runtime-proven` according to evidence. | P1 fix during naming cleanup |
| `complete` | roadmap/freeze/status docs | Completion of a prompt or doc does not prove production backend maturity. | `planning complete`, `inventory complete`, or `implementation scope complete` | P2 docs cleanup later |
| `production`, `prod-ready`, `production-ready` attached to compute lanes | README, docs, CI labels | Forbidden without artifact-backed local runtime tests and explicit authority boundaries. | `no production inference claim`; exact taxonomy label instead. | P0 must fix before implementation prompts |
| `native real backend` | backend docs if present in later cleanup search | Implies stronger evidence than compile/runtime fixture may provide. | `optional local backend dependency` or exact class label. | P1 fix during naming cleanup |
| `real model`, `real inference`, `LFM/LLM ready` | backend/model pipeline docs | Needs artifact/golden evidence per slot. | `fixture-backed toy lane`, `optional-real-compile lane`, or `optional-real-runtime candidate` | P0 must fix before implementation prompts |

## 6. Future Boundary Rules

| Rule | Applies to | Enforcement target | Future prompt |
|---|---|---|---|
| Backend identity must report canonical class. | All compute backend identities, packs, traits, and adapters. | API/trait metadata and unit tests. | Prompt 16 |
| Stub/toy must not report real. | `backend-stub`, `compute-stub`, `backend-toy`, `Toy*`, `Mock*`, toy Burn/Candle packs. | Metadata assertions and tests. | Prompt 16 |
| Optional-real-compile must not imply runtime inference. | `compute-candle`, `backend-candle`, `llm-candle`, `lfm-candle`, `compute-burn`, `backend-burn`, `llm-burn`, `lfm-burn`. | Feature-lane docs, backend identity fields, compile tests. | Prompt 16 |
| Optional-real-runtime must require explicit feature and local fixture/model artifact. | Future local Candle/Burn runtime proofs. | Fixture/golden tests and artifact validation. | Prompt 17+ |
| Remote/external must never be default. | `remote-compute`, `RemoteProxyV1`, remote workers/services. | Default-feature tests, env denial tests, CI matrix. | Prompt 16+ |
| No compute lane may mutate policy. | All compute lanes. | Boundary docs and future integration tests. | Prompt 16+ |
| No compute lane may override Minimal Spine `OutputRecord` unless explicitly modeled as derived compute output. | Runtime integration and output paths. | Tests around derived records vs authoritative output. | Prompt 17+ |
| No compute lane may append to Evidence/Archive as authority; it may only create auditable compute records later. | Evidence, archive, compute record paths. | Authority-boundary tests and docs. | Prompt 17+ |
| No compute lane may become required for Minimal Spine v1.x. | Workspace features, runtime, CI, docs. | Default path tests and Minimal Spine dependency checks. | Prompt 16+ |
| No compute lane may activate Blue-Brain, HH, microcircuit, DBM, Geist, Replay, or Capability integration by default. | Feature defaults, integration docs, runtime bridges. | Default-feature and docs checks. | Prompt 16+ |
| CI must prove default path remains no-real-compute. | CI, default features, readiness gates. | CI matrix and unit tests verifying canonical class defaults. | Prompt 16+ |
| Compatibility adapters must not be described as canonical runtime real backends. | `domains/ai-backends` and AI host ABI adapters. | Adapter docs and metadata labels. | Prompt 16+ |
| Backend family names must be paired with class labels. | Burn, Candle, LFM, LLM, LNN names in docs/API. | Docs lint/checklist and identity metadata. | Prompt 16+ |

## 7. Cleanup Plan

| Step | Scope | Files/modules | Goal | Prompt |
|---|---|---|---|---|
| A1 | Docs cleanup P0 | `README.md`, `runtime/ucf-compute/src/lib.rs`, `docs/backend_burn_world_v0.md`, `docs/backend_candle_*.md`, `docs/backends*.md` | Replace production/ready/real wording with exact taxonomy labels; no behavior changes. | Prompt 16 |
| A2 | Docs cleanup P1 | `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`, `docs/canonical_model_pipeline_architecture.md`, `docs/real_compute_exit_dossier_serie_l_v1.md`, roadmap inventory docs | Clarify pipeline/onboarding/status language as planning, compile, toy, or runtime-proven only. | Prompt 16 |
| A3 | Docs cleanup P2 | Historical readiness and completion docs | Add caveats only where current readers may mistake old claims for current production realness. | Prompt 17+ |
| B1 | API/trait metadata | `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/src/backends.rs`, compute backend traits | Add backend identity fields: canonical class, deterministic/offline flags, external dependency flag, runtime-proven flag. | Prompt 16 |
| B2 | Adapter metadata | `domains/ai-backends/src/*`, `core/crates/ucf-ai-port/src/lib.rs` | Mark compatibility adapters/ports as stub, experimental, or non-backend markers. | Prompt 16 |
| B3 | Capability labels | LLM/LFM/SAE/SSM/world model modules | Ensure capability names separate family (`Burn`/`Candle`) from class (`toy`/`optional-real-compile`/`runtime`). | Prompt 17+ |
| C1 | Test cleanup | `runtime/ucf-compute/src/*` tests and future `tests/*` | Add assertions that stub/toy/mock do not report real. | Prompt 16 |
| C2 | Test naming plan | `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs`, compute tests | Document or later rename misleading test names without changing behavior. | Prompt 17+ |
| C3 | Fixture/golden labels | `runtime/ucf-compute/fixtures/*`, golden docs | Label fixture-backed outputs as toy/stub unless they are future verified local real artifacts. | Prompt 17+ |
| D1 | CI/feature cleanup | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, `Cargo.toml`, package feature docs | Distinguish blocking default CI from non-blocking optional-real compile lanes. | Prompt 16 |
| D2 | Default feature checks | `runtime/ucf-compute/Cargo.toml`, runtime Cargo features, CI | Prove default remains no-real-compute and no remote/external activation. | Prompt 16 |
| D3 | Optional-real compile checks | CI matrix and package tests | Keep Candle/Burn compile checks optional and non-production-claiming. | Prompt 17+ |
| D4 | Remote/external disabled default | `runtime/ucf-compute/src/remote_compute.rs`, backend pack gating, CI | Add/maintain denial tests for missing explicit feature/env/policy allowlist. | Prompt 17+ |

## 8. Prompt 16 Readiness

Recommended next prompt: **UCF Prompt 16 — Compute Backend Trait Contract Hardening**.

Prompt 16 should:

1. Add backend identity/classification metadata to compute backend traits, backend packs, and compatibility adapters.
2. Preserve behavior except for explicit labels and metadata exposure.
3. Add tests proving stub, toy, mock, optional-real-compile, optional-real-runtime candidates, and remote/external labels are not confused.
4. Prove default CI remains no-real-compute and offline/deterministic.
5. Keep Minimal Spine v1.x independent and avoid Gateway, Evidence/Archive authority, policy override, output override, Blue-Brain, HH, microcircuit, DBM, Geist, Replay, and Capability integration changes.

Prompt 16 should not:

- Rename feature flags in code.
- Add runtime backend paths.
- Activate Real Compute.
- Add external services.
- Claim production real inference.

## 9. Open Questions

- Which names should be deprecated but kept as aliases, especially `CandleToyV1`, `BurnToyV1`, `compute-candle`, and `compute-burn`?
- Which feature flags are too risky to rename now because they are already used by CI, docs, or downstream scripts?
- Which backend is the first optional-real-compile hardening target: Candle, Burn, or another local dependency lane?
- Which docs are P0 cleanup beyond `README.md`, `runtime/ucf-compute/src/lib.rs`, and backend docs?
- Should compatibility adapter features `ai-candle` and `ai-burn` receive explicit stub metadata in their own crate or only through documentation first?
- What exact local artifact contract is sufficient to upgrade a future lane from optional-real-compile to optional-real-runtime without changing Evidence/Archive authority?

## 9. Prompt 16 Contract Hardening Addendum

This addendum records the small machine-readable contract added after the taxonomy plan. It is contract hardening only: it does not enable real compute, add a new backend implementation, add gateway integration, create policy/output override authority, change evidence/archive authority, or make Minimal Spine v1.x depend on compute.

### Backend identity contract

`runtime/ucf-compute` now exposes `BackendClass` and `BackendIdentity` as metadata/label types. The contract classifies backend lanes as `stub`, `toy`, `mock`, `optional-real-compile`, `optional-real-runtime`, `remote-external`, `experimental`, `deferred`, or `forbidden-for-now`, and carries deterministic/offline/external-service/runtime-inference/production-claim booleans.

Current class mapping is intentionally conservative:

| Path | Class | Runtime inference claim? | Offline? | External service required? | Production claim? |
|---|---|---:|---:|---:|---:|
| `CpuStubBackend`, `ComputeBackendKind::Stub`, `stub_v0` pack | `stub` | no | yes | no | no |
| `toy_v1`, `toy_lnn_v1` packs | `toy` | no | yes | no | no |
| test/mock identity helpers | `mock` | no | yes | no | no |
| `ComputeBackendKind::{Candle,Burn}`, `candle_toy_v1`, `candle_liquid_v1`, `burn_toy_v1` | `optional-real-compile` | no | yes | no | no |
| explicit future local fixture/golden identity helper | `optional-real-runtime` | yes, only when explicitly constructed | yes | no | no |
| `remote_v1` / remote proxy metadata | `remote-external` | no by default | no | yes | no |
| `worker_v1` | `experimental` | no | yes | no | no |

The tests assert that stub/toy/mock identities cannot be confused with real-runtime claims, optional-real-compile identities do not claim runtime inference, and remote/external identities are not default-safe.

### Readiness gate note

As observed in Prompt 15, `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` can hang reproducibly around the 300s timeout in this environment. Prompt 16 does not refactor the gate. Treat any local timeout as an environment/runtime observation for this prompt, not as evidence that compute was activated or changed.
