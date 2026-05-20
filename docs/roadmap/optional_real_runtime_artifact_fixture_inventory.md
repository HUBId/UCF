# UCF OptionalRealRuntime Artifact and Fixture Inventory

## 0. Purpose
- Inventory only.
- No runtime activation.
- No prod readiness claim.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD | `abc5014e52594f1a2ecfc0d9bd612385cd071be6` |
| Dirty state | clean |
| Workspace package count | 192 |
| OptionalRealRuntime roadmap present | yes |
| Compute closure present | yes |
| Compute CI matrix present | yes |
| `ucf-compute` present | yes |
| `ucf-ops` present | yes |

Related docs:
- [`docs/roadmap/optional_real_runtime_prod_compute_semantics_roadmap.md`](optional_real_runtime_prod_compute_semantics_roadmap.md)
- [`docs/roadmap/optional_real_runtime_pinned_local_fixture_plan.md`](optional_real_runtime_pinned_local_fixture_plan.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
- [`docs/roadmap/compute_feature_ci_matrix.md`](compute_feature_ci_matrix.md)

## 2. Backend / Artifact Inventory

| Concern | Path | Current behavior | Runtime-fixture relevance | Risk |
|---|---|---|---|---|
| Backend classes include OptionalRealRuntime contract symbol. | `runtime/ucf-compute/src/contracts.rs` | `BackendClass` includes `OptionalRealRuntime`; identity constructors exist. | Contract hook is available for later runtime lane wiring. | High overclaim risk if symbol presence is mistaken for implemented runtime lane. |
| Active Candle/Burn backend identities are OptionalRealCompile. | `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/tests/backend_identity_contract.rs` | Candle/Burn map to compile-only taxonomy and are tested as non-runtime inference. | Confirms compile-only baseline and protects against accidental runtime claims. | Runtime lane absent, prod remains blocked for runtime semantics. |
| Optional-real compile guard is explicit. | `runtime/ucf-compute/tests/optional_real_compile_gate.rs` | Compile-only lanes require `runtime_inference_supported=false`, `claims_runtime_real_inference=false`, `production_claim=false`. | Good safety rail for pre-runtime phase. | Needs additive runtime-class tests, not relaxation. |
| Existing backend/pack kinds. | `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs` | Backend kinds include Stub/Candle/Burn/Worker; pack kinds include `stub_v0`, `toy_v1`, `candle_toy_v1`, `burn_toy_v1`, etc. | Shows current deterministic fixture pathways are stub/toy-focused. | Toy naming could be over-read as real-runtime if docs are imprecise. |
| Model slot/artifact handling exists. | `runtime/ucf-compute/src/model_store.rs`, `runtime/ucf-compute/src/candle_weights.rs`, `runtime/ucf-compute/src/backends/candle_backend.rs`, `runtime/ucf-compute/src/backends/burn_backend.rs` | Local slot loading + verification paths and model hash flow exist, with disabled fallback handling. | Reusable substrate for future pinned OptionalRealRuntime fixture. | No approved OptionalRealRuntime artifact contract yet. |
| Hash/digest coverage exists in compute outputs and audit linkage. | `runtime/ucf-compute/tests/toy_compute_golden.rs`, `runtime/ucf-compute/tests/compute_output_link.rs`, `runtime/ucf-compute/tests/compute_audit_records.rs` | Deterministic digest assertions exist for toy/stub outputs and derived metadata records. | Confirms digest-based golden style is feasible. | No runtime-class golden fixture currently pinned. |
| Local fixtures/testdata exist (toy/stub + small weights fixtures). | `runtime/ucf-compute/tests/stub_compute_fixture.rs`, `runtime/ucf-compute/tests/toy_compute_golden.rs`, `runtime/ucf-compute/fixtures/` | Deterministic local fixture inputs are available and offline. | Pattern can be adapted for runtime fixture once artifact constraints are defined. | Current fixtures are explicitly non-runtime proof. |
| Candle/Burn feature lanes exist as compile checks. | `runtime/ucf-compute/Cargo.toml`, `.github/workflows/ci.yml`, `docs/roadmap/compute_feature_ci_matrix.md` | `backend-burn`, `backend-candle`, `compute-burn`, `compute-candle` lanes compile/check; CI labels compile lane coverage. | Confirms optional feature wiring for candidate backend family selection. | Compile pass alone is not runtime evidence. |
| ComputeOutputLink and ComputeAuditRecord exist as derived metadata only. | `runtime/ucf-compute/tests/compute_output_link.rs`, `runtime/ucf-compute/tests/compute_audit_records.rs` | Records are bounded metadata and non-authoritative. | Runtime fixture lane can integrate here without authority drift. | Must preserve non-authority boundary. |
| External/remote compute surfaces are non-default. | `runtime/ucf-compute/Cargo.toml` (`remote-compute`), docs in `docs/roadmap/compute_feature_ci_matrix.md` | Remote lane is optional compile surface and excluded from default/runtime claims. | Supports no-external-service-by-default requirement. | Remote candidate would raise determinism and overclaim risk. |
| Offline-first constraints are documented for gate workflows. | `docs/readiness_gate.md`, `docs/continuous_verification.md`, `.github/workflows/ci.yml` | Workspace tests and gates are explicitly offline-oriented and timeout semantics are strict (`TIMEOUT` not pass). | Sets strong precondition for deterministic runtime fixture design. | Any runtime lane must preserve offline determinism and strict timeout handling. |

## 3. OptionalRealRuntime Requirements Matrix

| Requirement | Required? | Existing? | Current evidence | Gap |
|---|---:|---:|---|---|
| explicit `BackendClass::OptionalRealRuntime` | yes | partial | Class symbol exists in contracts. | Not wired to active backend/pack mapping with tests. |
| `runtime_inference_supported = true` | yes | no | Compile-only tests enforce false. | Add runtime-class identity + tests. |
| `claims_runtime_real_inference = true` | yes | no | Compile-only tests enforce false. | Add explicit runtime claim contract for runtime lane only. |
| `production_claim = false` unless separately proven | yes | yes | Compile lanes already false and guarded. | Keep false for first runtime fixture line. |
| no external service by default | yes | partial | Remote lane is optional/non-default. | Runtime fixture spec must codify local-only default path. |
| offline deterministic fixture | yes | no | Deterministic fixtures exist only for stub/toy. | Add OptionalRealRuntime deterministic fixture. |
| pinned local artifact path | yes | no | No runtime-class artifact path contract. | Define canonical path + review process. |
| artifact hash/digest | yes | partial | Hash/digest infrastructure exists broadly. | Add explicit runtime artifact hash pin and validation tests. |
| artifact license/source note | yes | no | No runtime artifact introduced yet. | Add metadata requirement before artifact addition. |
| stable input fixture | yes | no | Toy/stub inputs stable. | Define runtime input fixture canonical format. |
| stable output digest | yes | no | Toy/stub digest stability tested. | Add runtime output digest stability test. |
| golden output bytes or digest | yes | no | Toy golden present only. | Add runtime golden bytes/digest fixture. |
| failure-mode test | yes | partial | Compile/disabled-path failures tested in current lanes. | Add runtime fixture failure-mode tests (missing artifact/hash mismatch/disabled). |
| timeout/cost bound | yes | partial | Gate docs enforce timeout semantics generally. | Add runtime fixture-specific bounded budget/timeout contract. |
| feature gate | yes | yes | Optional backend features already exist. | Add dedicated OptionalRealRuntime feature or explicit lane gate mapping. |
| CI lane explicit opt-in | yes | partial | Compile-only matrix documented; no runtime lane. | Add opt-in runtime fixture CI job when fixture exists. |
| docs guard against production claim | yes | yes | Existing overclaim guard language present across compute docs. | Extend guard text to runtime fixture lane explicitly. |
| `ComputeOutputLink` integration | yes | partial | Link record tests exist for current lanes. | Add runtime fixture coverage into link tests. |
| `ComputeAuditRecord` integration | yes | partial | Audit record tests exist for current lanes. | Add runtime fixture coverage into audit tests. |
| prod gate distinguishes compile-only vs runtime | yes | partial | Current docs state compile-only `backend-burn` lane is insufficient. | Add explicit prod runtime criterion when runtime lane lands. |

## 4. Candidate Backend Evaluation

| Candidate | Pros | Cons | Artifact availability | Determinism risk | Overclaim risk | Recommendation |
|---|---|---|---|---|---|---|
| A. Burn-based local fixture | Existing Burn feature lanes and pack taxonomy; mirrors current prod compile-lane framing. | Burn runtime semantics still compile-only; no approved runtime artifact fixture yet. | Not established for OptionalRealRuntime in current repo docs/tests. | Medium until artifact and golden are pinned. | Medium/high if confused with `backend-burn` compile lane. | Candidate after explicit artifact schema + hash pin plan. |
| B. Candle-based local fixture | Candle model-slot/weights path is already explicit and artifact-oriented. | Still currently compile-only claims at lane level; no runtime-class fixture yet. | Not established for OptionalRealRuntime in current docs/tests. | Medium until deterministic runtime golden is proven. | Medium if compile support is misreported as runtime-ready. | Candidate after contract/schema planning; comparable to Burn. |
| C. Tiny in-repo synthetic artifact backend | Max control over offline determinism and artifact pinning surface. | Can drift toward toy-like semantics; may not prove intended real-runtime family boundary. | Could be created, but absent now. | Low/medium if tightly scoped and pinned. | Medium if mistaken as sufficient for production runtime claim. | Possible only as clearly bounded OptionalRealRuntime fixture scaffold. |
| D. Promote existing Toy backend | Deterministic and already tested. | Violates compile-vs-runtime boundary and overclaim guard intent. | Existing, but not real-runtime evidence. | Low technically, high semantic mismatch. | High. | **Do not choose.** |
| E. Remote/external backend | Could represent operational runtime path conceptually. | Violates local/offline-first default and introduces network/service dependencies. | No governed runtime service evidence in current scope. | High. | High. | **Do not choose.** |

## 5. Policy Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| Option A — Docs-only inventory now, no implementation | no | Useful baseline, but inventory indicates a clear need for artifact-contract planning next. | Could stall progress if repeated without schema decisions. |
| Option B — Plan a tiny local artifact fixture (still no implementation now) | **yes** | Safest forward path: keep current no-activation boundary while specifying contract + artifact pin prerequisites. | Planning quality risk if schema remains underspecified. |
| Option C — Promote existing compile backend | no | Explicitly forbidden by current boundaries; no runtime fixture evidence exists. | Overclaim and gate-semantics regression risk. |
| Option D — Split prod-compute profiles first | maybe later | Useful as follow-up if gate semantics clarity still needed after runtime contract definition. | Profile complexity before runtime contract could add churn. |

## 6. Refined Prompt Series

| Prompt | Title | Goal | Acceptance criteria | Guardrails |
|---:|---|---|---|---|
| 79G | OptionalRealRuntime Backend Contract and Artifact Schema | Define runtime-class backend contract, artifact metadata schema, and claim boundaries. | `OptionalRealRuntime` contract fields and validation checklist documented with tests-to-add plan. | No runtime activation, no prod claim. |
| 79H | Pinned Local Runtime Artifact Fixture Plan | Specify local artifact location, hash pinning, license/source note, and review/update workflow. | Canonical artifact path + hash format + ownership workflow documented. | Offline-first, no external service default. |
| 79I | Deterministic Runtime Golden Test Implementation | Implement deterministic OptionalRealRuntime fixture tests with stable output digest. | Stable input fixture + output digest golden + failure-mode tests pass. | No promotion to production claim. |
| 79J | OptionalRealRuntime ComputeOutputLink / Audit Integration | Extend link/audit tests to runtime fixture lane without authority drift. | Runtime fixture emits bounded link/audit metadata coverage. | No Evidence/Archive authority expansion. |
| 79K | Prod Compute Runtime Gate Wiring | Add explicit gate criterion that distinguishes compile-only vs runtime evidence. | Prod gate semantics require runtime fixture evidence for runtime claim. | No gate weakening; timeout remains fail. |
| 79L | Prod Compute Runtime Docs Overclaim Guard | Update docs to prevent runtime/prod overclaim from compile or fixture-only outcomes. | Explicit prohibited-claim text merged in readiness/compute docs. | Keep production claim false unless separately proven. |
| 79M | Prod Compute Runtime Readiness Refresh | Re-run fresh evidence after prior prompts and restate blocker/readiness accurately. | Fresh reports + explicit status matrix with no stale-evidence reuse. | No stale reports as current truth. |

## 7. Current Prod Status
- Prod ready: no.
- Blocker: OptionalRealRuntime absent.
- `backend-burn` compile lane: pass/available as compile evidence but not sufficient for runtime readiness.

## 8. Open Questions
- Which backend first: Burn or Candle?
- Where should local artifacts live?
- How large can artifacts be?
- How should hashes be pinned and reviewed?
- How should licenses/sources be recorded?
- What deterministic input/output fixture format should be canonical?
- What timeout/cost bound should runtime fixture tests enforce?
- How should CI stay optional and explicit for runtime fixture lanes?
- How should `production_claim=false` be preserved for first runtime fixture stages?

## 9. Recommended Next Prompt
**UCF Prompt 79H2 — OptionalRealRuntime Artifact Format Decision** (or directly 79I if format is already approved).

Reason: Prompt 79H established the pinned-local fixture policy baseline; the remaining blocker before implementation is explicit artifact-format lock and then deterministic fixture/golden wiring.

## Prompt 79F-R — Validation Completion

The Prompt 79F inventory was revalidated after cleanup of generated reports. Workspace tests and workspace clippy completed successfully. This does not change the OptionalRealRuntime status: OptionalRealRuntime remains absent and roadmap-only, backend-burn remains compile evidence only, and no prod-readiness claim is made.

Recommended next prompt: UCF Prompt 79G — OptionalRealRuntime Backend Contract and Artifact Schema.

## Prompt 79G — Contract/Schema Layer (Metadata-Only)

Status: implemented as metadata contract types and tests; no runtime activation.

Added contract types in `runtime/ucf-compute/src/runtime_contract.rs`:
- `OptionalRealRuntimeArtifactSpec`
- `OptionalRealRuntimeFixtureSpec`
- `OptionalRealRuntimeCandidateContract`
- `OptionalRealRuntimeContractError`

Added validation test coverage in:
- `runtime/ucf-compute/tests/optional_real_runtime_contract.rs`

Boundary remains unchanged:
- No backend promoted to `OptionalRealRuntime` in active mappings.
- No runtime inference execution added.
- `production_claim` remains forbidden for this contract.

## Prompt 79G-S — Remaining Validation Completion

The OptionalRealRuntime contract validation was completed after the previous long-running workspace-test phase. `cargo test --workspace` and workspace clippy completed successfully. This does not change runtime status: OptionalRealRuntime remains absent and roadmap-only, no backend is promoted, backend-burn remains compile evidence only, and no prod-readiness claim is made.

Recommended next prompt: UCF Prompt 79H — Pinned Local Runtime Artifact Fixture Plan.
