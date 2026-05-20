# UCF OptionalRealRuntime / Prod Compute Semantics Roadmap

## 0. Purpose
- This document is a roadmap only.
- No runtime activation is performed or claimed.
- No prod-readiness claim is made.

## 1. Baseline
- Branch: `work`
- HEAD: `bae0f7779135aaf14f3cea5bbcc795408e25c113`
- Dirty state: clean at baseline capture.
- Workspace package count: 192.
- Related docs:
  - [`docs/roadmap/prod_profile_backend_feature_blocker_plan.md`](prod_profile_backend_feature_blocker_plan.md)
  - [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
  - [`docs/roadmap/compute_backend_naming_boundary_plan.md`](compute_backend_naming_boundary_plan.md)
  - [`docs/roadmap/compute_feature_ci_matrix.md`](compute_feature_ci_matrix.md)
  - [`docs/roadmap/optional_real_runtime_pinned_local_fixture_plan.md`](optional_real_runtime_pinned_local_fixture_plan.md)

## 2. Backend Identity Inventory

| Concern | Path | Current behavior | Risk |
|---|---|---|---|
| `OptionalRealRuntime` definition exists but is not wired into active backend kinds/packs. | `runtime/ucf-compute/src/contracts.rs`, `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs` | `BackendIdentity::optional_real_runtime(...)` exists; active Candle/Burn backend and pack identities report `OptionalRealCompile`. | Compile-only identity can be overread as runtime-ready if docs/gates are imprecise. |
| Optional-real compile is explicitly non-runtime and non-production. | `runtime/ucf-compute/tests/optional_real_compile_gate.rs`, `runtime/ucf-compute/tests/backend_identity_contract.rs` | Tests require `runtime_inference_supported=false`, `claims_runtime_real_inference=false`, and `production_claim=false` for OptionalRealCompile lanes. | Any future gate relaxation could create overclaim drift. |
| No current backend reports OptionalRealRuntime in shipped backend mappings. | `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/src/backend_pack.rs`, `runtime/ucf-compute/tests/backend_identity_contract.rs` | Stub/Toy/OptionalRealCompile/RemoteExternal identities are covered; OptionalRealRuntime remains explicit constructor-only contract. | Prod runtime expectations remain blocked until explicit runtime-class lane is implemented and evidenced. |
| Prod blocker text: compute backend disabled. | `runtime/ucf-ops/tests/profile_ladder_v2.rs`, `docs/roadmap/prod_profile_backend_feature_blocker_plan.md` | Prod split gate no longer fails on missing `backend-burn` feature only; current blocker is compute backend disabled/runtime semantics not satisfied. | Misclassification as mere compile feature issue would hide runtime readiness gap. |
| Artifact/model surfaces exist but current compute fixtures stay synthetic. | `runtime/ucf-compute/tests/stub_compute_fixture.rs`, `runtime/ucf-compute/tests/toy_compute_golden.rs`, `runtime/ucf-compute/tests/compute_output_link.rs`, `runtime/ucf-compute/tests/compute_audit_records.rs` | Stub and toy are deterministic/offline/non-real; model hash digests are zeroed in fixture lanes; link/audit records are derived metadata only. | No pinned local real-runtime artifact proof path yet. |
| CI documents compile lanes, not runtime-inference proof. | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, `docs/roadmap/compute_feature_ci_matrix.md` | Backend-burn/candle feature checks are compile/test coverage; runtime OptionalRealRuntime evidence lane is absent. | Operators may infer runtime confidence from compile-only CI lanes. |

## 3. OptionalRealRuntime Semantics

| Requirement | Required for OptionalRealRuntime? | Current status | Gap |
|---|---:|---|---|
| Backend identity explicitly classed as `OptionalRealRuntime`. | yes | Constructor exists in contracts; no active backend/pack mapping uses it. | Add explicit runtime-class backend identity mapping with tests. |
| `runtime_inference_supported=true` and runtime claim contract true. | yes | OptionalRealCompile lanes are explicitly false. | Add runtime lane with explicit true plus negative guards on compile-only lanes. |
| No `production_claim` unless separately proven. | yes | Current compute identities keep `production_claim=false`. | Preserve false default; add dedicated production-proof process if ever needed. |
| Pinned local artifact only; no default network/external dependency. | yes | Stub/Toy are offline and non-external, but not real-runtime artifact backed. | Define allowed local artifact class and pinning workflow. |
| Deterministic runtime fixture exists. | yes | Deterministic fixtures exist only for stub/toy. | Create runtime deterministic fixture contract for optional runtime lane. |
| Artifact hash recorded and reviewed. | yes | No real-runtime artifact hash contract in compute tests/docs. | Add artifact hash manifest + test assertions. |
| Runtime output digest/golden pinned. | yes | Toy golden pinned; no OptionalRealRuntime golden exists. | Add runtime golden digest pinning for optional runtime lane. |
| Failure modes deterministic and test-covered. | yes | Deterministic failure checks exist for current synthetic lanes only. | Add deterministic runtime failure mode fixtures/tests. |
| No Gateway/action authority implications. | yes | Current docs state compute lane is non-authoritative and optional. | Maintain explicit boundary statements in runtime-lane docs/tests. |
| No policy mutation authority. | yes | No current compute lane grants policy mutation authority. | Keep explicit prohibition in runtime lane acceptance criteria. |
| No Minimal Spine dependency introduced. | yes | Minimal spine remains compute-independent in existing roadmaps/docs. | Keep runtime lane optional and non-spine-blocking. |
| No hidden evidence/archive authority changes. | yes | ComputeOutputLink/Audit are metadata-only in tests/docs. | Preserve metadata-only boundary in any runtime-lane additions. |
| CI lane explicit and non-default. | yes | Compile lanes are explicit; runtime lane not present. | Add dedicated opt-in runtime-lane CI job once semantics/tests exist. |
| Docs explicitly classify optional runtime as non-production-ready. | yes | Existing docs strongly protect against overclaim for compile-only lanes. | Add matching overclaim guard text for OptionalRealRuntime lane if introduced. |

## 4. Prod Gate Semantics

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A — Prod gate requires OptionalRealRuntime. | partial alignment | Aligns with strict runtime semantics and preserves blocker correctness. | Requires new runtime lane evidence before unblocking prod. |
| B — Prod gate accepts OptionalRealCompile for compile-only prod profile. | no | Redefines prod semantics and risks overclaiming runtime readiness. | High risk of policy/readiness overstatement. |
| C — Split profiles (`prod-compile` vs `prod-runtime`). | later candidate | Can reduce ambiguity while preserving strict runtime gate semantics. | Additional profile/governance complexity and naming drift risk. |
| D — Keep current prod blocked, add roadmap. | **yes (now)** | Safest interpretation consistent with current contracts and Prompt 79C outcome. | No immediate unblock; requires follow-up prompt series. |

## 5. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Guardrails |
|---:|---|---|---|---|---|
| 79F | OptionalRealRuntime Artifact and Fixture Inventory | Define candidate local runtime artifacts and deterministic fixture strategy. | Docs + inventory only. | Candidate list, pinning fields, offline constraints documented. | No runtime activation; no prod claim. |
| 79G | Runtime Backend Trait Contract for OptionalRealRuntime | Specify trait/identity contract boundaries for runtime lane. | Contract/tests design only (or minimal non-activating scaffolding if approved). | Contract explicitly separates OptionalRealCompile vs OptionalRealRuntime. | No scheduler/queue/worker; no gateway. |
| 79H | Pinned Local Runtime Artifact Fixture Plan | Define local artifact packaging, hash pinning, and reproducible loading. | Docs + fixture-plan metadata schema. | Hash pinning/review flow and offline loading rules defined. | No network/external service by default. |
| 79I | Deterministic OptionalRealRuntime Golden Test | Specify deterministic output digest strategy for runtime lane. | Test-plan and golden governance. | Deterministic fixture inputs, expected outputs, and digest policy defined. | No production claim from passing golden alone. |
| 79J | OptionalRealRuntime Evidence/Audit Metadata | Extend metadata boundaries to represent runtime lane without authority drift. | ComputeOutputLink/Audit schema boundary planning only. | Runtime-lane metadata fields mapped with non-authority guarantees. | No evidence/archive append authority expansion. |
| 79K | Prod Compute Runtime Gate Wiring | Plan strict prod gate condition for OptionalRealRuntime evidence. | Gate semantics + CI wiring plan. | Clear fail/pass conditions and required artifacts listed. | No gate weakening; no stale report acceptance. |
| 79L | Prod Compute Runtime Docs Overclaim Guard | Add docs safeguards against runtime/production overclaim. | Docs updates only. | Explicit forbidden-claim section merged where prod/compute docs reference runtime lane. | No readiness claim language. |
| 79M | Prod Compute Runtime Readiness Refresh | Refresh inventory after prior prompts and verify blocker state. | Evidence refresh + report only. | Updated blocker status with fresh command outputs and report timestamps. | Timeouts are not PASS; no prod claim without all evidence. |

## 6. Current Prod Status
- Prod split gate remains blocked by `compute backend disabled` semantics.
- `backend-burn` compile lane evidence is necessary but not sufficient.
- No prod-readiness claim is valid from current OptionalRealCompile evidence.

## 7. Open Questions
- Which backend should become the first OptionalRealRuntime candidate?
- Burn or Candle as first candidate?
- What local artifact class is allowed for the candidate?
- How is artifact hash pinning recorded and reviewed?
- How are runtime outputs made deterministic across environments?
- How is network/external service usage prevented by default?
- How is production-claim prohibition enforced in code/tests/docs?
- How should prod profile semantics distinguish compile vs runtime?
- Should `prod-compile` and `prod-runtime` be split profiles?

## 8. Recommended Next Prompt
**UCF Prompt 79G — OptionalRealRuntime Backend Contract and Artifact Schema**.

Rationale: Prompt 79F inventory is now captured in [`docs/roadmap/optional_real_runtime_artifact_fixture_inventory.md`](optional_real_runtime_artifact_fixture_inventory.md), and the immediate next gap is contract/schema definition before any runtime-fixture implementation.

## 8.1 Linked Inventory
- [`docs/roadmap/optional_real_runtime_artifact_fixture_inventory.md`](optional_real_runtime_artifact_fixture_inventory.md)

## Prompt 79E-R — Validation Completion

- workspace tests completed.
- clippy completed.
- no prod-readiness claim.
- OptionalRealRuntime remains roadmap-only.
- Recommended next prompt: UCF Prompt 79F — OptionalRealRuntime Artifact and Fixture Inventory.

## Prompt 79G update

A metadata-only OptionalRealRuntime candidate contract/schema layer now exists in
`runtime/ucf-compute/src/runtime_contract.rs` with validation tests in
`runtime/ucf-compute/tests/optional_real_runtime_contract.rs`.

This update does not activate any runtime backend and does not change production readiness.
Recommended next prompt remains: **UCF Prompt 79H — Pinned Local Runtime Artifact Fixture Plan**.
