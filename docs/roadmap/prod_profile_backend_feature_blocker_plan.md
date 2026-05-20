# UCF Prod-Profile Backend Feature Blocker Plan

## 0. Purpose
- Plan only (no runtime behavior changes in this prompt).
- No prod-readiness claim.
- No gate weakening.

## 1. Baseline
- Branch: `work`
- HEAD: `7db677686e0d4db3fff142805a0818b1746c609d`
- Dirty state: clean at start.
- Workspace package count: 192.
- Context anchor: `docs/roadmap/workspace_prod_readiness_refresh.md`.

## 2. Blocker Reproduction
- Command: `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile prod --workspace-test-report ./out/workspace_test_report.json --out ./out/gate_report_prod_split.json`
- Result: expected fail in current default feature build.
- Message: `pack burn_toy_v1 requires feature backend-burn`.

## 3. Feature-Pack / Backend Inventory
| Concern | Path | Current behavior | Risk |
|---|---|---|---|
| `burn_toy_v1` definition | `runtime/ucf-compute/src/backend_pack.rs` | `BackendPackKind::BurnToyV1` parses/serializes as `burn_toy_v1`; identity class is `OptionalRealCompile`. | Burn naming can be over-read as runtime-ready. |
| `backend-burn` enforcement | `runtime/ucf-compute/src/feature_matrix.rs` | `validate_pack(BurnToyV1)` hard-fails without `backend-burn` (`pack burn_toy_v1 requires feature backend-burn`). | Prod gate blocks in default feature lane. |
| Feature declarations | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-ops/Cargo.toml` | `backend-burn` exists and aliases `compute-burn`; default features are stub/toy (no burn). | Local/default prod probe fails unless feature lane selected. |
| Prod profile requirement path | `runtime/ucf-ops/src/lib.rs`, `runtime/ucf-ops/tests/profile_ladder_v2.rs` | Prod ladder requires burn backend semantics and tests assert missing-feature fail text. | Ambiguity between compile feature requirement and runtime-readiness claims if undocumented. |
| Optional-real taxonomy | `docs/roadmap/real_compute_lane_inventory.md`, `docs/roadmap/compute_backend_naming_boundary_plan.md`, `runtime/ucf-compute/tests/optional_real_compile_gate.rs` | Burn/Candle lanes are explicitly bounded as compile-only optional-real (not OptionalRealRuntime). | Overclaim risk if CI/docs wording is loose. |
| CI lane shape | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, `docs/roadmap/compute_feature_ci_matrix.md` | Burn feature lane exists (non-blocking matrix), but prod-gate invocation/expectation is not clearly split as dedicated prod-compile lane in all docs. | Drift between operator expectations and current CI semantics. |

## 4. Semantics Decision
| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A — keep prod requiring backend-burn compile feature | partial | Requirement is codified/tested and should not be silently removed. | Can be misunderstood as runtime proof unless framed. |
| B — remove burn requirement from prod mapping | no (unless future drift proof) | Would weaken current explicit requirement without evidence of accidental mapping. | Gate weakening / policy drift. |
| C — keep default local prod fail and only document | partial | Accurate as a current fact, but insufficient alone for operator clarity. | Persistent confusion and repeated false blocker reports. |
| D — explicit prod-compile lane + docs/CI alignment | **yes (recommended)** | Preserves strict requirement while separating compile-feature proof from runtime readiness claims. | Needs careful wording to avoid overclaim. |

## 5. Fix Plan Matrix
| Fix option | Files likely touched | Acceptance criteria | Boundary guardrails | Risk |
|---|---|---|---|---|
| 1) Docs-only: prod requires backend-burn feature lane | `docs/readiness_gate.md`, roadmap docs | Prod gate docs explicitly show feature-enabled invocation and classify as compile-feature requirement only. | No code-path gate semantics change. | Medium (docs-only may not prevent CI drift). |
| 2) Add explicit CI prod-feature lane | `.github/workflows/ci.yml`, optional nightly workflow docs | CI contains clearly labeled prod compile-feature lane (`backend-burn`) with no runtime claims. | Keep non-runtime, no external services, no scheduler/worker activation. | Medium (CI time/maintenance). |
| 3) Update readiness-gate invocation guidance | `docs/continuous_verification.md`, `docs/readiness_gate.md` | Distinct commands documented for default lane vs prod-compile-feature lane. | No SKIP/PASS reinterpretation. | Low. |
| 4) Explicit split framing: `prod-gate-core` + `prod-compute-feature-pack` + runtime deferred | docs + possibly ops workflow wrappers/scripts | Operators can distinguish pass criteria by lane without weakening checks. | OptionalRealRuntime remains deferred and explicitly non-claimed. | Low/medium. |
| 5) Mapping correction (`burn_toy_v1` -> other pack) only if drift proven | `runtime/ucf-ops`/`runtime/ucf-compute` mapping code and tests | Only after explicit spec decision and evidence that current mapping is unintended. | No silent requirement removal; preserve determinism and test coverage. | High (behavior-changing). |

## 6. Recommended Fix Path
Choose Option D with A constraints:
1. Keep current `backend-burn` requirement intact.
2. Add explicit prod backend feature lane language in docs and CI plan.
3. Keep compile-only vs runtime distinction explicit (`OptionalRealCompile` != `OptionalRealRuntime`).
4. Avoid any runtime activation or production inference claims.

## 7. Non-Goals
- No runtime inference implementation/activation.
- No `OptionalRealRuntime` claim.
- No gate weakening or SKIP->PASS reinterpretation.
- No prod-readiness claim until fresh prod split gate evidence passes in the intended feature lane.

## 8. Recommended Next Prompt
**UCF Prompt 79B — Prod Backend Feature Lane Alignment**.

Rationale: current code/tests/docs indicate requirement is intentional; needed follow-up is lane/documentation/CI alignment, not immediate pack-remap.


## Prompt 79B — Backend Feature Lane Alignment

### Chosen option
- Option C (readiness-gate docs + compute CI matrix alignment).
- Rationale: keep `backend-burn` requirement unchanged and make the prod feature-lane invocation explicit without broad workflow churn in this prompt.

### Implemented changes
- `docs/readiness_gate.md` now documents the explicit prod split invocation using `cargo run -p ucf-ops --features backend-burn -- readiness-gate --profile prod ...` and states compile-only scope.
- `docs/continuous_verification.md` now includes the prod backend feature-lane probe commands and a no-runtime-inference boundary note.
- `docs/roadmap/compute_feature_ci_matrix.md` now includes a dedicated `prod-backend-feature-gate` lane row with exact commands and explicit non-runtime/non-production claim flags.

### Feature-lane command set
```bash
cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json
cargo run -p ucf-ops --features backend-burn -- readiness-gate --profile prod --out ./out/gate_report_prod_split.json --workdir ./.ucf_gate_prod --workspace-test-report ./out/workspace_test_report.json
```

### Validation outcome (Prompt 79B run)
- See validation table in prompt report for exact command outcomes.
- `backend-burn` requirement remains preserved and visible.
- No runtime inference claim added.

### Prod gate status
- Prod gate in backend-burn feature context: validated in this prompt run (pass/fail recorded in report table).
- If failing, blocker text is reported verbatim and carried forward.

### Boundary statement
- No OptionalRealRuntime activation/claim.
- No real-compute runtime activation.
- No gate weakening.
- No production-readiness claim unless fresh prod split evidence passes.

### Next prompt recommendation
- If prod split gate still fails: `UCF Prompt 79C — Prod Feature-Pack Mapping / Feature Propagation Fix`.
- If prod split gate passes but closure still needed: `UCF Prompt 79D — Prod Readiness Refresh After Backend Lane Alignment`.
