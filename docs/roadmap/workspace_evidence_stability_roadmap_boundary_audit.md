# UCF Workspace Evidence Stability Roadmap and Boundary Audit

## 0. Purpose
- Roadmap/boundary audit only.
- No gate weakening.
- No UCF semantics change.
- No timeout-as-pass.

## 1. Baseline
- Branch: `work`
- HEAD: `84e2142c1ce10b8df118b1f2c7e8c1055080d410`
- Dirty state: clean
- Workspace package count: 192
- Links:
  - [docs/roadmap/post_archive_roadmap_selection.md](./post_archive_roadmap_selection.md)
  - [docs/roadmap/readiness_gate_timeout_stability_audit.md](./readiness_gate_timeout_stability_audit.md)
  - [docs/readiness_gate.md](../readiness_gate.md)
  - [docs/artifact_convention_v0.md](../artifact_convention_v0.md)

## 2. Workspace Evidence / Gate Surface Inventory

| Concern | Existing surface | Path | Current behavior | Risk |
|---|---|---|---|---|
| workspace-test-check definition | `workspace_test_check` | `runtime/ucf-ops/src/lib.rs` | Defines and runs canonical workspace evidence command and writes report JSON with phase timings. | Timeout/no-report on cold cache blocks split evidence.
| exact workspace command | `WORKSPACE_TEST_CHECK_COMMAND` | `runtime/ucf-ops/src/lib.rs`; `docs/readiness_gate.md` | Command is `cargo test --workspace --offline`; gate expects command match. | Wrong-command report must fail.
| workspace report schema | `WorkspaceTestReport`, `WorkspaceTestCommandResult`, metadata, `phase_timings` | `runtime/ucf-ops/src/lib.rs` | Includes `generated_at_utc`, `command`, `git_head_full`, `git_head_short`, `git_dirty`, command exit/success and timings. | Stale/mismatched metadata invalidates evidence.
| readiness-gate split-evidence validation | `check_workspace_test_report` / `validate_workspace_test_report` | `runtime/ucf-ops/src/lib.rs`; `docs/readiness_gate.md` | Missing/stale/dirty-mismatch/wrong-command/non-PASS report fails `build_workspace_tests`; never converted to PASS. | Overclaim risk if operators ignore failure semantics.
| env-driven split evidence wiring | `UCF_GATE_WORKSPACE_TEST_REPORT` | `runtime/ucf-ops/src/lib.rs` | If set, gate consumes external workspace report instead of in-process run. | Stale external path misuse.
| skip path | `UCF_SKIP_GATE_WORKSPACE_TESTS` and CI bypass | `runtime/ucf-ops/src/lib.rs`; `docs/roadmap/readiness_gate_timeout_stability_audit.md` | Gate can SKIP internal workspace tests for diagnostics/CI contexts; does not redefine PASS semantics for missing split evidence. | Misinterpretation of diagnostic skip as readiness claim.
| gate phase instrumentation | `GatePhaseTiming`, gate logs | `runtime/ucf-ops/src/lib.rs` | Gate emits per-phase start/done and stores `phase_timings[]`; workspace-test-check emits metadata/command/report phases. | Limited per-package detail for long workspace phase.
| CI split-evidence usage | workspace-test-check + readiness-gate `--workspace-test-report` steps | `.github/workflows/ci.yml` | CI has split-evidence test-profile readiness lanes using generated workspace report artifact path. | CI/local divergence if local path/timeouts differ.
| nightly split-evidence usage | workspace evidence + gate in nightly | `.github/workflows/nightly_verify.yml` | Nightly runs `workspace-test-check` before test-profile gate with explicit report file. | Nightly runtime variance/cold cache variance.
| prod-profile surfaces | `required_stage_profile`, `required_records` checks | `runtime/ucf-ops/src/lib.rs`; `docs/readiness_gate.md` | `test` profile skips prod-only checks; `prod` enforces required stage/record checks. | Prod/test confusion risk.
| report freshness convention | root `out/*.json` as generated artifacts | `docs/artifact_convention_v0.md`; `docs/readiness_gate.md` | Reports are freshness-bound artifacts; root reports are not enduring truth across HEAD changes. | Stale report reuse in docs/claims.

## 3. Historical Failure / Pass Inventory

| Run / Source | Command | Result | Duration / phase | Artifact produced? | Notes |
|---|---|---|---|---:|---|
| Timeout stability audit | `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT (`124`) | 300s | no | Initial timeout baseline documented.
| Timeout stability audit | `UCF_OFFLINE=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT (`124`) | 300s | no | Offline flag alone did not remove timeout.
| Timeout stability audit | `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` after warm-up | PASS | 213s | yes | Demonstrates cold/warm sensitivity.
| Timeout stability audit | `timeout 300s cargo test --workspace --offline` | TIMEOUT (`124`) | 300s | no | Isolated dominant bottleneck.
| Post-archive selection and closure docs | `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json` | Recurrent timeout caveat in selection line | 600s guard in plan | often no | Missing fresh report blocks split gate claims.
| Timeout stability audit | `UCF_OFFLINE=1 UCF_SKIP_GATE_WORKSPACE_TESTS=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_skip_workspace.json` | PASS (diagnostic) | 3s | yes | Non-workspace phases are fast; not a gate weakening.
| Timeout stability audit | `cargo test --workspace` | PASS (after warm-up) | not fixed in doc table | n/a | Useful signal but not equivalent to canonical workspace evidence report.
| Nightly/CI workflows | split-evidence pipeline commands | configured | workflow-level | yes in workflow | CI and nightly both model split-evidence path.

## 4. Boundary Decisions

| Boundary | Decision | Reason |
|---|---|---|
| timeout handling | Timeout is FAIL/caveat, never PASS. | No fresh complete evidence exists on timeout.
| missing workspace report | Missing report is FAIL for split evidence. | Gate requires concrete report file and parse/validation success.
| stale report | Stale HEAD/dirty mismatch is FAIL. | Freshness metadata binds evidence to current repo state.
| dirty mismatch | Dirty mismatch is FAIL. | Prevents claiming clean-state evidence from different working tree.
| direct cargo test relation | Direct `cargo test --workspace` is informative but not equivalent to split-evidence report unless policy is explicitly changed/versioned. | Split evidence requires command+metadata+status in report object.
| split evidence | Acceptable only when report is fresh, PASS, canonical-command matching, and state matching. | Prevents bypass and stale report reuse.
| prod profile | Prod readiness remains separate from test profile and requires prod-only checks. | `required_stage_profile` / `required_records` semantics differ by profile.
| root reports | Root reports are generated artifacts, not self-validating truth when stale. | Artifact convention and freshness policy.

## 5. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| timeout misclassified as pass | critical | Timeout audit entries (300s and 600s caveats). | Enforce explicit timeout != pass rule in docs/tests.
| stale report used as current truth | critical | Freshness mismatch noted in audits. | Validate HEAD/dirty/command/status before acceptance.
| direct cargo test conflated with workspace report | high | Docs explicitly separate both signals. | Keep split-evidence contract tied to report schema.
| split-evidence bypass | critical | Env/path-based split evidence can be misused if unchecked. | Keep strict `validate_workspace_test_report` checks.
| CI/nightly divergence from local policy | high | Separate workflow lanes and local runs can differ in cache/time. | Keep common command contract + freshness checks.
| missing phase detail | medium | Phase timings exist but workspace phase remains coarse. | Prompt 73 phase decomposition/timing expansion.
| prod/test profile confusion | high | test profile skips prod-only checks. | Keep explicit profile-specific claims and docs guard.
| root report self-reference confusion | medium | Root out reports often present historically. | Mark generated artifacts as freshness-bound.
| report committed with stale HEAD | high | Multiple docs mention stale-root caveats. | Avoid committing root reports; regenerate when needed.
| workspace breadth/cold-cache variance | high | 192-package workspace and observed timeout variance. | Use adequate timeout budgets + diagnostic phase timing.

## 6. Target Scope

| Layer | Goal | Inputs | Outputs | Explicit non-goals |
|---|---|---|---|---|
| Workspace timing decomposition | Attribute workspace-test-check duration by deterministic phases/packages where feasible. | `workspace_test_check`, current phase timings, timeout observations. | Timing report design/tests/docs. | No gate criteria change; no timeout PASS.
| Evidence freshness enforcement | Harden/verify stale/missing/non-PASS rejection. | Existing validator/tests and docs. | Additional tests and clarified failure semantics. | No bypass switch, no skip-to-pass path.
| Split-evidence CI alignment | Keep CI/nightly/local command/report contract aligned. | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, gate docs. | Alignment plan and bounded workflow/doc updates when prompted. | No broad CI redesign.
| Prod-profile audit | Inventory prod required checks/records and skips. | `readiness_gate` checks and docs. | Gap report and bounded checklist updates. | No prod readiness claim without evidence.
| Overclaim guard | Keep readiness/prod wording bounded by fresh evidence. | roadmap + current state docs. | Doc guard updates and references. | No semantic/runtime feature changes.
| Evidence refresh discipline | Require fresh reports after relevant changes. | artifact convention + gate docs. | Refresh checklist and caveat language. | No committing stale root reports as authority.

## 7. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 73 | Workspace-Test-Check Phase Decomposition and Timing Report | Decompose workspace-test-check timing into actionable deterministic phases. | `runtime/ucf-ops` instrumentation/tests/docs. | Phase timing output pinpoints timeout locus; deterministic output. | No timeout->PASS, no gate weakening.
| 74 | Workspace Evidence Freshness Enforcement Tests | Expand tests for stale/missing/wrong-command/non-PASS evidence rejection. | Validation tests in `runtime/ucf-ops`. | Deterministic FAIL on invalid evidence paths. | No bypass mode; no silent SKIP/PASS conversion.
| 75 | Readiness-Gate Split Evidence CI Alignment | Align CI/nightly split evidence invocation and path conventions. | Workflow/doc alignment only. | Both CI and nightly consume fresh generated workspace report before gate. | No criteria relaxation.
| 76 | Prod-Profile Readiness Inventory and Gap Report | Inventory prod checks, required records, and remaining caveats. | Docs+inventory only. | Explicit prod readiness gap table with no overclaim. | No prod semantic/runtime rollout.
| 77 | Prod-Profile Required Records / Skips Audit | Clarify required vs skip-permitted checks for prod profile. | Gate docs/tests/policy wording. | Deterministic classification and remediation hints. | No test-profile semantics drift.
| 78 | Prod-Profile Docs Overclaim Guard | Add stronger wording boundaries around readiness/prod claims. | Docs-only updates. | Overclaim-resistant wording merged and linked. | No behavior change.
| 79 | Workspace/Prod Readiness Refresh | Re-run bounded evidence chain and publish fresh status with caveats. | Validation execution + docs refresh. | Fresh evidence set with explicit timeout handling. | Timeout/missing/stale not pass.
| 80 | Post-Prod-Roadmap Selection: Gateway vs Runtime Scheduler vs Identity Anchor | Re-rank next line after evidence/prod hardening. | Planning doc only. | Clear prioritization with dependencies/risks. | No implementation.

## 8. Open Questions
- Is 600s sufficient across CI/local cold cache?
- Should workspace-test-check expose per-package phase timings?
- Should it stream child cargo output or summarize package phases?
- How should direct cargo test --workspace relate to workspace-test-check?
- Should prod-profile require split evidence always?
- Should root reports remain uncommitted generated artifacts?
- What is acceptable CI/local divergence?

## 9. Recommended Next Prompt
UCF Prompt 74 — Workspace Evidence Freshness Enforcement Tests
