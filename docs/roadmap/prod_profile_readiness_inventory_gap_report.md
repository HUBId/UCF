# UCF Prod-Profile Readiness Inventory and Gap Report

## 0. Purpose
- Inventory only.
- No prod readiness claim.
- No gate weakening.

## 1. Baseline
- Branch: `work`
- HEAD: `112ddc160c5fc0af3caf55ca83d2d03f2922cee5`
- Dirty state: clean at inventory start
- Workspace package count: 192
- Links:
  - `docs/roadmap/workspace_evidence_stability_roadmap_boundary_audit.md`
  - `docs/readiness_gate.md`
  - `docs/continuous_verification.md`

## 2. Readiness Profile Inventory

| Concern | Existing implementation/doc | Path | Current behavior | Prod relevance |
|---|---|---|---|---|
| Profile switch | `--profile test|prod` | `runtime/ucf-ops/src/main.rs`, `runtime/ucf-ops/src/lib.rs`, `docs/readiness_gate.md` | Explicit profile selector exists; gate report embeds selected profile. | Central to prod-vs-test distinction. |
| Required stage profile check | `required_stage_profile` gate check | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json` sample | `test` profile SKIP with remediation to run `--profile prod`; prod intended to enforce. | Likely prod blocker surface. |
| Required records check | `required_records` gate check | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json` sample | In test sample: SKIP when CandidateSet/Output/CapabilityIssuance missing in fixture bringup. | Needs explicit prod policy for missing record handling. |
| Feature-pack check | `feature_pack_disabled_fast_fail` | `runtime/ucf-ops/src/lib.rs`, `docs/readiness_gate.md`, `out/gate_report.json` sample | Can surface SKIP in test profile depending on fixture/backend path. | Must be classified for prod: skip-allowed vs fail-blocking. |
| Workspace split evidence | `workspace-test-check` + `--workspace-test-report` | `runtime/ucf-ops/src/lib.rs`, `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, `docs/readiness_gate.md` | Freshness-validated split evidence supported; wrong/stale/non-pass rejected by tests. | Required for trustworthy prod evidence claims. |
| Workspace canonical command | `cargo test --workspace --offline` contract | `runtime/ucf-ops/src/lib.rs`, `docs/readiness_gate.md`, snapshot schema | Command mismatch is rejected for split evidence. | Prevents non-canonical prod evidence substitution. |
| Replay checks | replay verify + recompute checks | `runtime/ucf-ops/src/lib.rs`, gate logs | Gate runs verify and recompute checks; status reported per check. | Prod likely needs explicit pass criterion per replay check. |
| Adversarial evidence | `adversarial-run --suite v1` in CI/nightly | `.github/workflows/nightly_verify.yml`, docs | Produced nightly; linked into nightly summary. | Prod hardening evidence lane; freshness needed if used for claims. |
| Goldens evidence | `goldens verify` in CI/nightly | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml` | Verified in CI/nightly; not automatically in local gate invocation. | Prod confidence lane; clarify mandatory scope. |
| Drift evidence | `drift report` best-effort nightly | `.github/workflows/nightly_verify.yml` | Nightly best-effort and conditionally summarized. | Prod relevance high, but policy currently optional/best-effort. |
| Artifact/schema checks | `spec artifact-schemas-check` + snapshots | docs snapshots, CI command set | Schema contracts are explicitly checked and documented. | Required integrity baseline for prod-significant reports. |
| PASS/FAIL/SKIP semantics | Explicit SKIP behavior documented | `docs/readiness_gate.md`, `docs/portability_gate.md`, gate outputs | SKIP exists at check-level; non-PASS must not be overclaimed as readiness. | Prompt 77 must define prod-acceptable SKIPs precisely. |

## 3. Report / Artifact Inventory

Inventory rule: root `out/*.json` is freshness-bound, not auto-authoritative.

| Artifact | Present? | Embedded HEAD? | Status | Fresh for current HEAD? | Prod relevance |
|---|---:|---|---|---:|---|
| `out/docs_lint_report.json` | yes | `5ab6a939...` | `pass` | no (stale head + dirty=true mismatch) | Medium (docs integrity), but stale locally. |
| `out/gate_report.json` | yes | `5ab6a939...` | `PASS` | no (stale head + dirty=true mismatch) | High, but not current evidence. |
| `out/workspace_test_report.json` | no fresh complete report | n/a | n/a | no | High for split evidence; missing fresh report blocks split-gate claims. |
| `out/readiness_spine_check.json` | not present at phase-3 scan | n/a | n/a | no | Medium-high for readiness consistency. |
| `out/artifact_schema_check.json` | not present at phase-3 scan | n/a | n/a | no | Medium-high for schema stability. |
| `out/goldens_report.json` | not present at phase-3 scan | n/a | n/a | no | Medium (nightly/CI lane). |
| `out/adversarial_report.json` | not present at phase-3 scan | n/a | n/a | no | Medium (nightly/CI lane). |
| `out/drift_report.json` | not present at phase-3 scan | n/a | n/a | no | Medium (best-effort nightly). |
| `out/nightly_summary.json` | not present at phase-3 scan | n/a | n/a | no | Medium summary artifact only when fresh inputs exist. |

## 4. Test/Prod Profile Probe Results

| Command | Result | Notes |
|---|---|---|
| `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json` | TIMEOUT (`124`) | Timed out in workspace cargo test phase; no timeout-as-pass interpretation. |
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_unsplit_diagnostic.json` | TIMEOUT (`124`) | Diagnostic unsplit run timed out at internal workspace test/offline check phase. |
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile prod --out ./out/gate_report_prod_unsplit_diagnostic.json` | FAIL (`exit 1`) | Early fail: backend pack feature requirement (`burn_toy_v1 requires backend-burn`); inventory signal only. |

## 5. Prod Readiness Gap Matrix

| Area | Current status | Test-profile behavior | Prod-profile expectation | Gap | Priority |
|---|---|---|---|---|---|
| Workspace evidence freshness | Fresh split report not available (timeout) | May rely on split report when provided; otherwise unsplit gate path | Fresh split evidence should be required and validated | Missing fresh split evidence path under local timeout pressure | P0 |
| Docs lint | Root report stale | Test gate not equivalent to docs-lint freshness | Fresh strict docs lint evidence per HEAD | Current root docs report stale | P1 |
| Readiness spine | Separate check exists; freshness currently run-dependent | Not always in test gate invocation | Fresh readiness-spine evidence for claimed prod readiness | Not guaranteed in local inventory run yet | P1 |
| Required records | Missing records surfaced as SKIP in test sample | SKIP in test fixture path | Explicit prod rule needed for missing required records | Acceptance semantics not fully codified for prod claims | P0 |
| Required stage profile | SKIP in test sample | SKIP in test by design | Enforced in prod profile | Prod pass criteria and expected evidence need explicit documentation | P0 |
| Feature pack | Can SKIP/fail depending on fixtures/backends | SKIP can occur in test | Prod must classify allowed skip vs blocker | Skip semantics not fully audited for prod | P1 |
| Replay verify/recompute | Checks exist | Runs and reports check status | Likely required PASS or explicit bounded SKIP policy | Prod acceptance rules not explicit enough | P1 |
| Drift report | Nightly best-effort | Not core test-gate local path | Prod claims should define drift evidence requirement | Optionality vs requirement unresolved | P2 |
| Goldens | CI/nightly coverage exists | Outside minimal local test gate | Prod readiness likely needs fresh golden evidence policy | Requirement boundary unclear | P2 |
| Adversarial suite | Nightly coverage exists | Outside minimal local test gate | Prod readiness likely needs fresh adversarial evidence policy | Requirement boundary unclear | P2 |
| Artifact schema snapshots | Snapshot + check infrastructure exists | Checked in CI | Prod claims should include fresh schema check | Not yet bound into explicit prod checklist doc | P1 |
| Spec snapshots | Snapshot tooling exists | Verified in nightly | Prod claims need explicit snapshot freshness policy | Policy link exists, but prod-specific criterion unclear | P2 |
| Nightly summary | Optional aggregate | Generated when inputs available | Useful prod observability, non-authoritative alone | Potential overclaim if consumed without freshness checks | P3 |
| CI Linux split evidence | Configured | Runs workspace report + test gate split path | Should remain mandatory for prod confidence | Local timeout parity issues still possible | P1 |
| CI Windows split evidence | Configured (Prompt 75 alignment) | Runs split workspace evidence path | Should match Linux policy semantics | Practical parity/failure mode criteria still need explicit definition | P1 |
| Windows/Linux parity | Partial policy alignment documented | Both lanes exist | Explicit acceptable parity baseline required | Pass/fail parity criteria not codified | P2 |
| Report freshness metadata | Present in key reports | Enforced in workspace split checks | Must be strict for prod evidence | Some artifacts still can be cited stale without discipline | P0 |
| Root report policy | Documented as generated artifacts | Can be misread if stale | Prod claims must reject stale root reports | Overclaim risk persists without stricter checklist language | P1 |
| Optional probes | Present in readiness and nightly ecosystem | May SKIP in test contexts | Prod must define optional vs required probe boundaries | Ambiguity remains | P2 |
| Prod-only blockers/SKIPs | Not fully enumerated in a dedicated matrix | Test includes intentional SKIPs | Prod must enumerate blocker-level checks | Missing explicit prod blocker matrix | P0 |

## 6. Prompt 77 Scope

| Prompt 77 target | Reason | Acceptance criteria |
|---|---|---|
| `required_records` audit | Test profile currently allows SKIP with missing records. | Complete table mapping each required record to test/prod expectation and blocker status. |
| `required_stage_profile` audit | Test SKIP is explicit; prod enforcement path needs precision. | Deterministic prod rule documented with examples of pass/fail/skip behavior. |
| `feature_pack` skip semantics | Current behavior mixes skip/fail contexts. | Clarified classification: allowable test SKIPs vs prod blockers, with remediation text. |
| Test vs prod expectation matrix | Avoid overclaim and profile confusion. | Single canonical table in docs mapping each gate check across profiles. |
| Prod blocker doc guard | Prevent stale/non-pass evidence from being cited as prod-ready. | Explicit “no stale/non-pass/no-timeout-as-pass” criteria section in readiness docs. |

## 7. Open Questions
- What is the exact prod profile pass criterion?
- Which SKIPs are acceptable in test but blockers in prod?
- Should prod require split workspace evidence always?
- Should prod require fresh root reports or external CI artifacts?
- How should optional probes be handled for prod claims?
- What is acceptable Windows/Linux parity?
- How do we prevent stale local reports from being cited as prod evidence?

## 8. Recommended Next Prompt
UCF Prompt 77 — Prod-Profile Required Records / Skips Audit


## 9. Prompt 77 Follow-up
- Prompt 77 audit added: `docs/roadmap/prod_profile_required_records_skips_audit.md`.
- Recommended next prompt: `UCF Prompt 78 — Prod-Profile Docs Overclaim Guard` (or `UCF Prompt 77A` for blocker planning first).


## 10. Prod-Profile Overclaim Guard

- Prod readiness is **not** claimed in this document.
- Test-profile `PASS` is not prod-profile `PASS`.
- `SKIP` is not `PASS`.
- `TIMEOUT` is not `PASS`.
- Missing workspace evidence is not `PASS`.
- Stale reports are not current truth.
- `cargo test --workspace` is useful but not a substitute for fresh `workspace-test-check` evidence.
- Split-evidence `readiness-gate` requires a fresh `workspace_test_report.json`.
- The prod backend feature requirement remains a blocker unless explicitly fixed and re-validated.
- Required records/skips must be resolved or policy-waived before any prod-readiness claim.
- Root `out/*.json` reports are generated artifacts; if stale, they are not self-validating truth.
- Minimal Spine and bounded UCF lines are not production-readiness claims.

## 11. Future Claim Checklist

### Before claiming prod readiness
- fresh workspace-test report exists and is `PASS`;
- readiness-gate prod profile passes with split evidence;
- `required_records` pass;
- `required_stage_profile` pass;
- feature-pack/backend feature requirements pass;
- docs lint passes;
- readiness-spine passes;
- artifact schema check passes;
- workspace tests pass;
- clippy passes;
- stale/root report caveats resolved;
- SKIPs are classified and acceptable by prod policy;
- prod blocker list is empty or explicitly waived by documented governance.

### Before claiming release readiness
- all prod-readiness criteria above;
- CI/nightly pass on required OSes;
- reports retained as CI artifacts;
- versioned release evidence bundle exists;
- no local stale reports are cited as current evidence.

## 12. Recommended Next Prompt

UCF Prompt 79 — Workspace/Prod Readiness Refresh

## Prompt 79S refresh link
- Refresh document: `docs/roadmap/workspace_prod_readiness_refresh.md`.
- Outcome snapshot: fresh workspace evidence PASS, test split PASS, prod split FAIL on backend feature requirement.
