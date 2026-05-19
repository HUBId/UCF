# UCF Prod-Profile Required Records / Skips Audit

## 0. Purpose
- Audit only.
- No prod readiness claim.
- No gate weakening.

## 1. Baseline
- Branch: `work`
- HEAD: `cbf54342f4cef4edaa802ed20368435dd29382d3`
- Dirty state: clean at audit start
- Workspace package count: 192
- Links:
  - `docs/roadmap/prod_profile_readiness_inventory_gap_report.md`
  - `docs/readiness_gate.md`
  - `docs/continuous_verification.md`

## 2. Required Records / Skips Code Inventory

| Concern | Existing implementation/doc | Path | Current behavior | Prod relevance |
|---|---|---|---|---|
| Required records gate | `required_records` check with warnings/remediation | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json` (stale sample) | Missing `CandidateSetRecord`, `OutputRecord`, `CapabilityIssuanceRecord` currently yields `SKIP` in test sample. | P0: prod policy must not overclaim if required records absent. |
| Required stage profile | `required_stage_profile` check | `runtime/ucf-ops/src/lib.rs`, `docs/readiness_gate.md`, `out/gate_report.json` (stale sample) | `test` profile: SKIP with remediation to run prod; prod path enforces. | P0 prod blocker surface. |
| Feature-pack/backend gate | `feature_pack_disabled_fast_fail` and scenario bringup errors | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json`, `out/gate_report_prod_unsplit_diagnostic.json` | Test sample may SKIP on missing optional feature (`lfm-candle`); prod probe fails on missing required backend feature (`backend-burn`). | P0 prod blocker in current environment. |
| Replay checks | `replay_verify_only`, `replay_recompute` | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json` (stale sample) | Simplified fixtures currently SKIP with drift remediation text. | P1 for strict prod evidence. |
| Optional probes | EBM/formal/tool/emergency lanes | `runtime/ucf-ops/src/lib.rs`, `out/gate_report.json` (stale sample) | Multiple checks can SKIP with explicit remediation; not equivalent to PASS. | P1/P2: classify allowed optional SKIPs explicitly for prod claims. |
| Workspace evidence split path | `workspace-test-check` + `--workspace-test-report` | `runtime/ucf-ops/src/lib.rs`, `docs/readiness_gate.md`, CI/nightly workflows | Missing/timeout report prevents split-gate proof; no timeout-as-pass. | P0 for fresh prod evidence chain. |

## 3. Gate Probe Results

| Command | Result | Artifact | Notes |
|---|---|---|---|
| `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json` | TIMEOUT | no fresh PASS report | Timed out (`EXIT:124`); split gate not provable. |
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_test_unsplit_diagnostic.json` | TIMEOUT | no completed report | Diagnostic only; timed out during internal workspace-test phase. |
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile prod --out ./out/gate_report_prod_unsplit_diagnostic.json` | FAIL | `out/gate_report_prod_unsplit_diagnostic.json` not emitted as PASS report; CLI failure observed | Fails early: `pack burn_toy_v1 requires feature backend-burn`. |
| `jq ... out/gate_report*.json` | DIAGNOSTIC_ONLY | stale `out/gate_report.json` inspected | Only stale test-profile report available for detailed SKIP inventory; not current truth for HEAD `cbf54342`. |

## 4. SKIP Classification

| Skip / Check | Current trigger | Test profile acceptable? | Prod profile blocker? | Required action |
|---|---|---:|---:|---|
| `required_stage_profile` | Running test profile | yes | yes | Enforce and document prod-only required-stage pass condition. |
| `feature_pack_disabled_fast_fail` | Missing optional fixture feature (`lfm-candle`) in test path | bounded/diagnostic only | yes when prod backend feature missing | Resolve prod backend feature-pack requirement and document distinction. |
| `required_records` | Missing `CandidateSetRecord`/`OutputRecord`/`CapabilityIssuanceRecord` | bounded for fixture test lane | yes for prod claim | Emit/verify required records or explicitly defer prod readiness. |
| `replay_verify_only` / `replay_recompute` | Drift on simplified fixture records | bounded with explicit caveat | likely blocker for strict prod evidence | Run full ESS audit-linked replay inputs. |
| `tool_deny_by_default` | No tool-intent in fixture run | potentially acceptable as optional lane | blocker only if prod policy marks required | Add dedicated tool-intent fixture if promoted to required. |
| `workspace evidence missing/stale` | timeout or missing split report | no (cannot claim split PASS) | yes | Regenerate fresh workspace report and split gate run. |
| `GPU/SAE/SSM optional lanes` | optional/unconfigured paths | yes if explicitly optional | depends on prod profile policy | Keep explicit optionality and never map SKIP to PASS. |

## 5. Required Records Matrix

| Required record | Current source / expected source | Current status | Test-profile role | Prod-profile role | Gap | Priority |
|---|---|---|---|---|---|---|
| CandidateSetRecord | Runtime decision/ESS record | Missing in stale test sample | can SKIP in fixture lane | required for prod claim | not proven fresh | P0 |
| OutputRecord | Runtime output/ESS record | Missing in stale test sample | can SKIP in fixture lane | required for prod claim | not proven fresh | P0 |
| CapabilityIssuanceRecord | Capability/tool issuance audit | Missing in stale test sample | often absent in baseline fixture | open question for prod | deferred semantics unresolved | P0/P1 |
| WorkspaceTestReport | `workspace-test-check` artifact | Timeout in this prompt | prerequisite for split gate claims | prerequisite for prod evidence freshness | no fresh PASS report | P0 |
| ReadinessGateReport | `readiness-gate` report | fresh test/prod split not proven | diagnostic signal only | required for bounded prod assertion | no fresh complete prod PASS | P0 |
| DocsLintReport | docs lint artifact | may exist but freshness-bound | supporting evidence | required freshness for release claims | refresh needed per HEAD | P1 |
| Artifact schema check/report | `spec artifact-schemas-check` + snapshots | command available; not yet rerun in this prompt section | supporting evidence | required schema integrity | refresh needed for this HEAD | P1 |
| Drift/Goldens/Adversarial reports | nightly/CI artifacts | not regenerated here | supplemental in test lane | expected prod hardening evidence | freshness and requirement boundary must be explicit | P2 |

## 6. Backend Feature / Feature-Pack Matrix

| Feature / Pack | Current status | Required by prod? | Test behavior | Prod behavior | Gap | Suggested next action |
|---|---|---:|---|---|---|---|
| `lfm-candle` for `candle_toy_v1` | unavailable in stale sample lane | not strictly required in current test lane | appears as SKIP diagnostic | not the observed prod blocker here | test fixture optional path still unresolved | keep as optional diagnostic unless policy promotes it. |
| `backend-burn` for `burn_toy_v1` | missing in current environment | yes (observed prod probe requirement) | n/a in timed-out unsplit test probe | hard FAIL at prod scenario bringup | prod probe blocked before full check set | Prompt 77A planning for backend feature-pack resolution and policy wording. |
| Stage profile requirement (NSR/LFM coverage) | enforced only in prod check | yes | SKIP by design in test | blocker if unmet | prod criteria must be explicit in docs | carry into Prompt 78 overclaim guard and Prompt 79 refresh gate conditions. |

## 7. Prod Blocker Summary

| Blocker | Source | Priority | Suggested next action |
|---|---|---|---|
| Missing fresh workspace evidence report due timeout | Prompt 77 probe run | P0 | Stabilize/complete `workspace-test-check`, then rerun split gate. |
| Prod backend feature requirement (`backend-burn`) missing | Prompt 77 prod diagnostic | P0 | Plan and document feature-pack/backend path (Prompt 77A). |
| Required records not proven in fresh prod evidence | stale sample + no fresh prod report | P0 | Define strict prod required-record acceptance and gather fresh evidence. |
| Test/prod SKIP semantics not yet codified as blocker matrix in canonical docs | docs/code inventory | P1 | Prompt 78 docs overclaim guard + matrix hardening. |

## 8. Prompt 78/79 Implications
- Prompt 78 docs overclaim guard should keep test-vs-prod and SKIP-vs-PASS distinctions explicit.
- Prompt 79 readiness refresh should run only after P0/P1 blockers are either addressed or explicitly documented as unresolved blockers.

## 9. Open Questions
- Should `CapabilityIssuanceRecord` remain deferred in prod?
- Should prod require `backend-burn`/`lfm-candle` or another feature-pack baseline?
- Should prod require zero SKIPs, or a narrow allowlist of optional SKIPs?
- Which optional probes remain optional in prod?
- Can prod readiness exist without runtime scheduler?
- How should prod-readiness wording avoid production-runtime overclaims?

## 10. Recommended Next Prompt
UCF Prompt 78 — Prod-Profile Docs Overclaim Guard

Alternative when blocker planning is required first:
UCF Prompt 77A — Prod-Profile Blocker Fix Planning

## 11. Prompt 77A Follow-up
- Prompt 77A investigated and minimally fixed model-lifecycle snapshot drift by restoring an explicit second-slot declaration in `docs/series_state_snapshot.md`.
- No prod-readiness claim was added or implied.
- Prompt 78 remains the next step only after workspace + clippy validation is green.
