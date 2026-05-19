# UCF Post-Archive Roadmap Selection

## 0. Purpose
- Select the next major roadmap line after bounded Evidence/Archive append/readback closure.
- Analysis, prioritization, and boundary planning only.
- No implementation of runtime scheduler/queue/worker, Gateway read/write API, identity authority, or ISM writes/upserts.
- No readiness overclaim: bounded append/readback implementation and closure documentation are available, with a workspace-evidence caveat.

## 1. Baseline
- Branch: `work`
- HEAD: `eed91e4bb903d3b293c461a46831d486ca81e394`
- Dirty state: clean
- Workspace package count: 192
- Links:
  - [docs/roadmap/evidence_archive_append_readback_closure.md](./evidence_archive_append_readback_closure.md)
  - [docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md](./evidence_archive_append_contracts_roadmap_boundary_audit.md)
  - [docs/roadmap/geist_ism_closure.md](./geist_ism_closure.md)
  - [docs/minimal_spine_v1_freeze.md](../minimal_spine_v1_freeze.md)

## 2. Candidate Inventory

| Candidate | Relevant paths | Current maturity | Tests present | Boundary risk | Dependency on completed lines | Can remain bounded? | Difficulty | Notes |
|---|---|---|---|---|---|---:|---|---|
| A. Prod-profile / Workspace Evidence Stability | `runtime/ucf-ops`, `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml`, `docs/roadmap/evidence_archive_append_readback_closure.md` | partial | yes | low | high | yes | M | Directly addresses workspace-test/check and split-evidence readiness caveat. |
| B. Runtime Replay/Sleep Scheduler / Queue | `runtime/ucf-replay`, `core/crates/ucf-sleep-coordinator`, related roadmap docs | docs-only | bounded tests only (no runtime) | critical | medium | limited | XL | High coupling risk (runtime worker/action semantics), should stay deferred. |
| C. Gateway Read API Expansion | Gateway/docs surfaces and append/readback docs | docs-only | no direct bounded tests for Gateway surface | high | medium | yes (if strictly read-only) | L | Useful visibility lane, but authority-confusion risk if introduced before stronger evidence hygiene. |
| D. Identity Anchor Authority Roadmap | `domains/geist/crates/ucf-geist`, policy/governance docs | docs-only | no bounded identity-finalization tests | critical | low | limited | XL | Highest semantic/authority risk; roadmap only and deferred until stronger governance/evidence baseline. |
| E. Protocol Schema / Provenance Evolution | `protocol/crates/ucf-protocol/*`, append/readback docs | partial | yes (protocol tests + bounded append tests) | high | high | yes (docs/schema bounded) | L | Important cross-line semantics cleanup; broad blast radius suggests staging after evidence stability hardening. |
| F. Runtime Geist/ISM / ISM Write/Upsert Roadmap | `domains/geist/crates/ucf-geist` (`IsmStore::upsert_anchor`) | skeleton | no bounded write-path tests for this line | critical | low | no (without broad semantic changes) | XL | Explicitly deferred; conflicts with candidate-only bounded line and identity authority deferral. |

## 3. Selection Criteria Score

| Candidate | Freeze safety | Builds on completed lines | Reduces overclaim risk | Unlocks future work | CI-friendliness | Authority clarity | Addresses caveat | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Prod-profile / Workspace Evidence Stability | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 34 |
| B. Runtime Replay/Sleep Scheduler / Queue | 2 | 3 | 1 | 4 | 2 | 1 | 0 | 13 |
| C. Gateway Read API Expansion | 3 | 3 | 2 | 4 | 3 | 2 | 1 | 18 |
| D. Identity Anchor Authority Roadmap | 2 | 2 | 1 | 5 | 2 | 1 | 0 | 13 |
| E. Protocol Schema / Provenance Evolution | 4 | 4 | 3 | 5 | 4 | 3 | 2 | 25 |
| F. Runtime Geist/ISM / ISM Write/Upsert Roadmap | 1 | 2 | 0 | 4 | 1 | 0 | 0 | 8 |

## 4. Roadmap Decision

| Decision | Selected line | Reason | Risks | Guardrails |
|---|---|---|---|---|
| Primary | A. Prod-profile / Workspace Evidence Stability | Prompt 70 closure remains caveated due missing fresh workspace-test evidence and split-evidence readiness-gate PASS in-session; this line reduces overclaim risk first. | Can drift into gate loosening if not constrained. | Strict freshness + no gate weakening + no semantic widening. |
| Secondary | E. Protocol Schema / Provenance Evolution | Natural follow-up once evidence stability is hardened; clarifies cross-line payload/provenance semantics. | Schema breadth can create accidental behavior overreach. | Docs/schema-first, deterministic compatibility checks only. |
| Deferred | C. Gateway Read API Expansion | Valuable but should follow evidence stability to avoid visibility-as-authority confusion. | Could be misread as write/authority surface. | Explicit read-only and no action authority. |
| Deferred | B. Runtime Replay/Sleep Scheduler / Queue | Runtime coupling should not start before stable evidence/gate hygiene. | Worker/runtime activation and nondeterministic behavior risk. | No activation before deterministic bounded test envelope. |
| Deferred | D. Identity Anchor Authority Roadmap | Highest semantic/authority risk; should come after governance and evidence hardening. | Identity overclaim and authority confusion. | Roadmap-only, no identity finalization semantics. |
| Deferred | F. Runtime Geist/ISM / ISM Write/Upsert Roadmap | Depends on identity/policy authority decisions and stronger readiness basis. | Violates bounded candidate-only line if premature. | No write/upsert activation until explicitly authorized later. |

## 5. Guardrails for Selected Line
- No weakening of gates.
- No treating timeout or missing report as PASS.
- Workspace evidence freshness is mandatory.
- Stale reports must fail freshness checks.
- Split-evidence readiness-gate must consume a fresh `workspace_test_report.json`.
- Root `out/*.json` reports are not committed as self-referential truth unless explicitly intended by policy/process.
- Preserve progress logging and timeout diagnostics.
- Do not change UCF semantics or Minimal Spine v1.x behavior.
- Do not mark prod readiness without explicit prod-profile evidence.
- Keep docs/report metadata current and date/commit aligned.

## 6. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 72 | Workspace Evidence Stability Roadmap and Boundary Audit | Establish complete scope and non-goals for evidence stability hardening. | Docs + inventory + boundary audit only. | Clear caveat capture, no overclaims, explicit guardrails. | No gate changes, no runtime behavior changes. |
| 73 | Workspace-Test-Check Phase Decomposition and Timing Report | Decompose workspace-test-check into deterministic phases and timing diagnostics. | Instrumentation/docs/tests around phase reporting only. | Repeatable phase timing output and timeout attribution. | No pass-on-timeout; no criteria relaxation. |
| 74 | Workspace Evidence Freshness Enforcement Tests | Encode freshness requirements for workspace evidence report consumption. | Tests + validation logic for freshness checks. | Stale/missing reports deterministically fail. | No bypass paths; no semantic widening. |
| 75 | Readiness-Gate Split Evidence CI Alignment | Align CI/nightly split-evidence invocation with freshness rules. | Workflow + ops docs alignment. | CI uses fresh report path and deterministic failure modes. | No gate weakening; no synthetic PASS. |
| 76 | Prod-Profile Readiness Inventory and Gap Report | Inventory prod-profile prerequisites and evidence gaps. | Docs + checklist inventory. | Complete gap table with bounded/non-goals explicit. | No prod readiness claim. |
| 77 | Prod-Profile Required Records / Skips Audit | Define required vs skip-allowed records for prod-profile evidence. | Policy/docs/tests around report completeness. | Deterministic required/optional classification with failure semantics. | No runtime feature rollout. |
| 78 | Prod-Profile Docs Overclaim Guard | Add explicit wording guardrails against prod overclaim. | Docs guard updates only. | Wording blocks false readiness claims. | No implementation semantics changes. |
| 79 | Workspace/Prod Readiness Refresh | Re-run full evidence path with freshness policy and produce bounded status refresh. | Validation and status docs refresh. | Fresh reports, explicit caveat handling, deterministic result statement. | Timeout != pass, no overclaim. |
| 80 | Post-Prod-Roadmap Selection: Gateway vs Runtime Scheduler vs Identity Anchor | Re-select next major line after evidence/prod stability work. | Planning document only. | Ranked options with updated risks/dependencies. | No implementation. |

## 7. Deferred Lines
- **Runtime scheduler/queue (B)** waits for stronger workspace/prod evidence stability to prevent coupling runtime rollout to uncertain gate evidence.
- **Gateway read API (C)** waits until evidence freshness and authority-language are hardened so visibility does not get misread as authority.
- **Identity anchor roadmap (D)** waits for stronger governance/policy boundaries and proven evidence hygiene.
- **Runtime Geist/ISM writes/upserts (F)** waits for identity and policy authority sequencing; currently incompatible with bounded candidate-only ISM line.
- **Protocol schema/provenance (E)** is promoted as secondary but intentionally not parallelized as primary to keep caveat-resolution first.

## 8. Revalidation Rules
- Before each prompt in this line, run at minimum:
  - `cargo fmt --check`
  - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
  - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
  - `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`
  - If workspace report exists: `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --workspace-test-report ./out/workspace_test_report.json --out ./out/gate_report.json`
- Freshness policy:
  - Reports must be generated on current HEAD and current run window.
  - Missing or stale workspace report blocks split-evidence readiness-gate PASS claims.
- Caveat policy:
  - Bounded append/readback closure may be referenced.
  - Full readiness closure must remain caveated until fresh workspace evidence + split gate PASS are present.
- Update trigger:
  - Re-run this selection if evidence stability line changes, if CI behavior materially changes, or if deferred lines gain new bounded prerequisites.

## 9. Next Prompt
- **Recommended next prompt:** UCF Prompt 72 — Workspace Evidence Stability Roadmap and Boundary Audit.
- **Reason:** it directly addresses the current workspace-evidence caveat without widening runtime, Gateway, or identity authority semantics.


## 10. Workspace Evidence Audit Availability
- Workspace evidence roadmap/boundary audit is available at `docs/roadmap/workspace_evidence_stability_roadmap_boundary_audit.md`.
- Recommended next prompt remains **UCF Prompt 73 — Workspace-Test-Check Phase Decomposition and Timing Report**.
