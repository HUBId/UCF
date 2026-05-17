# UCF Post-Replay Roadmap Selection

## Historical status note

This document is the Prompt 45 selection record. It remains useful for why Sleep was selected after bounded Replay closure, but current Sleep claims are now governed by [`docs/roadmap/sleep_integration_roadmap_boundary_audit.md`](sleep_integration_roadmap_boundary_audit.md) and [`docs/roadmap/sleep_record_authority_schema_alignment.md`](sleep_record_authority_schema_alignment.md). As of Prompt 52, allowed Sleep claims are limited to `SleepPlanCandidate` candidate-only, `SleepPlanAudit` verify-only, `SleepAppliedBoundary` local-only, and bounded Sleep E2E determinism; no Sleep runtime, Sleep Cycle Coordinator activation, Geist/ISM integration, identity finalization/anchor, memory stabilization, Evidence/Archive sleep append, Gateway-visible Sleep, or production Sleep readiness is implied here.

## 0. Purpose

Post-Sleep update: bounded Sleep closure is now complete, and [`docs/roadmap/post_sleep_roadmap_selection.md`](post_sleep_roadmap_selection.md) supersedes this earlier next-line planning for decisions after Sleep closure.

Prompt 53 closure is now available in [`docs/roadmap/sleep_closure.md`](sleep_closure.md). It closes only the bounded Sleep candidate/audit/local-boundary/E2E line with fresh validation evidence and does not claim Sleep runtime readiness, Geist/ISM readiness, identity finalization, memory stabilization, Evidence/Archive append, Gateway-visible Sleep, or production Sleep readiness.


- Select the next line after bounded Replay closure.
- No implementation is introduced by this document.
- No Sleep, Geist, ISM, runtime Replay Scheduler, production Replay, Gateway write, capability, Evidence/Archive replay append, or identity readiness claim is made here.
- Minimal Spine v1.x remains frozen and unchanged.
- This is a roadmap-selection and boundary-planning document only.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `dfe5eb5909d919a7a584adea3c747e08d3fdb362` |
| HEAD short | `dfe5eb59` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Replay closure present | yes |
| Consolidation closure present | yes |
| Compute closure present | yes |
| Freeze doc present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |

Baseline commands used for this selection: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -30`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and presence checks for the Replay closure, consolidation closure, real-compute closure, Minimal Spine freeze, `runtime/ucf-replay`, and `domains/geist/crates/ucf-geist`.

Required context links:

- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/roadmap/replay_scheduler_roadmap_boundary_audit.md`](replay_scheduler_roadmap_boundary_audit.md)
- [`docs/roadmap/replay_record_authority_schema_alignment.md`](replay_record_authority_schema_alignment.md)
- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

## 2. Candidate Inventory

| Candidate | Relevant paths | Current maturity | Tests present | Boundary risk | Dependency on completed lines | Can remain bounded? | Difficulty | Notes |
|---|---|---|---|---|---|---:|---|---|
| A. Sleep Integration Roadmap and Boundary Audit | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `domains/geist/crates/ucf-geist/src/lib.rs`; `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs`; `docs/roadmap/replay_closure.md`; `docs/current_state_architecture_index.md`; `docs/module_implementation_depth_registry.md` | mixed | Sleep coordinator has in-source unit tests; Replay has targeted builder/audit/applied-boundary/E2E tests; Geist has in-source tests; no replay-to-sleep E2E integration test is present. | medium | Builds naturally on bounded Replay `ReplayAudit` and `ReplayAppliedBoundary`, plus bounded consolidation artifacts, but must not consume them as identity or Geist authority. | yes | L | Best strategic next line if kept docs-first: it clarifies the replay-to-sleep boundary before any runtime scheduler or Geist/ISM work. Main risk is mistaking existing sleep coordinator code for approved integration readiness. |
| B. Geist/ISM Roadmap and Boundary Audit | `domains/geist/crates/ucf-geist/src/lib.rs`; `core/crates/ucf-ism`; `core/crates/ucf-recursion-controller`; `core/crates/ucf-sleep-coordinator/src/lib.rs`; current-state and registry docs | partial | `ucf-geist` has in-source tests; no separate `domains/geist/crates/ucf-geist/tests/` directory was found. Workspace coverage compiles the crate. | critical | Depends on Replay/Sleep/Consolidation semantics being explicit first; otherwise Replay or Sleep effects can be over-promoted into self-state authority. | yes, if projection-only | XL | Defer. The semantic risk is high because identity anchor, self-model authority, and recursion claims can arise from ambiguous wording even before code changes. |
| C. Replay Evidence/Archive Append Contract | `runtime/ucf-replay/src/lib.rs`; `core/crates/ucf-evidence/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs`; `domains/archive/crates/ucf-archive-store/src/lib.rs`; `runtime/ucf-replay/tests/*`; `docs/roadmap/replay_record_authority_schema_alignment.md`; `docs/minimal_spine_v1_freeze.md` | partial | Replay E2E tests are present; Minimal Spine router and Gateway tests already cover canonical evidence/archive append/readback for the frozen spine; no explicit replay append/readback contract is implemented. | medium | Builds directly on ReplayAudit and ReplayAppliedBoundary, but must preserve Evidence/Archive as the canonical append authority and avoid a second event log. | yes | M | Strong secondary line. It is technically close to Replay closure, but less architecturally unlocking than Sleep boundary planning. |
| D. Runtime Replay Scheduler / Queue | `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/*`; `crates/replay_executor/src/*`; `crates/replay_evidence/src/*`; `.github/workflows/ci.yml`; `docs/roadmap/replay_scheduler_roadmap_boundary_audit.md` | functional-prototype | Replay has deterministic token, schedule, audit, applied-boundary, and E2E tests; runtime queue/worker tests are not part of the bounded closure. | high | Depends on bounded Replay; should wait for Sleep boundary or replay Evidence/Archive append authority to prevent hidden Sleep/Geist activation. | yes | L | Defer behind Sleep or append-contract planning. Scheduler language can imply execution authority, background worker semantics, or runtime readiness if not isolated. |
| E. Prod-profile / Workspace Evidence Stability | `runtime/ucf-ops/src/readiness_spine.rs`; `runtime/ucf-ops/src/main.rs`; `runtime/ucf-ops/tests/*`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml`; `docs/readiness_gate.md`; `docs/artifact_convention_v0.md`; `docs/roadmap/readiness_gate_timeout_stability_audit.md`; `docs/current_state_architecture_index.md` | mixed | Docs lint, readiness-spine, workspace-test-check, readiness-gate, workspace tests, clippy, and nightly CI surfaces exist; workspace-test evidence stability remains environment-sensitive. | low | Operationally supports all completed lines and future lines, but does not itself advance UCF functional architecture. | yes | M | Recommended as a parallel validation line. It must not weaken gates; it should make freshness, stale-report failure, and profile expectations more explicit. |
| F. Protocol Schema / Provenance Evolution | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/messages_v1.md`; `protocol/crates/ucf-protocol/spec/v1.md`; `protocol/crates/ucf-protocol/tests/*`; consolidation/replay docs and tests | code-near-spec | Protocol canonical tests are present; consolidation and Replay use wrapper provenance/digests; no new schema evolution is authorized here. | high | Touches Micro/Meso/Macro/Replay provenance and could invalidate Minimal Spine v1.x if not versioned and gated. | yes, if docs-first | XL | Defer unless schema debt blocks Sleep boundary planning. Any schema change requires explicit versioning, fixtures/goldens, migration notes, and full gates. |

## 3. Selection Criteria Score

Scores are 0-5 where 5 is strongest for the criterion. The `Strategic value` column summarizes future-architecture unlock value without asserting readiness.

| Candidate | Freeze safety | Builds on replay/consolidation | Reduces overclaim risk | Unlocks future work | CI-friendliness | Authority clarity | Strategic value | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Sleep Integration Roadmap and Boundary Audit | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 33 |
| B. Geist/ISM Roadmap and Boundary Audit | 3 | 3 | 4 | 5 | 3 | 2 | 4 | 24 |
| C. Replay Evidence/Archive Append Contract | 4 | 5 | 4 | 4 | 4 | 3 | 4 | 28 |
| D. Runtime Replay Scheduler / Queue | 3 | 5 | 3 | 4 | 3 | 3 | 4 | 25 |
| E. Prod-profile / Workspace Evidence Stability | 5 | 3 | 4 | 3 | 4 | 5 | 3 | 27 |
| F. Protocol Schema / Provenance Evolution | 2 | 4 | 4 | 4 | 3 | 3 | 4 | 24 |

## 4. Roadmap Decision

| Decision | Selected line | Reason | Risks | Guardrails |
|---|---|---|---|---|
| Primary next line | A. Sleep Integration Roadmap and Boundary Audit | It is the most natural architectural step after bounded Replay closure because it can define how ReplayAudit and ReplayAppliedBoundary may be consumed by Sleep planning without implementing runtime scheduler, Geist/ISM, or identity behavior. | Existing sleep coordinator code can be overread as approved integration; replay-to-sleep wording can imply hidden runtime activation. | Start docs-only; no coordinator implementation in the first prompt; no Geist/ISM writes; no identity finalization or anchor; no Evidence/Archive authority change. |
| Secondary next line | C. Replay Evidence/Archive Append Contract | It is the closest technical continuation after Replay closure and can make explicit append/readback boundaries if Sleep planning requires replay evidence durability. | Could create a second event-log authority or confuse replay audit records with canonical output/event authority. | Preserve `ucf-evidence`, `ucf-archive`, and `ucf-archive-store` authority; use explicit append/readback only; no Gateway write or scheduler behavior. |
| Parallel validation line | E. Prod-profile / Workspace Evidence Stability | It hardens operational truthfulness and report freshness across environments while feature-roadmap work remains bounded. | Gate hardening can accidentally become gate weakening if stale reports are allowed or profile expectations blur. | Do not weaken gates; stale workspace evidence fails; profile expectations must be explicit; generated `out/*.json` remain uncommitted unless release workflow requires them. |
| Deferred line | B. Geist/ISM Roadmap and Boundary Audit | Geist/ISM should wait until Replay/Sleep boundary semantics are explicit enough to avoid identity/self-state overclaiming. | Identity anchor, self-model authority, recursive stabilization, and policy mutation claims. | Projection-only first when later selected; no identity anchor/finalization; no hidden macro/replay promotion; no Gateway/action authority. |
| Deferred line | D. Runtime Replay Scheduler / Queue | Runtime scheduling should wait until either Sleep boundary or replay append contract prevents hidden activation ambiguity. | Queue/worker semantics can imply runtime Replay readiness or trigger Sleep/Geist implicitly. | Planned schedule remains distinct from runtime execution; no background worker, Sleep trigger, Geist write, Gateway write, or production readiness claim. |
| Deferred line | F. Protocol Schema / Provenance Evolution | It is important but only urgent if Sleep planning uncovers blocking schema debt. | Protocol version drift, fixture/golden churn, Minimal Spine freeze invalidation. | Docs-first audit before schema changes; version bump and migration notes if later authorized; run full protocol/spec/docs gates. |

## 5. Guardrails for Selected Line

The selected primary line is Sleep Integration Roadmap and Boundary Audit. The first prompt in that line must obey these guardrails:

- Sleep integration starts as roadmap/boundary audit only.
- No Sleep Cycle Coordinator implementation in the first prompt.
- No Geist/ISM write.
- No identity finalization.
- No identity anchor.
- No `ReplayAppliedBoundary` to Geist promotion.
- Sleep completion does not mean identity stabilization.
- `ReplayAudit` and `ReplayAppliedBoundary` may be inputs only after an explicit boundary is documented and tested.
- Evidence/Archive authority remains unchanged.
- No Gateway writes.
- No runtime scheduler hidden activation.
- Deterministic tests are required before any integration is promoted beyond planning.
- Docs must distinguish replay schedule, sleep plan, and Geist/ISM identity effects.
- Existing `ucf-sleep-coordinator` code is inventory evidence only; this selection does not approve new coordinator behavior.
- Existing `ucf-geist` sleep/replay references are inventory evidence only; this selection does not approve Geist/ISM integration.

## 6. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 46 | Sleep Integration Roadmap and Boundary Audit | Create a docs-only sleep boundary audit after Replay closure. | Inventory `ucf-sleep-coordinator`, Replay audit/applied records, Geist references, CI visibility, and docs claims. | New roadmap/boundary document; no code changes; docs lint and formatting pass; current boundaries explicitly listed. | No coordinator implementation; no Geist/ISM write; no identity/finalization/anchor; no Evidence/Archive authority change. |
| 47 | Sleep Record Authority and Schema Alignment | Decide whether sleep planning needs local records, wrappers, or protocol/schema changes before implementation. | Docs-first authority map for sleep plan candidate, verify-only audit, and applied boundary surfaces. | Authority table names canonical owners and deferred schema decisions; no schema change unless separately authorized. | Preserve Minimal Spine v1.x; no duplicate event log; no protocol mutation without explicit versioned prompt. |
| 48 | Deterministic Sleep Plan Candidate from Replay Boundary | Plan a bounded deterministic candidate builder from ReplayAudit/ReplayAppliedBoundary inputs. | Specify candidate inputs/outputs and deterministic ordering before code. | Candidate spec distinguishes replay schedule from sleep plan; negative boundaries documented. | No runtime scheduler; no Sleep Cycle Coordinator activation; no Geist/ISM or identity effect. |
| 49 | Sleep Plan Verify-Only Audit Contract | Plan a verify-only sleep audit analogous to ReplayAudit. | Define what can be checked without applying or appending sleep effects. | Audit is verify-only and local; docs identify required deterministic tests. | No append/readback unless a later prompt selects it; no Gateway writes; no policy mutation. |
| 50 | Sleep Applied Boundary Without Geist/ISM | Plan a local-only sleep applied boundary marker. | Define marker semantics and forbidden promotions. | Applied boundary remains local bookkeeping; docs state it is not identity stabilization. | No `SleepApplied` to Geist promotion; no identity finalization; no macro/replay hidden promotion. |
| 51 | Sleep E2E Determinism | Plan a bounded deterministic sleep E2E test path. | Test design only unless a later prompt authorizes implementation. | E2E design covers fresh-run determinism, ordering, and negative boundaries. | No runtime queue/worker; no Evidence/Archive append; no production readiness claim. |
| 52 | Sleep Docs Overclaim Guard | Align current docs with sleep boundary vocabulary. | Docs-only cleanup if claims drift is found. | Docs distinguish sleep planning from runtime Sleep readiness and from Geist/ISM identity effects. | No feature claims; no Minimal Spine v1.x change; no gate criteria weakening. |
| 53 | Sleep Readiness Refresh | Re-run and document readiness checks for the bounded sleep planning line. | Docs/gate validation and report-freshness review. | Fresh docs lint, readiness-spine, workspace evidence policy, targeted tests, workspace tests, fmt, clippy. | Generated `out/*.json` usually uncommitted; stale reports cannot support readiness claims. |
| 54 | Post-Sleep Roadmap Selection: Geist/ISM vs Runtime Scheduler vs Prod-Profile | Select the next line after bounded Sleep boundary planning. | Compare Geist/ISM, Runtime Replay Scheduler, replay append contract, prod-profile, and schema evolution after Sleep boundary is explicit. | New selection document or update; clear primary/secondary/parallel/deferred decisions. | Geist/ISM remains deferred unless projection-only authority is explicit; no hidden runtime activation. |
| 55 | Optional Sleep Append/Readback Contract Decision | If needed, decide whether sleep records require explicit Evidence/Archive append/readback. | Planning only unless separately authorized. | Append/readback decision preserves canonical Evidence/Archive authority and names deferred implementation prompts. | No second event log; no Gateway writes; no production storage claim. |

## 7. Deferred Lines

- Geist/ISM is deferred because self-state and identity language has critical authority risk. It should follow an explicit Replay/Sleep boundary, and its first future step should be projection-only with no identity anchor or finalization.
- Replay Evidence/Archive Append Contract is secondary. It can follow Sleep boundary planning or be pulled forward if Sleep requires durable replay audit/readback inputs. It must not create a second event log.
- Runtime Replay Scheduler / Queue is deferred until hidden activation risk is lower. Planned replay ordering is already bounded; runtime queue/worker behavior would be a separate authority surface.
- Prod-profile / Workspace Evidence Stability is parallel because it improves operational reliability without changing feature behavior. It should not block docs-only Sleep planning unless current gates become stale or failing.
- Protocol Schema / Provenance Evolution is deferred unless Sleep planning exposes blocking schema debt. Any schema evolution must be versioned, deterministic, fixture/golden-aware, and fully gated.

## 8. Revalidation Rules

- Before each prompt in the Sleep line, capture `pwd`, branch, `git status --short`, full and short HEAD, recent log, workspace package count, and required-file presence relevant to the prompt.
- If `git status --short` is non-empty at baseline, stop and report the dirty paths before making changes.
- For docs-only prompts, run at minimum:
  - `cargo fmt --check`
  - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
  - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
  - targeted package tests affected by the prompt when practical
  - `git diff --check`
- Before closing a multi-prompt Sleep planning block, attempt:
  - `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`
  - `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --workspace-test-report ./out/workspace_test_report.json --out ./out/gate_report.json` only when the workspace-test report exists and is fresh
  - `cargo test -p ucf-replay --test minimal_spine_replay_e2e -- --nocapture`
  - `cargo test -p ucf-geist --all-targets`
  - `cargo test -p ucf-consolidation --test minimal_spine_consolidation_pipeline_e2e -- --nocapture`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Root `out/*.json` reports are current evidence only for the HEAD/run whose embedded metadata matches the evaluated HEAD. They should normally remain uncommitted.
- Workspace evidence stability remains an operational caveat across environments. A timeout or missing workspace-test report must not be treated as a pass.
- Update this roadmap selection if Sleep code is implemented, if Replay append/readback becomes a prerequisite, if Geist/ISM authority is explicitly selected, if Minimal Spine v1.x boundaries change, if CI/gate criteria change, or if protocol schema debt blocks the selected line.

## 9. Next Prompt

Prompt 46 is now recorded in [`docs/roadmap/sleep_integration_roadmap_boundary_audit.md`](sleep_integration_roadmap_boundary_audit.md). It remains a docs-only boundary audit and does not authorize Sleep implementation, Geist/ISM activation, identity finalization, runtime scheduling, or Evidence/Archive authority changes.

Recommended next prompt title: **UCF Prompt 47 — Sleep Record Authority and Schema Alignment**.

Reason: bounded Replay closure is complete for local deterministic Token->Schedule->Audit->AppliedBoundary validation, but no runtime scheduler, Sleep, Geist/ISM, identity, Evidence/Archive replay append, Gateway write, or production readiness should be claimed. A docs-only Sleep boundary audit is the safest next step because it clarifies how Replay outputs may later inform Sleep planning while keeping Geist/ISM and runtime activation deferred.
