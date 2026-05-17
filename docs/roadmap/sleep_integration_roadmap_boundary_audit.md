# UCF Sleep Integration Roadmap and Boundary Audit

## 0. Purpose

- This is a roadmap and boundary audit only.
- No Sleep implementation is introduced here.
- This is not Geist/ISM readiness.
- This is not identity finalization.
- It does not implement a Sleep Cycle Coordinator, Replay runtime scheduler, queue, worker, Gateway write API, capability issuance, real-compute activation, Evidence/Archive authority change, second event-log authority, or Evidence/Archive append.
- Minimal Spine v1.x remains frozen and unchanged.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `c250599649c9f255b84aeef5a9e4ddccb4792be4` |
| HEAD short | `c2505996` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Post-Replay selection present | yes |
| Replay closure present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |
| Replay E2E present | yes |

Baseline commands used for this audit: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -30`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and presence checks for the Post-Replay selection, Replay closure, `runtime/ucf-replay`, `domains/geist/crates/ucf-geist`, and `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs`.

Required context links:

- [`docs/roadmap/post_replay_roadmap_selection.md`](post_replay_roadmap_selection.md)
- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)

## 2. Sleep Code / Docs Inventory

| Concern | Existing API/type/doc | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|---|
| Sleep coordinator crate/module | `ucf-sleep-coordinator` with `WalSleepCoordinator`, `SleepHeuristics`, `SleepState`, `SleepTrigger`, `SleepReplaySummary`, `SleepStateUpdater`, `SleepPhaseRunner`, `SleepTriggered` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Maintains local WAL-style sleep trigger state, recent metrics, evidence window, replay summary field, structural stats/proposal fields, deterministic trigger evaluation, and an adapter to `ucf-rsa::SleepCoordinator`; in-source unit tests cover deterministic triggers and policy-gated report append behavior. | functional-prototype | Existing runnable sleep-trigger/report code can be overread as approved post-Replay Sleep integration. It is inventory evidence only for this roadmap. |
| Sleep report runner | `SleepReportReady`, `SleepCoordinator::run_sleep_phase` | `core/crates/ucf-rsa/src/lib.rs` | If `SleepPhaseGate::allow_sleep()` passes, builds a sleep context/report, appends an experience record through the configured archive appender, optionally commits structural data, publishes `SleepReportReady`, and returns the event. | functional-prototype | It performs an archive append in its existing local crate context; this audit does not authorize new Sleep Evidence/Archive append contracts or a changed authority model. |
| Temporal sleep/replay gating | `TemporalCoordinator`, `TcfCore`, `TcfPlan.sleep_active`, `TcfPlan.replay_active` | `core/crates/ucf-tcf/src/lib.rs` | Deterministic temporal coordinator calculates bounded sleep/replay active flags and gain caps from fixed-point state and inputs. | partial | Names can imply runtime orchestration. For Sleep v1 planning they are context only, not a runtime scheduler/worker authority. |
| SleepPlan/SleepCycle/SleepApplied records | No current `SleepPlan`, `SleepCycle`, `SleepApplied`, `SleepAppliedBoundary`, or sleep-specific replay-boundary record found in the audited paths. | Repository-wide search over docs, core, runtime, domains, protocol, and workflows | No schema-aligned SleepPlan candidate, verify-only SleepPlan audit, or local SleepApplied boundary exists yet. | unknown | Prompt 47 must decide authority and schema alignment before any implementation. |
| Sleep tests | In-source tests for `ucf-sleep-coordinator` and `ucf-rsa`; workspace tests also compile these crates when selected. | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `core/crates/ucf-rsa/src/lib.rs` | Tests cover trigger thresholds, deterministic trigger behavior, default policy denial, allowed sleep report append, and `SleepReportReady` production in local contexts. | partial | No replay-to-sleep E2E determinism test exists; current tests do not prove Sleep Integration v1. |
| Replay boundary types | `MinimalSpineReplayScheduleAudit`, `MinimalSpineReplayAppliedBoundary`, `MinimalSpineReplayScheduleBuildOutput`, `MinimalSpineReplayTokenBuildOutput` | `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/*` | Replay produces deterministic tokens, planned schedules, verify-only audits, and a local applied-boundary marker from a PASS audit. Audit fails if applied, sleep-cycle, Geist-ingested, identity-anchor, or Evidence/Archive appended flags are set. | functional-prototype | These are safe inputs for future Sleep planning only as immutable metadata; Sleep must not execute replay or mutate replay schedules/tokens. |
| Replay E2E determinism | `minimal_spine_replay_e2e` | `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs` | Exercises deterministic token-to-schedule-to-audit-to-applied-boundary replay path without runtime replay apply, Sleep, Geist/ISM, identity, Gateway, or Evidence/Archive append. | functional-prototype | It proves bounded replay determinism only, not Sleep integration. |
| Replay-to-Sleep references | `SleepReplaySummary` in sleep coordinator; `sleep_cycle` flag in replay token/schedule/audit boundaries; Post-Replay and Replay closure docs name Sleep as deferred/next planning line. | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `runtime/ucf-replay/src/lib.rs`; `docs/roadmap/post_replay_roadmap_selection.md`; `docs/roadmap/replay_closure.md` | Existing references are summary/flag/planning language, not an implemented replay-to-sleep integration. | docs-only / skeleton | Main risk is promoting ReplayAudit/ReplayAppliedBoundary into hidden Sleep activation. |
| Geist/ISM references | `GeistKernel`, `SelfState`, `InMemoryIsm`, `ReplayStabilization`, `apply_replay_effects`, `SleepStateHandle` | `domains/geist/crates/ucf-geist/src/lib.rs` | Geist can ingest macro milestones, compute self-state anchors, gate ISM upserts, update optional sleep state with consistency/evidence, and compute replay stabilization from legacy `ReplayApplied` effects. | partial | Critical overclaim risk: Sleep planning must not trigger Geist ingestion, ISM writes, self-state authority, identity anchor, or replay-stabilization claims. |
| Evidence/Archive references | `ucf_archive::ExperienceAppender`, `ucf_archive_store::{ArchiveAppender, ArchiveStore, RecordKind}`, consolidation append/readback helpers, replay append explicitly deferred. | `core/crates/ucf-rsa/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `docs/roadmap/replay_record_authority_schema_alignment.md` | Existing components append local experience/archive records in their own contracts; bounded Replay currently does not append replay records. | partial | Sleep must not become a second Evidence/Archive authority; any Sleep append contract needs a later explicit prompt. |
| Runtime scheduler references | `ucf-ops workspace-test-check`, readiness gate, CI/nightly workflows, replay scheduler roadmap docs, temporal coordinator names | `runtime/ucf-ops/src/lib.rs`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml`; `docs/roadmap/replay_scheduler_roadmap_boundary_audit.md`; `core/crates/ucf-tcf/src/lib.rs` | Operational tools run checks and gates; scheduler/queue/worker runtime replay remains deferred. | docs-only / operational | Sleep roadmap must not hide background worker, scheduler, queue, or production runtime activation behind planning language. |
| Safe APIs for Prompt 47 | Existing replay audit/applied-boundary metadata, existing sleep coordinator/RSA/TCF/Geist APIs as inventory, docs and tests as evidence. | Same audited paths | Safe next step is authority/schema alignment and terminology, not implementation. | docs-only | Prompt 47 should avoid behavior changes unless explicitly authorized. |
| Too broad/risky APIs for Prompt 47 | `SleepCoordinator::run_sleep_phase` append path, `GeistKernel::ingest_macro`, `GeistKernel::apply_replay_effects`, ISM upserts, runtime scheduler/worker code, Gateway writes, Evidence/Archive append contracts. | `core/crates/ucf-rsa/src/lib.rs`; `domains/geist/crates/ucf-geist/src/lib.rs`; runtime/Gateway/Evidence/Archive paths | These can imply real Sleep application, Geist/ISM integration, identity stabilization, or canonical append authority. | partial / functional-prototype | Keep out of scope until dedicated prompts authorize them. |

Inventory answers:

- A Sleep coordinator crate exists at `core/crates/ucf-sleep-coordinator`, but it is not a post-Replay Sleep Integration v1 implementation.
- No SleepPlan/SleepCycle/SleepApplied records were found.
- Sleep-related tests exist in `ucf-sleep-coordinator` and `ucf-rsa`; no replay-to-sleep E2E integration test was found.
- Replay-to-Sleep references exist as summary fields, flags, and planning language only.
- Geist/ISM references exist and are intentionally deferred for Sleep v1.
- Evidence/Archive references exist through local appenders and consolidation contracts, but bounded Replay has no replay append and Sleep v1 must not change authority.
- Runtime scheduler references exist as roadmap/operational context only; no Sleep prompt in this line may implement scheduler/queue/worker behavior without explicit authorization.

## 3. Sleep / Replay Boundary

| Boundary | Decision | Reason |
|---|---|---|
| Sleep input source | Future Sleep v1 may consume immutable bounded Replay metadata: a PASS `MinimalSpineReplayScheduleAudit`, `MinimalSpineReplayAppliedBoundary`, and the audited schedule/token digests they reference. | These artifacts are deterministic and local, and they avoid runtime replay execution. |
| `ReplayAppliedBoundary` role | Local input marker only; it can prove a replay subsystem boundary was locally marked after a PASS audit, but it is not a Sleep trigger, replay execution proof, memory write, or Geist/ISM signal. | Replay closure defines it as local-only bookkeeping, not runtime apply or downstream authority. |
| `ReplayAudit` role | Verify-only prerequisite metadata. Sleep may require PASS status and stable audit digest in a future plan builder/audit, but must not change audit semantics. | The audit already rejects applied/sleep/geist/identity/archive flags and is safe as read-only evidence. |
| `ReplaySchedule` role | Planned ordering reference only. Sleep may read ordering/digest/provenance after audit, but must not mutate tokens, schedules, or planned replay order. | Schedule is a deterministic plan, not execution or Sleep plan authority. |
| Evidence/Archive role | Unchanged. Sleep planning may reference existing Evidence/Archive IDs as inputs if already present, but no Sleep append or replay append is authorized by this roadmap. | Prevents second event-log authority and preserves existing append/readback boundaries. |
| Runtime scheduler role | None for Sleep v1 planning. No queue, worker, background loop, production scheduler, or hidden replay runtime apply is allowed. | Keeps Sleep integration deterministic, offline-first, and bounded. |

Required Replay guardrails for later implementation prompts:

- Sleep may consume `ReplayAppliedBoundary` or `ReplayAudit` PASS metadata as input.
- Sleep must not execute replay.
- Sleep must not write Geist/ISM.
- Sleep must not finalize identity.
- Sleep must not claim memory stabilized.
- Sleep must not mutate Replay schedules or tokens.
- Sleep must not become Evidence/Archive authority.
- Any Sleep record append must be explicit in a later authorized prompt.

## 4. Sleep / Geist / ISM Boundary

| Area | Existing API/type | Current behavior | Boundary |
|---|---|---|---|
| Geist/ISM | `GeistKernel`, `IsmStore`, `InMemoryIsm`, `upsert_anchor` | Geist macro ingestion can build self-state anchors and gate ISM upserts. | SleepPlan is not Geist ingestion; SleepApplied is not an ISM write; no Sleep prompt may call Geist/ISM paths implicitly. |
| SelfState | `SelfState`, `SelfStateBuilder`, `GeistLoopState` | Deterministic self-state and loop-state commitments are built inside Geist. | Sleep completion must not be represented as a SelfState or self-model authority. |
| Identity Anchor | `anchor` fields and ISM anchor storage in Geist/index paths | Anchors are digest commitments used by Geist/ISM/index code. | Sleep stabilization is not an identity anchor; Sleep outputs must not be described as identity stabilization. |
| MacroMilestone finalization | Consolidation macro candidate/finalization boundaries and Geist `ingest_macro` | Consolidation has bounded macro candidate/local finalization semantics; Geist ingestion can consume macro milestones. | Sleep must not promote ReplayAppliedBoundary or Sleep completion into MacroMilestone finalization or Geist ingestion. |
| ReplayAppliedBoundary | `MinimalSpineReplayAppliedBoundary` | Local replay-subsystem marker after PASS audit. | It can be a future Sleep input reference only; it must not become a Geist/ISM trigger. |
| Sleep completion | Existing sleep coordinator can produce `SleepReportReady` in local policy-gated contexts; no SleepPlan/SleepApplied completion record exists. | Existing completion means local report readiness only. | Future SleepAppliedBoundary, if authorized, must mean local Sleep boundary bookkeeping only, not memory stabilization, ISM write, identity finalization, or archive authority. |

## 5. Target Scope

Sleep Integration v1 should later mean a deterministic, bounded planning/audit layer built from bounded Replay artifacts, with no hidden downstream activation.

| Layer | Goal | Required inputs | Outputs | Explicit non-goals |
|---|---|---|---|---|
| SleepPlan candidate | Deterministically derive a candidate SleepPlan from bounded Replay artifacts and existing evidence identifiers. | PASS `ReplayAudit` metadata, optional `ReplayAppliedBoundary`, schedule/token digests, provenance digests, and explicit static config. | Local candidate record/wrapper and deterministic digest, if Prompt 47 authorizes the record authority. | No replay execution, no scheduler, no sleep phase run, no Geist/ISM write, no identity claim, no Evidence/Archive append. |
| SleepPlan audit | Verify-only audit of candidate ordering, provenance, Replay boundary references, and forbidden flags. | SleepPlan candidate plus its Replay audit/applied-boundary references and deterministic config. | PASS/FAIL audit with failure reasons and audit digest. | No mutation, no apply, no report append, no Gateway exposure, no gate criteria weakening. |
| SleepAppliedBoundary | Optional local boundary marker that a PASS SleepPlan audit was accepted by the Sleep subsystem boundary. | PASS SleepPlan audit, stable candidate digest, explicit config, and local subsystem identifier. | Local-only SleepAppliedBoundary digest/metadata, if later authorized. | Not Sleep execution, not Geist ingestion, not ISM upsert, not identity finalization, not memory stabilization, not Evidence/Archive append. |

## 6. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| Sleep completion overclaim | high | Existing `SleepReportReady` and `WalSleepCoordinator::maybe_trigger` can produce local sleep reports. | Describe existing completion as local report readiness only; future SleepAppliedBoundary is bookkeeping only unless separately authorized. |
| Geist/ISM hidden integration | critical | `GeistKernel` accepts optional sleep state and has ISM upsert paths. | No Sleep prompt may call Geist ingestion, `apply_replay_effects`, or `IsmStore::upsert_anchor` without a dedicated Geist/ISM prompt. |
| Identity stabilization overclaim | critical | Geist self-state anchors and ISM anchor storage exist. | Sleep stabilization wording is forbidden; Sleep does not finalize identity or create identity anchors. |
| Replay runtime overclaim | high | Replay schedule/audit/applied-boundary path exists and scheduler roadmap docs exist. | Sleep consumes immutable metadata only; no replay execution, queue, worker, or runtime scheduler. |
| Evidence/Archive authority confusion | high | RSA SleepCoordinator can append experience records; consolidation has explicit append/readback contracts. | Sleep roadmap does not authorize new append paths; any append/readback contract needs an explicit later prompt. |
| Nondeterministic sleep ordering | medium | Existing coordinator uses windows and queues; future SleepPlan would order Replay inputs. | Prompt 48 must specify sorted/digest-based deterministic ordering and tests before implementation. |
| Hidden scheduler/background worker | high | Temporal coordinator and readiness tooling use scheduler-like/gating language. | No background loops; CLI/tests only until explicitly authorized. |
| Production readiness overclaim | high | CI/nightly and readiness gates exist and can look production-like. | Passing gates are validation evidence only; Sleep v1 is not production readiness. |
| Historical docs overclaim | medium | Many historical roadmap/anchor/closure docs mention Sleep, replay, Geist, and identity concepts. | Current-state index and this audit outrank historical docs for Sleep boundary claims. |

## 7. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 47 | Sleep Record Authority and Schema Alignment | Decide whether Sleep planning needs local records, wrappers, protocol/schema changes, or docs-only records before implementation. | Authority table for SleepPlan candidate, SleepPlan audit, SleepAppliedBoundary, Evidence/Archive interaction, Replay references, and deferred Geist/ISM links. | New authority/schema doc; no behavior change unless explicitly authorized; clear answer to whether existing records are sufficient. | Preserve Minimal Spine v1.x; no duplicate event log; no Evidence/Archive append; no identity or Geist/ISM activation. |
| 48 | Deterministic SleepPlan Candidate from Replay Boundary | **Complete.** Implemented a deterministic candidate-only builder from Replay boundary metadata in `ucf-sleep-coordinator`. | Required PASS `MinimalSpineReplayScheduleAudit`, optional matching `MinimalSpineReplayAppliedBoundary`, replay schedule/audit/boundary digests, token count, and replay source. | Targeted tests prove stable repeated output, digest changes when replay provenance changes, FAIL/invalid audit rejection, optional boundary matching, provenance preservation, and hard false side-effect flags. | No replay execution, no scheduler, no sleep phase run, no archive append, no Geist/ISM write, no identity anchor, no Gateway visibility. |
| 49 | SleepPlan Verify-Only Audit Contract | **Complete.** Added a local verify-only SleepPlan audit contract in `ucf-sleep-coordinator`. | PASS/FAIL status, deterministic failure reasons, audit digest, candidate digest consistency, Replay audit/schedule/optional boundary digests, token count, source and replay source checks. | Targeted tests prove PASS for a valid candidate, stable audit digest/bytes, deterministic FAIL reasons for tampering, provenance preservation, and hard false Sleep/Geist/ISM/identity/Evidence/Archive/Gateway flags. | Verify-only; no SleepApplied, no Sleep completion, no coordinator runtime trigger/report/WAL, no report append, no scheduler, no Gateway write. |
| 50 | SleepApplied Boundary Without Geist/ISM | Define optional local SleepAppliedBoundary semantics after a PASS SleepPlan audit. | Local boundary marker, digest, provenance, forbidden downstream flags. | Boundary can be built only from PASS audit; failure cases tested; docs state local-only meaning. | Not Sleep execution, not ISM write, not identity finalization, not Evidence/Archive append. |
| 51 | Sleep E2E Determinism | Prove deterministic ReplayBoundary→SleepPlan→SleepAudit→SleepAppliedBoundary chain. | Targeted E2E test with fixed fixtures and repeated digest comparisons. | Repeated runs match; source guard asserts no Geist/ISM, scheduler, Gateway, or archive append paths are invoked. | Bounded local test only; no runtime queue/worker; no Minimal Spine v1.x change. |
| 52 | Sleep Docs Overclaim Guard | Update current docs to prevent Sleep overclaims. | Current-state index, module registry, roadmap docs, docs lint rules if needed. | Docs distinguish current, historical, and deferred Sleep claims; docs lint passes. | No behavior change; no historical docs deletion; no gate weakening. |
| 53 | Sleep Readiness Refresh | Refresh validation evidence for bounded Sleep planning/tests. | Formatting, docs lint, readiness spine, workspace-test-check, readiness gate with split workspace evidence, targeted Replay/Sleep/Geist/Consolidation tests. | Reports are fresh for the evaluated HEAD; stale/missing workspace evidence is not treated as pass. | Validation only; no production readiness claim. |
| 54 | Post-Sleep Roadmap Selection: Geist/ISM vs Runtime Scheduler vs Prod-Profile | Select next line after bounded Sleep boundary is explicit. | Compare Geist/ISM projection, Runtime Replay Scheduler, Replay/Sleep append contracts, prod-profile/workspace evidence, schema evolution. | New selection document with primary/secondary/parallel/deferred decisions. | Geist/ISM remains deferred unless projection-only scope is explicit; no hidden runtime activation. |
| 55 | Optional Sleep Evidence/Archive Append Contract, If Authorized | If explicitly selected, decide whether/how Sleep records append to canonical Evidence/Archive. | Authority ownership, record kinds, append/readback tests, no second event log. | Contract doc and tests only after explicit authorization; append path preserves canonical authority. | No Gateway write, no identity, no Geist/ISM, no scheduler; no append without explicit prompt. |
| 56 | Optional Sleep-to-Geist Handoff Boundary, If Authorized | If explicitly selected after Sleep v1, design a safe projection-only handoff to Geist/ISM. | Read-only projection, handoff records, policy gates, no implicit ISM upsert. | Handoff boundary rejects identity finalization and hidden anchor creation by default. | Dedicated Geist/ISM prompt required; no identity anchor unless separately authorized. |

## 8. Remaining Open Questions

- What does SleepApplied mean without Geist/ISM? It should mean only local Sleep subsystem boundary bookkeeping after a PASS audit, not execution, memory stabilization, identity finalization, or archive append.
- What gets archived and when? Nothing new is archived by this audit; any Sleep append/readback contract must be separately authorized.
- How does Sleep later hand off to Geist/ISM safely? Only through a dedicated later prompt with projection-only semantics, explicit gates, and no implicit ISM upsert or identity anchor.
- What remains out of scope until Geist prompts? Geist ingestion, ISM writes, SelfState authority, identity anchor/finalization, macro finalization promotion, and replay-stabilization claims.

## 9. Prompt 47/48/49 Completion and Recommended Next Prompt

Prompt 47 is complete in [`docs/roadmap/sleep_record_authority_schema_alignment.md`](sleep_record_authority_schema_alignment.md). It confirmed that `SleepPlan`, `SleepCycle`, `SleepApplied`, and `SleepBoundary` were not canonical records yet; kept the existing sleep coordinator/report surfaces as prototype inventory; deferred `ucf-types`/`ucf-protocol` promotion; and preserved the no-runtime, no-Geist/ISM, no-identity, no-Gateway, and no Evidence/Archive append boundaries.

Prompt 48 is complete as a local `MinimalSpineSleepPlanCandidate` builder in `core/crates/ucf-sleep-coordinator`. It consumes bounded Replay metadata only: a PASS `MinimalSpineReplayScheduleAudit` and optional matching `MinimalSpineReplayAppliedBoundary`. It does not activate Sleep runtime behavior, does not implement SleepApplied, does not run Replay, does not touch Geist/ISM or identity, does not append Evidence/Archive, and does not expose Gateway visibility.

Prompt 49 is complete as a local `MinimalSpineSleepPlanAudit` wrapper in `core/crates/ucf-sleep-coordinator`. It verifies `MinimalSpineSleepPlanCandidate` deterministically, emits PASS/FAIL plus a stable audit digest and failure reasons, preserves replay provenance fields, and keeps all forbidden side-effect flags false. PASS is only candidate consistency; it is not SleepApplied, not SleepCompleted, not Geist/ISM/identity state, not Evidence/Archive append, and not Gateway visibility.

Recommended next prompt: **UCF Prompt 50 — SleepApplied Boundary Without Geist/ISM**.

Prompt 50 should define only an optional local SleepApplied boundary after a PASS SleepPlan audit. It must remain local boundary bookkeeping with no Geist/ISM handoff, no identity finalization, no Evidence/Archive append, no coordinator runtime activation, and no Gateway write unless a later prompt explicitly authorizes those contracts.
