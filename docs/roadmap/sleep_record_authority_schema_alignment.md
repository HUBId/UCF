# UCF Sleep Record Authority and Schema Alignment

## 0. Purpose

- This document decides Sleep-facing record authority only.
- It does not implement Sleep behavior, SleepApplied, a Sleep Cycle Coordinator runtime, Replay scheduler/queue/worker behavior, Gateway writes, capability issuance, real-compute activation, Evidence/Archive append, or a second event-log authority. Prompt 48 adds only a deterministic SleepPlan candidate builder from bounded Replay metadata.
- It does not integrate Geist/ISM and does not finalize identity.
- It preserves the Minimal Spine v1.x freeze and does not change bounded Replay or bounded Consolidation behavior.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `2f21c19f0451793343be673c8d3108840ff72a8d` |
| HEAD short | `2f21c19f` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Sleep roadmap present | yes |
| Replay closure present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |
| Sleep-related dirs | `./core/crates/ucf-sleep-coordinator` |

Baseline commands used for this alignment: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -30`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, presence checks for the Sleep roadmap, Replay closure, `runtime/ucf-replay`, `domains/geist/crates/ucf-geist`, and `find . -path "*target*" -prune -o -type d -name "*sleep*" -print | sort`.

Required context links:

- [`docs/roadmap/sleep_integration_roadmap_boundary_audit.md`](sleep_integration_roadmap_boundary_audit.md)
- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

## 2. Sleep Record / API Inventory

| Record / Type / API | Path | Fields / role summary | Current use | Maturity | Risk |
|---|---|---|---|---|---|
| `MinimalSpineSleepPlanCandidate`, `MinimalSpineSleepPlanInput`, `build_sleep_plan_candidate_from_replay_audit`, `build_sleep_plan_candidate_from_replay_boundary` | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `core/crates/ucf-sleep-coordinator/tests/minimal_spine_sleep_plan_candidate.rs` | Prompt 48 local wrapper and pure deterministic builder from PASS `MinimalSpineReplayScheduleAudit` plus optional matching `MinimalSpineReplayAppliedBoundary`, or from validated digest input. Hard flags keep `candidate_only=true` and Sleep/Geist/ISM/identity/Evidence/Archive/Gateway flags false. | Implemented candidate-only; not canonical protocol schema and not runtime authority. | functional-prototype / candidate-only | Safe only as bounded metadata. It must not be used to claim Sleep execution, SleepApplied, coordinator runtime activation, Geist/ISM ingestion, identity finalization, Evidence/Archive append, or Gateway visibility. |
| `SleepCycle` | not found as a canonical record | Existing `cycle_id` fields occur in sleep/report/RSA/TCF contexts, but no bounded SleepCycle record exists. | Naming only in local reports and docs. | unknown | The term can be misread as an active runtime scheduler loop. |
| `SleepApplied` / `SleepAppliedBoundary` | not found | No local Sleep applied boundary or canonical applied record exists. | Missing; future local-only boundary may be designed later. | unknown | The term can overclaim Geist ingestion, ISM write, identity stabilization, or memory anchoring. |
| `SleepBoundary` | not found | No generic SleepBoundary record exists. | Missing. | unknown | Too broad unless constrained to a precise local boundary. |
| `WalSleepCoordinator`, `SleepHeuristics`, `SleepState`, `SleepTrigger`, `RecentMetrics`, `SleepReplaySummary`, `SleepTriggered`, `SleepStateUpdater`, `SleepPhaseRunner` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Local WAL-style trigger/report state, bounded recent metrics/evidence window, replay summary counters, structural stats/proposal slots, deterministic trigger evaluation, and adapter to `ucf-rsa::SleepCoordinator`. | Runnable local functional prototype with in-source unit tests. | functional-prototype | Dangerous if promoted to post-Replay Sleep v1 authority because it can trigger a report path and is not the planned bounded Replay metadata candidate builder. |
| `SleepCoordinator`, `SleepReportReady`, `build_sleep_record` | `core/crates/ucf-rsa/src/lib.rs` | Policy-gated sleep phase runner that builds a sleep report `ExperienceRecord`, appends through an archive appender, optionally commits structural records, publishes `SleepReportReady`, and returns the event. | Existing local RSA sleep-report prototype. | functional-prototype / unsafe-broad for Prompt 47 | It appends archive/evidence-like records in its current context; Prompt 47 does not authorize any new Sleep Evidence/Archive append contract. |
| `TemporalCoordinator`, `TcfCore`, `TcfPlan.sleep_active`, `TcfPlan.replay_active` | `core/crates/ucf-tcf/src/lib.rs` | Deterministic fixed-point temporal gating with sleep/replay active flags and gain caps. | Contextual temporal prototype. | partial | Names can imply runtime orchestration; not SleepPlan authority. |
| `ReplayToken`, `ReplayScheduled`, `ReplayApplied` primitives | `core/crates/ucf-types/src/lib.rs` | Shared replay primitives; `ReplayToken` has bounded metadata, `ReplayScheduled` mirrors token shape, `ReplayApplied` carries an effect digest. | Existing replay-facing primitive layer, not Sleep authority. | partial / unsafe-broad for applied semantics | `ReplayApplied` can overclaim actual replay effects or Geist stabilization and must not become SleepApplied. |
| `MinimalSpineReplayTokenBuildOutput`, `MinimalSpineReplayScheduleBuildOutput`, `MinimalSpineReplayScheduleAudit`, `MinimalSpineReplayAppliedBoundary` | `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs` | Deterministic replay token builder output, planned schedule, verify-only audit, and local replay-subsystem applied boundary with forbidden side-effect flags. | Future immutable metadata inputs for SleepPlan candidate construction. | functional-prototype | Safe only as read-only inputs; Sleep must not execute Replay or mutate schedules/tokens. |
| `GeistKernel`, `SelfState`, `InMemoryIsm`, `ReplayStabilization`, `apply_replay_effects`, optional `SleepStateHandle` updates | `domains/geist/crates/ucf-geist/src/lib.rs` | Geist can ingest macro milestones, append a derived record, gate ISM anchor upserts, update optional sleep state, and compute replay stabilization from legacy `ReplayApplied` effects. | Existing partial Geist prototype. | partial / unsafe-broad | Out of scope for Sleep v1; must not be called by SleepPlan/SleepApplied prompts. |
| `EvidenceEnvelope`, `EvidenceStore::append` | `core/crates/ucf-evidence/src/lib.rs` | Evidence append/get/len authority. | Canonical evidence store support. | operational for Minimal Spine | Sleep does not take Evidence authority and does not append in Prompt 47. |
| `ExperienceAppender`, `FileArchive::append_and_fold`, `InMemoryArchive` | `domains/archive/crates/ucf-archive/src/lib.rs` | Archive/evidence append helper surfaces for experience records and fold state. | Canonical archive append support. | operational for Minimal Spine | Existing RSA sleep report append does not authorize new Sleep append semantics. |
| `ArchiveStore`, `ArchiveAppender`, `RecordKind::ReplayToken`, `RecordKind::ReplayApplied`, `RecordKind::CyclePlan`, `RecordKind::IsmAnchor` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Deterministic local archive records and kinds. | Archive-store support surface. | functional-prototype | Existing broad record kinds are not Sleep append authority. |
| Protocol replay evidence types | `protocol/crates/ucf-protocol/src/lib.rs` | Protocol-level `ReplayRunEvidence` and microcircuit evidence references; no SleepPlan/SleepApplied schema found. | Protocol-facing legacy/support schema. | partial | Promoting Sleep to protocol now would freeze schema too early. |
| Ingestion sleep loop adapters | `core/crates/ucf-ingestion/src/lib.rs` | Optional sleep loop wiring can publish `SleepTriggered` envelopes and accept a sleep loop handle. | Integration prototype. | partial | Runtime-facing surface; not authorized as Sleep Cycle Coordinator behavior here. |

Inventory answers:

- `SleepPlan`: Prompt 48 added a local `MinimalSpineSleepPlanCandidate` wrapper and builder; canonical protocol/shared SleepPlan schema remains deferred.
- `SleepCycle`: missing as a canonical record; only names/fields such as `cycle_id` and `sleep_cycle` exist.
- `SleepApplied`: missing.
- `SleepBoundary`: missing.
- `SleepCoordinator`: exists as local functional prototype split between `ucf-sleep-coordinator` and `ucf-rsa`.
- Trigger/report/WAL state: exists in `ucf-sleep-coordinator`; report append exists in `ucf-rsa`.
- Canonical digest/canonical encoding for local SleepPlan candidate: implemented in `MinimalSpineSleepPlanCandidate::deterministic_bytes` and `digest`; canonical SleepApplied encoding remains missing.
- Tests: existing local sleep coordinator/RSA tests exist; no bounded Replay-to-Sleep E2E exists.
- Replay inputs: bounded Replay audit/schedule/token/applied-boundary outputs exist and may be future immutable inputs.
- Geist/ISM references: exist in `ucf-geist`, including optional sleep-state update and replay stabilization from `ReplayApplied`; out of scope.
- Evidence/Archive append references: exist in RSA sleep report and canonical evidence/archive crates; no new Sleep append is authorized.
- Gateway/runtime scheduler references: runtime/gateway write behavior is not present for SleepPlan authority; ingestion sleep loop wiring is prototype/runtime-facing only.

## 3. Authority Decisions

| Concern | Decision | Reason |
|---|---|---|
| SleepPlan authority | Prompt 48 implements Option B as a local `ucf-sleep-coordinator` candidate wrapper. | SleepPlan starts as a pure deterministic candidate over bounded Replay metadata, not as shared protocol/types schema and not as the existing report-running coordinator. |
| SleepCycle authority | No canonical authority yet; if used later, keep it local to Sleep planning as a bounded descriptor. | Current cycle naming is broad and can imply an active runtime loop. |
| SleepApplied authority | No canonical authority yet; if used later, keep it as local Sleep subsystem bookkeeping only. | Existing `ReplayApplied` and Geist replay stabilization are too broad and must not be reused as SleepApplied. |
| SleepCoordinator authority | Existing `ucf-sleep-coordinator` remains a functional prototype/inventory source, not post-Replay Sleep v1 authority. | It has trigger/report behavior and can invoke a report path; Prompt 47 is authority alignment only. |
| Replay input role | PASS `MinimalSpineReplayScheduleAudit` is required for the direct builder; `MinimalSpineReplayAppliedBoundary` is optional immutable provenance and must match audit digest, schedule digest, and token count. A local digest input exists to avoid widening handle surfaces. | Replay is the upstream bounded metadata authority; Sleep must not execute Replay or mutate schedules/tokens. |
| Geist/ISM role | out of scope | SleepPlan and SleepApplied do not call Geist, write ISM, create anchors, or claim identity/memory stabilization. |
| Evidence/Archive role | unchanged | Evidence/Archive remain their own authorities; no Sleep append is added in Prompt 47. |
| Protocol role | deferred/current | No `ucf-protocol` SleepPlan/SleepApplied promotion in Prompt 47 because protocol-facing schema would freeze semantics too early. |

Prompt 47 therefore chooses **Option E for this prompt**: docs-only alignment now, with **Option B as the recommended future implementation direction** if Prompt 48 is authorized.

## 4. Naming / Semantics Boundary

| Term | Allowed meaning now | Explicitly not allowed |
|---|---|---|
| `SleepPlan` | Prompt 48 local deterministic candidate/plan over bounded Replay metadata such as PASS Replay audit, optional Replay applied-boundary provenance, and schedule/token digests. | Not execution, not sleep completion, not report append, not Geist/ISM, not identity finalization, not Gateway-visible behavior. |
| `SleepCycle` | If introduced later, only a bounded descriptor in local Sleep planning. | Not a runtime scheduler loop, queue, worker, replay executor, or global cycle authority. |
| `SleepApplied` | If introduced later, local Sleep subsystem bookkeeping after a PASS SleepPlan audit. | Not Geist ingestion, not ISM write, not identity stabilization, not memory anchor, not Replay execution proof, not Evidence/Archive append. |
| `SleepAppliedBoundary` | Preferred future name over broad `SleepApplied` for a local-only boundary marker. | Not a completed cognitive/memory event and not cross-domain authority. |
| `SleepCompleted` | Avoid if possible; if present historically, read as local and non-identity only. | Not memory stabilization, not identity finalization, not human-level completion, not production readiness. |
| Sleep stabilization | At most local bounded planning/audit stability for deterministic digests. | Not Geist replay stabilization, ISM stabilization, identity stabilization, or memory consolidation proof. |
| Geist/ISM handoff | later | No implicit handoff, upsert, anchor, self-state finalization, or identity claim. |

Future names allowed with narrow semantics: `SleepPlanCandidate`, `SleepPlanAudit`, `SleepAppliedBoundary`, `SleepReplayInputRef`, and `SleepPlanProvenance`.

Names to avoid or require explicit caveats: `SleepCompleted`, `SleepStabilized`, `SleepFinalized`, `MemoryStabilized`, `IdentityStabilized`, `GeistHandoff`, `IsmApplied`, `ReplayExecutedBySleep`, and generic `SleepBoundary` without a narrower suffix.

## 5. Replay Input Boundary

- PASS `MinimalSpineReplayScheduleAudit` is the direct input to Prompt 48 SleepPlan candidate construction.
- `MinimalSpineReplayAppliedBoundary` is an optional immutable provenance input; when supplied it must match audit digest, schedule digest, and token count.
- `MinimalSpineReplayScheduleBuildOutput`, scheduled token provenance, `MinimalSpineReplayTokenBuildOutput` digests, and token/schedule provenance may be consumed only as read-only metadata.
- Sleep does not execute Replay.
- Sleep does not mutate Replay schedules or tokens.
- Sleep does not turn `ReplayApplied` into SleepApplied, and does not claim Replay execution proof.

## 6. Evidence / Archive Boundary

- No Evidence/Archive append is added in Prompt 47.
- Existing Evidence/Archive modules remain the append/readback authorities.
- Existing RSA sleep report append behavior remains a local historical/prototype surface, not a new Sleep v1 append contract.
- Any future Sleep append must be explicit, must name the authority, must avoid a second event log, and must include append/readback tests and overclaim guards.

## 7. Geist / ISM / Identity Boundary

- Geist, ISM, identity finalization, identity anchoring, and memory stabilization are out of scope for Prompt 47.
- `SleepApplied` is not Geist ingestion and is not an ISM write.
- Sleep completion is not identity finalization.
- Future Sleep-to-Geist handoff requires a dedicated prompt with projection-only semantics, explicit gates, and no implicit anchor creation.

## 8. Prompt 48 Acceptance Criteria

Prompt 48 satisfies the following before a SleepPlan candidate claim is allowed:

1. Implement a pure deterministic SleepPlan candidate builder only.
2. Accept input only from bounded Replay artifacts or their digests/provenance: PASS Replay audit, optional Replay applied-boundary reference, schedule digest, token digests, and bounded provenance.
3. Reject failed Replay audit status and forbidden Replay/Sleep/Geist/ISM/identity/Evidence/Archive/Gateway flags.
4. Produce stable deterministic bytes and a stable digest across repeated runs.
5. Define deterministic ordering and duplicate handling for Replay token/provenance inputs.
6. Add boundary tests proving no Sleep runtime, no SleepCoordinator execution, no Replay execution, no scheduler/queue/worker, no Geist/ISM, no Evidence/Archive append, no identity finalization, and no Gateway visibility.
7. Keep `SleepApplied` and `SleepAppliedBoundary` out unless a later prompt explicitly introduces a local boundary after a PASS SleepPlan audit.
8. Do not promote SleepPlan to `ucf-protocol` or `ucf-types` unless a later authority prompt explicitly authorizes schema-wide promotion.
9. Preserve Minimal Spine v1.x and current bounded Replay/Consolidation behavior.

## 9. Remaining Open Questions

- What does the future verify-only SleepPlan audit status/failure schema look like?
- What does SleepApplied mean without Geist/ISM beyond local bookkeeping?
- What gets archived and when, if a later prompt authorizes a Sleep append contract?
- How does Sleep later hand off to Geist/ISM safely without implicit upsert, anchor creation, or identity claims?

## 10. Prompt 48 Candidate Builder Completion

Prompt 48 implements a local candidate-only surface in `core/crates/ucf-sleep-coordinator`:

| Surface | Path | Meaning | Boundary |
|---|---|---|---|
| `MinimalSpineSleepPlanInput` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Digest/provenance wrapper containing replay audit digest, replay schedule digest, optional replay applied-boundary digest, token count, and replay source. | Metadata only; no runtime/store/appender/Gateway/Geist/ISM/scheduler handles. |
| `MinimalSpineSleepPlanCandidate` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Deterministic candidate value with `sleep_plan_digest` and replay-boundary provenance. | `candidate_only=true`; `sleep_applied`, `sleep_completed`, `geist_ingested`, `ism_written`, `identity_anchor`, `evidence_archive_appended`, and `gateway_visible` are hard false. |
| `build_sleep_plan_candidate_from_replay_audit` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Direct builder from PASS `MinimalSpineReplayScheduleAudit` plus optional matching `MinimalSpineReplayAppliedBoundary`. | Rejects FAIL audits, audit digest mismatch, schedule digest mismatch, forbidden flags, and boundary audit/schedule/token-count mismatches. |
| `build_sleep_plan_candidate_from_replay_boundary` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Pure builder from already-extracted replay-boundary digests. | Rejects zero digests, zero token count, and empty source. |
| Targeted tests | `core/crates/ucf-sleep-coordinator/tests/minimal_spine_sleep_plan_candidate.rs` | Determinism, digest-change, FAIL/invalid rejection, optional boundary matching, provenance preservation, and hard side-effect flags. | No Sleep runtime activation and no Geist/ISM/identity/Evidence/Archive/Gateway effects. |

Schema note: Prompt 48 intentionally uses a local wrapper instead of promoting a canonical protocol/shared SleepPlan schema. The remaining schema gap is a future verify-only SleepPlan audit contract and, later only if authorized, a local SleepApplied boundary.

## 11. Recommended Next Prompt

**UCF Prompt 49 — SleepPlan Verify-Only Audit Contract**.
