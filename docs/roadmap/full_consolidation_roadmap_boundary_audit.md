# UCF Full Micro→Meso→Macro Consolidation Roadmap and Boundary Audit

## 0. Purpose

- This is a roadmap and boundary audit only.
- This document adds no consolidation pipeline implementation, no macro-finalization implementation, no replay scheduler, and no Gateway write API.
- This is not full Replay, Geist, ISM, neuromod scheduler, DBM, HPA, real-compute, capability-issuance, or production readiness.
- Minimal Spine v1.x remains frozen; this document plans the next line without changing Minimal Spine v1.x semantics.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `23ab26a54556d762be089b025c65957f2467c168` |
| HEAD short | `23ab26a5` |
| Dirty state at audit start | clean |
| Workspace package count | 192 |
| Freeze doc present | yes |
| Compute closure present | yes |
| Micro hook test present | yes |
| `ucf-consolidation` present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -20`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and the requested file/directory presence checks.

Companion links:

- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
- [`docs/minimal_ucf_spine_v1.md`](../minimal_ucf_spine_v1.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/roadmap/consolidation_record_authority_schema_alignment.md`](consolidation_record_authority_schema_alignment.md)

## 2. Consolidation Code Inventory

| Concern | Existing API/type | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|---|
| Minimal Spine derived micro hook | `MinimalSpineMicroMilestoneCandidate` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Builds a local derived candidate from Minimal Spine Evidence/Archive/Protocol links, exposes deterministic bytes/digest, and validates non-zero links; it is explicitly not a canonical event-log record. | functional-prototype | Low if kept candidate-only; medium if interpreted as an emitted milestone. |
| Protocol milestone builders | `build_micro`, `build_meso`, `build_macro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Constructs protocol `MicroMilestone`, `MesoMilestone`, and `MacroMilestone` values from `ExperienceRecord` or lower-tier milestones. | partial | Builders exist but are broad and do not yet define candidate-vs-emitted authority for the full pipeline. |
| Deterministic milestone commits | `commit_milestone_micro`, `commit_milestone_meso`, `commit_milestone_macro`; canonical encoders in `ucf-protocol` | `core/crates/ucf-commit`; `protocol/crates/ucf-protocol/src/lib.rs` | Commit helpers and canonical encoders provide stable digests for protocol milestone structs. | functional-prototype | Medium if protocol field sufficiency is assumed without a schema-alignment prompt. |
| Append sink | `MilestoneSink`, `ArchiveMilestoneSink` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Converts milestones into derived `ExperienceRecord`s and appends them through `ExperienceAppender`; macro emission can also publish a macro-finalized index envelope. | broad-risky | High because append side effects and macro-finalized publication can be confused with authority/finalization. |
| Cycle runner | `ConsolidationKernel::run_one_cycle` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Reads recent records, selects energy-biased records, builds micro/meso/macro milestones, and appends all derived milestones. | broad-risky | High for first full-consolidation step because it combines construction, append, and macro publication. |
| Internal memory graph | `MemoryMilestoneGraph`, `build_memory_micro`, `build_memory_meso`, `build_memory_macro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Builds non-protocol memory milestone structs from digests and commits them for replay selection. | partial | Medium; these are not protocol records and must not be treated as schema authority. |
| Replay surfaces | `ReplayCascade`, `ReplayOutcome`, `run_sleep_replay`, `ReplayScheduled`, `ReplayApplied` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `core/crates/ucf-types/src/lib.rs` | Selects replay targets, constructs replay tokens, appends replay records to `ArchiveStore`, and can update sleep/workspace summaries when explicitly invoked. | broad-risky | Critical for this roadmap; next prompts must not activate or extend replay scheduler behavior. |
| Sleep integration | `SleepStateHandle`, `SleepReplayContext` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Optional state updates for derived records and replay summaries. | partial | High if full consolidation is mistaken for sleep-cycle readiness. |
| Index events | `IndexEventPublishers`, `build_record_appended_envelope`, `build_macro_finalized_envelope` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/index/crates/ucf-vector-index` | Optional publication of record-appended and macro-finalized envelopes during sink emission. | partial | High; macro-finalized event naming can overclaim identity/ISM finality. |
| Tests | Unit tests and `tests/minimal_spine_micro_hook.rs` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_micro_hook.rs` | Cover deterministic builders, append counts, replay selection, sleep replay invocation, and Minimal Spine candidate boundaries. | partial | Medium; tests prove selected behavior, not a full authority-aligned pipeline. |

Inventory answers:

- Micro/Meso/Macro structs exist in two forms: protocol-facing `ucf_protocol::v1::spec::{MicroMilestone,MesoMilestone,MacroMilestone}` and internal `ucf_types::consolidation::{MicroMilestone,MesoMilestone,MacroMilestone}` memory/replay structs.
- A candidate-vs-finalized distinction exists only for `MinimalSpineMicroMilestoneCandidate`; full micro/meso/macro candidate/finalized semantics are not yet aligned.
- Deterministic bytes/digests exist for the Minimal Spine micro candidate, protocol canonical encoders, milestone commit helpers, and internal memory/replay commits.
- Append/write APIs exist through `ExperienceAppender`, `ArchiveMilestoneSink`, `ConsolidationKernel::run_one_cycle`, `ArchiveStore::append`, and replay-specific appends in `run_sleep_replay`.
- `ArchiveMilestoneSink` exists and appends derived milestone records; it must remain an explicit sink, not an event-log authority.
- Replay APIs exist in consolidation and `runtime/ucf-replay`; they are out of scope for the next implementation prompt.
- Macro-finalized event APIs exist via optional index publishers; they are too broad for the next step.
- APIs too broad for the first step are `ConsolidationKernel::run_one_cycle`, `ArchiveMilestoneSink::emit_macro`, `run_sleep_replay`, `IndexEventPublishers::macro_finalized`, and any sleep/workspace update path.

## 3. Protocol / Schema Authority

| Record | Authority module | Existing fields | Used by consolidation? | Risk |
|---|---|---|---:|---|
| `MicroMilestone` | `ucf-protocol` / `ucf-types::v1::spec` | `milestone_id`, `achieved_at_ms`, `label` | yes | Medium; fields are minimal and may be insufficient for full source/evidence linkage without wrapping or companion records. |
| `MesoMilestone` | `ucf-protocol` / `ucf-types::v1::spec` | `milestone_id`, `achieved_at_ms`, `label`, sorted `micro_milestone_ids` | yes | Medium; aggregation linkage is ID-only and needs deterministic ID/commit policy. |
| `MacroMilestone` | `ucf-protocol` / `ucf-types::v1::spec` | `milestone_id`, `achieved_at_ms`, `label`, sorted `meso_milestone_ids` | yes | High; macro record existence must not imply identity, Geist, or ISM finalization. |
| `ExperienceRecord` | `ucf-protocol` / Archive/Evidence append surfaces | `record_id`, `observed_at_ms`, `subject_id`, `payload`, optional `digest`, `vrf_tag`, `proof_ref` | yes | High; it is the archive/evidence carrier and must not be bypassed by a second event log. |
| `ProofEnvelope` | `ucf-protocol` / `ucf-evidence` / `ucf-archive` | `envelope_id`, `payload`, optional `payload_digest`, repeated `vrf_tags`, repeated `signature_ids` | yes, through archive readback | Medium; proof content and append semantics must remain evidence/archive authority. |
| `CandidateSetRecord` | `ucf-protocol` | `version`, `input_digest`, `policy_decision_digest`, `candidate_count`, sorted `candidate_digests`, `candidates_digest`, `provenance` | indirectly via Minimal Spine candidate digests | Medium; consolidation should reference/derive from this record, not redefine it. |
| `OutputRecord` | `ucf-protocol` | `version`, `input_digest`, `candidate_set_digest`, `selected_candidate_digest`, `output_digest`, `policy_status`, `status`, `provenance`, optional `evidence_id` | indirectly via Minimal Spine candidate digests/status | Medium; consolidation should consume links/status only and must not claim output authority. |
| Replay/sleep records | `ucf-types::consolidation` plus `ucf-archive-store::RecordKind::{ReplayToken,ReplayApplied}` | digest-only replay token, scheduled, and applied structs; archive store record kinds | yes, in replay APIs | High; replay scheduling remains deferred. |

Authority decisions:

- Protocol-facing authority for milestone record schemas is `ucf-protocol` and the `ucf-types::v1::spec` re-export.
- Consolidation may construct or derive protocol milestone values, but it is not schema authority.
- Consolidation's `MinimalSpineMicroMilestoneCandidate` is a derived local candidate, not a protocol record and not an archive authority.
- Full candidate/finalized semantics for meso and macro are missing and must be defined before broad append behavior is expanded.

## 4. Evidence / Archive Boundary

| Path/API | Appends? | To what? | Authority risk | Needed for future pipeline? |
|---|---:|---|---|---:|
| `ucf_evidence::EvidenceStore::append` | yes | `EvidenceEnvelope` | Low when used as canonical evidence append authority. | yes |
| `ucf_archive::ExperienceAppender::append_with_proof` / `append` | yes | Evidence-backed archive entries with `ProofEnvelope` payloads | Low as authority; high if called implicitly by broad consolidation APIs. | yes |
| `InMemoryArchiveHandle::append_with_proof` | yes | Underlying `InMemoryArchive` | Medium; convenience handle can hide append side effects. | possibly |
| `ArchiveMilestoneSink::emit_micro` / `emit_meso` / `emit_macro` | yes | Derived `ExperienceRecord`s through `ExperienceAppender` | High; sink appends and can publish events. | yes, but only behind explicit contract/tests. |
| `ConsolidationKernel::run_one_cycle` | yes | Multiple derived milestone `ExperienceRecord`s | High; construction and append are coupled. | later, after builders and append contract. |
| `ConsolidationKernel::run_sleep_replay` | yes | `ArchiveStore` records of kind `ReplayToken` and `ReplayApplied` | Critical for this line; not needed for first consolidation prompts. | no for next prompt |
| `ArchiveStore::append` | yes | `ArchiveRecord` and root commitment surface | Medium; separate local archive store must not become a second event log for milestones. | yes, only for explicit archive/readback tests. |
| `MinimalSpineMicroMilestoneCandidate` | no | none | Low; intentionally derived/no append. | yes as source-link pattern |

Boundary conclusions:

- Evidence/Archive remains the canonical append/readback proof surface.
- Consolidation may derive milestone records and candidate records but must not become a second event-log authority.
- `ArchiveMilestoneSink` should be reused or wrapped only after Prompt 26 defines record authority and Prompt 28 defines an explicit append contract.
- Future tests must prove that append is opt-in, auditable, replay-free, Geist-free, and readback-verifiable.

## 5. Replay / Sleep / Geist Boundary

| Area | Existing API/type | Current behavior | Out-of-scope for next prompt? | Boundary |
|---|---|---|---:|---|
| Replay runtime | `runtime/ucf-replay` | Provides deterministic replay/report tooling as a separate runtime package. | yes | Do not wire scheduler behavior into consolidation prompts. |
| Consolidation replay | `ReplayCascade`, `ReplayOutcome`, `run_sleep_replay` | Schedules digest-only replay targets and can append replay token/applied records when called. | yes | Leave untouched; no new replay records, scheduler, or activation. |
| Sleep state | `SleepStateHandle`, `SleepReplayContext` | Optional derived-record and replay-summary updates. | yes | No sleep-cycle readiness claim from consolidation milestones. |
| Geist | `domains/geist/crates/ucf-geist` | Existing research/partial Geist state, recursion, sleep, policy, and archive surfaces. | yes | No Geist writes, no identity finalization, no self-state authority. |
| ISM/archive anchors | `RecordKind::IsmAnchor` and related references | Archive store can represent ISM anchors, but full consolidation does not need them. | yes | Macro milestones must not imply ISM anchoring. |
| Neuromod | `domains/ucf-neuromod/src/minimal_spine.rs` | Minimal metadata envelope hook is derived and scheduler-free. | yes | No neuromod scheduler, DBM, or HPA integration. |

## 6. Target Scope

| Layer | Goal | Required inputs | Outputs | Explicit non-goals |
|---|---|---|---|---|
| Micro | Deterministically create micro milestone candidates/records from Minimal Spine links, `OutputRecord`, and evidence/archive readback. | `CandidateSetRecord` digest, `OutputRecord` digest/status, `EvidenceId`, archive output key/event digest, input digest, source/provenance. | Derived micro candidate first; later protocol `MicroMilestone` plus explicit archive payload when contracted. | No replay scheduling, macro finalization, Gateway writes, output authority, or second event log. |
| Meso | Deterministically aggregate approved/archived micro milestones into meso candidates/records. | Ordered or sorted micro milestone IDs/commits, source batch metadata, evidence/archive references from micro layer. | Meso candidate first; later protocol `MesoMilestone` plus explicit archive/readback proof. | No sleep cycle activation, no policy override, no nondeterministic grouping. |
| Macro | Deterministically build macro milestone candidates from meso milestones with clearly bounded finalization language. | Meso milestone IDs/commits, aggregation policy, archive/evidence provenance, explicit candidate/finalized state semantics. | Macro candidate first; later protocol `MacroMilestone` append only after boundary tests. | No Geist/ISM identity finalization, no capability issuance, no replay scheduler, no real-compute activation. |

Full Micro→Meso→Macro Consolidation later means deterministic creation/aggregation/candidate behavior, explicit Evidence/Archive append behavior, stable digests, and no implicit Replay/Geist/ISM/identity claims.

## 7. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| Macro finalization overclaim | critical | Existing sink can publish macro-finalized envelopes during `emit_macro`. | Prompt 26 must define macro candidate/finalized vocabulary; no macro-finalized event expansion before dedicated boundary tests. |
| Replay accidental activation | critical | `run_sleep_replay` appends replay token/applied archive records when invoked. | Keep replay APIs untouched; tests for full consolidation must assert no replay records unless explicit replay prompt. |
| Geist/ISM identity overclaim | critical | Geist exists as partial/research surface and archive store has `IsmAnchor`. | Macro milestones are not identity or ISM finality; no Geist/ISM writes in consolidation prompts. |
| Evidence/Archive authority confusion | high | `ArchiveMilestoneSink` appends derived records through archive/evidence APIs. | Treat Evidence/Archive as authority and consolidation as caller/deriver only; append must be explicit and audited. |
| Derived candidate vs emitted milestone confusion | high | Minimal Spine hook is candidate-only while broader builders emit protocol milestones. | Introduce schema/candidate alignment before implementation; name derived candidates explicitly. |
| Protocol schema mismatch | high | Protocol milestone fields are minimal and may lack evidence/source fields. | Prompt 26 evaluates whether wrapper/companion records are needed before changing behavior. |
| Non-deterministic aggregation | high | Builders preserve some input order while canonical encoders sort repeated ID lists. | Define deterministic input ordering and golden digests for each layer. |
| Hidden append side effects | high | `run_one_cycle` builds and appends in one call. | Separate pure builders from sinks; first implementation stage should be pure/no-append. |
| Test fixture drift | medium | Existing tests cover specific fixtures and append counts only. | Add golden fixtures and current HEAD validation in later prompts. |
| Docs overclaim from historical consolidation docs | medium | Historical docs contain broad consolidation/replay/sleep language. | Current roadmap/index docs must outrank historical docs and label them audit trail unless refreshed. |

## 8. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 26 | Consolidation Record Authority and Schema Alignment | Decide protocol-vs-derived authority for micro/meso/macro records before implementation. | Docs/tests only unless compile-time assertions are needed. | Authority matrix updated; missing fields/wrappers identified; no behavior changes. | No append, no macro finalization, no replay, no Geist/ISM. |
| 27 | Deterministic MicroMilestone Builder from Minimal Spine Links | Add a pure deterministic micro builder aligned with Prompt 26. | Pure builder and golden tests from Minimal Spine links. | Stable bytes/digest; candidate/record semantics explicit. | No archive append or scheduler. |
| 28 | MicroMilestone Evidence/Archive Append Contract | Define and test explicit micro append/readback behavior. | Narrow sink or wrapper; audit tests. | Append is opt-in, readback verified, no hidden side effects. | Evidence/Archive remains authority; no second event log. |
| 29 | Deterministic MesoMilestone Aggregation | Add pure meso aggregation from archived/approved micro milestones. | Pure aggregation, ordering policy, golden digest tests. | Same inputs produce same meso candidate/record. | No macro, replay, sleep, or Geist. |
| 30 | MesoMilestone Archive/Readback Tests | Add explicit meso append/readback proof tests. | Meso-only archive contract. | Meso archive payload and proof read back deterministically. | No macro-finalized events. |
| 31 | MacroMilestone Candidate Builder | Add pure macro candidate builder from meso milestones. | Candidate-only macro construction and golden tests. | Macro candidate stable and clearly non-identity. | No finalization, no Geist/ISM, no replay. |
| 32 | MacroMilestone Finalization Boundary Without Geist/ISM | Define bounded macro emission/finalization semantics. | Boundary docs/tests; possibly no-op or disabled sink checks. | Macro finalization cannot imply identity/ISM or capability authority. | No Geist/ISM write, no identity finalization claim. |
| 33 | Consolidation Pipeline E2E Determinism | Add pure or explicitly appended end-to-end determinism tests. | Micro→meso→macro deterministic flow under controlled fixtures. | Repeated runs match digests and readback. | Replay scheduler remains absent; no compute activation. |
| 34 | Consolidation Docs Overclaim Guard | Update current docs to guard historical claims. | Docs-only. | Current-state index and registry point to validated capabilities and deferred surfaces. | Do not delete historical docs. |
| 35 | Consolidation Readiness Refresh | Re-run workspace, targeted tests, docs lint, readiness gate, and clippy for the consolidation line. | Validation/report discipline. | Fresh validation summary; no committed `out/*.json` unless required. | No new features. |

## 9. Open Questions

- Are protocol milestone records sufficient for source/evidence/archive linkage, or are wrapper/companion records required?
- Where should micro/meso/macro builders live: `ucf-consolidation`, a code-near protocol helper, or a narrower submodule?
- Should `ArchiveMilestoneSink` be reused directly, wrapped behind an explicit contract, or split into pure builder plus append sink?
- What is candidate vs finalized semantics for each layer, especially macro?
- What gets archived and when: candidate, protocol milestone, derived `ExperienceRecord`, proof envelope, archive-store record, or all of these?
- What remains out of scope until Replay/Geist prompts: replay scheduler, sleep cycles, Geist/ISM writes, identity finalization, neuromod scheduler, capability issuance, and Gateway write APIs?

## 10. Prompt 26 Schema Alignment Status

- Prompt 26 schema alignment is available at [`docs/roadmap/consolidation_record_authority_schema_alignment.md`](consolidation_record_authority_schema_alignment.md).
- Recommended next prompt: **UCF Prompt 27 — Deterministic MicroMilestone Builder from Minimal Spine Links**.

## 11. Prompt 27 Completion Note

Prompt 27 is complete. The deterministic MicroMilestone builder from Minimal Spine links is implemented in `ucf-consolidation` as a pure wrapper-returning builder, not as an append sink.

| Prompt | Status | Implemented surface | Boundary result | Recommended next prompt |
|---:|---|---|---|---|
| 27 | complete | `build_micro_milestone_from_minimal_spine_candidate` and `MinimalSpineMicroMilestoneBuildOutput` | Micro-only, append-free, replay/sleep/geist/ISM-free, no meso/macro aggregation, no Minimal Spine v1.x changes | Prompt 28 — MicroMilestone Evidence/Archive Append Contract |

Prompt 27 chose Option B because the current protocol `MicroMilestone` fields are not a complete provenance container for Minimal Spine links. The wrapper preserves candidate digest, input digest, candidate-set digest, output-record digest, evidence id, archive output key, archive output event digest, protocol micro milestone digest, and source marker while keeping Evidence/Archive append authority deferred to Prompt 28.
