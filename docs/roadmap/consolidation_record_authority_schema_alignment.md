# UCF Consolidation Record Authority and Schema Alignment

## 0. Purpose

This document establishes Micro/Meso/Macro record authority before implementation of a new consolidation pipeline.
It is not a pipeline implementation, not Macro finalization implementation, and not Replay/Geist/ISM readiness.
It preserves Minimal Spine v1.x, Evidence/Archive authority, and protocol schema authority.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `23ab26a54556d762be089b025c65957f2467c168` |
| HEAD short | `23ab26a5` |
| Dirty state at audit start | clean |
| Workspace package count | 192 |
| Consolidation roadmap present at audit start | no |
| Micro hook test present | yes |
| `ucf-consolidation` present | yes |
| `ucf-protocol` present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |

Required companion sources:

- [`docs/roadmap/full_consolidation_roadmap_boundary_audit.md`](full_consolidation_roadmap_boundary_audit.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/minimal_ucf_spine_v1.md`](../minimal_ucf_spine_v1.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -20`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and file/directory presence probes for consolidation roadmap, micro hook test, `ucf-consolidation`, `ucf-protocol`, `ucf-replay`, and `ucf-geist`.

## 2. Record Inventory

| Record / Type | Path | Module | Current role | Candidate or emitted? | Schema authority? | Notes |
|---|---|---|---|---|---|---|
| `MinimalSpineMicroMilestoneCandidate` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | `ucf_consolidation` | local-candidate | candidate-only | `ucf-consolidation` | Derived from Minimal Spine links; not a protocol milestone and not an append surface by itself. |
| `MicroMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md`; `protocol/crates/ucf-protocol/spec/messages_v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing milestone with canonical encoding; currently no version field and only ID/time/label fields. |
| `MesoMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md`; `protocol/crates/ucf-protocol/spec/messages_v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing aggregate milestone; references micro milestone IDs. |
| `MacroMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md`; `protocol/crates/ucf-protocol/spec/messages_v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing macro milestone; references meso milestone IDs. The `finalized` word is not present in this spec record. |
| Top-level `MacroMilestone` / `MacroMilestoneAppend` | `protocol/crates/ucf-protocol/src/lib.rs` | `ucf_protocol::v1` | protocol-record | unclear | `ucf-protocol` | Separate top-level macro append shape with `MacroMilestoneState::Finalized`; must not be treated as Geist/ISM identity finalization. |
| `ucf_types::consolidation::MicroMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | emitted-record | `ucf-types` | Memory/consolidation primitive with digest commitments; used by replay cascade helpers, not the protocol-facing milestone schema authority. |
| `ucf_types::consolidation::MesoMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | emitted-record | `ucf-types` | Memory/consolidation aggregate primitive over micro commits. |
| `ucf_types::consolidation::MacroMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | emitted-record | `ucf-types` | Memory/consolidation aggregate primitive over meso commits and trait-update digest. |
| `ReplayToken`, `ReplayScheduled`, `ReplayApplied` | `core/crates/ucf-types/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | `ucf_types::consolidation`; `ucf_consolidation` | replay-token | emitted-record | `ucf-types` | Existing replay cascade records are out of scope for Prompt 27 and must not be triggered by pure Micro builder work. |
| `ExperienceRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `core/crates/ucf-evidence/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs`; `runtime/ucf-replay/src/lib.rs` | `ucf_protocol::v1::spec` plus evidence/archive/replay users | archive-payload | emitted-record | `ucf-protocol` | Canonical payload/proof carrier for Evidence/Archive and replay audit inputs. |
| `ProofEnvelope` | `protocol/crates/ucf-protocol/src/lib.rs`; `core/crates/ucf-evidence/src/lib.rs` | `ucf_protocol::v1::spec` | archive-payload | emitted-record | `ucf-protocol` | Proof wrapper used by evidence stores; not a second event log. |
| `ArchiveMilestoneSink` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | `ucf_consolidation` | archive-payload | emitted-record | `ucf-consolidation` | Broad side-effecting sink: appends derived records, updates sleep state, and can publish index/macro-finalized events. Not safe as pure builder authority. |
| `RecordKind::ReplayToken`, `RecordKind::ReplayApplied`, `RecordKind::IsmAnchor`, `RecordKind::OutputEvent` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | `ucf_archive_store` | archive-payload | emitted-record | `ucf-archive-store` | Archive-store record kinds exist, but Prompt 27 must not use replay or ISM anchor writes. |
| `commit_milestone_micro`, `commit_milestone_meso`, `commit_milestone_macro` | `core/crates/ucf-commit/src/lib.rs` | `ucf_commit` | builder-output | not-applicable | `ucf-commit` | Deterministic commitments over protocol milestone records; useful for later digest tests, not schema authority. |
| Geist-related records | `domains/geist/crates/ucf-geist/src/*` | `ucf_geist` | docs-only/unknown for this prompt | not-applicable | unknown | Geist/ISM write and identity finalization are out of scope. |

## 3. Protocol Schema Alignment

| Protocol record | Exists? | Versioned? | Canonical encoding tested? | Minimal Spine link capable? | Overclaim risk | Gap |
|---|---:|---:|---:|---:|---|---|
| `CandidateSetRecord` | yes | yes | yes | yes, via digest references | low | None for Prompt 26; already Minimal Spine authority. |
| `OutputRecord` | yes | yes | yes | yes, via `candidate_set_digest`, `output_digest`, and optional `evidence_id` | low | None for Prompt 26; already Minimal Spine authority. |
| `ExperienceRecord` | yes | no | yes | yes, via payload/proof reference | medium | No milestone-specific payload contract for Micro/Meso/Macro append yet. |
| `MicroMilestone` | yes | no | yes, after Prompt 26 boundary test hardening | partially, only by deterministic ID/label convention today | medium | No explicit candidate provenance or Minimal Spine link fields. |
| `MesoMilestone` | yes | no | yes, after Prompt 26 boundary test hardening | indirectly, through micro milestone IDs | medium | No explicit aggregation provenance field. |
| `MacroMilestone` | yes | no | yes, after Prompt 26 boundary test hardening | indirectly, through meso milestone IDs | high | Existing macro schema does not define consolidation-level finalization semantics. |
| Top-level `MacroMilestoneAppend` | yes | no | not specifically tested in this prompt | no | high | Contains `Finalized` state naming; must be bounded as consolidation-only and not identity/Geist/ISM finalization. |
| `ProofEnvelope` | yes | no | yes for canonical encoding support | yes, can wrap payloads after explicit contract | medium | Appendix contract must define when milestone payloads are wrapped and appended. |

Protocol-facing Micro/Meso/Macro records exist in `ucf-protocol` and are re-exported through `ucf-types::v1::spec`.
They do not have `version` fields today, unlike `CandidateSetRecord` and `OutputRecord`.
Their canonical encoding is implemented, and Prompt 26 adds small tests for deterministic canonical bytes and prost roundtrip.
They can reference Minimal Spine material only indirectly today through deterministic IDs/labels or future builder conventions; they do not yet carry structured `CandidateSetRecord`, `OutputRecord`, `EvidenceId`, or archive-key fields.
No protocol milestone field may be interpreted as Macro identity anchor production, Geist/ISM finalization, Replay completion proof, Sleep completion, Gateway write permission, or capability issuance.

## 4. Consolidation API Authority Audit

| API / Function | Path | Pure? | Side effects? | Uses Archive/Evidence? | Uses Replay? | Safe for Prompt 27? | Reason |
|---|---|---:|---:|---:|---:|---:|---|
| `MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | references only | no | yes | Local deterministic candidate constructor. |
| `MinimalSpineMicroMilestoneCandidate::deterministic_bytes` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | references only | no | yes | Stable candidate byte encoding. |
| `MinimalSpineMicroMilestoneCandidate::digest` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | references only | no | yes | Domain-separated local candidate digest. |
| `MinimalSpineMicroMilestoneCandidate::validate_links_nonzero` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | references only | no | yes | Boundary validation without append. |
| `build_micro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | consumes `ExperienceRecord` values | no | no | Existing broad builder builds protocol micro milestones from experience records, not from Minimal Spine candidates; Prompt 27 needs a narrower builder. |
| `build_meso` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | no | no | no | Meso aggregation is later scope. |
| `build_macro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | no | no | no | Macro candidate/finalization is later scope. |
| `ArchiveMilestoneSink::emit_micro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | no | yes | appends via `ExperienceAppender` | no | no | Side-effecting append, publish, and sleep-state integration surface. |
| `ArchiveMilestoneSink::emit_meso` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | no | yes | appends via `ExperienceAppender` | no | no | Side-effecting append, publish, and sleep-state integration surface. |
| `ArchiveMilestoneSink::emit_macro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | no | yes | appends via `ExperienceAppender` | no | no | Also publishes macro-finalized event when publishers are configured. |
| `ConsolidationKernel::run_one_cycle` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | no | yes | appends milestones | no | no | Existing end-to-end cycle is broad and must not become Prompt 27 authority. |
| `ConsolidationKernel::run_sleep_replay` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | no | yes | appends replay archive records | yes | no | Replay Scheduler/Sleep integration is explicitly out of scope. |
| `ReplayCascade::schedule` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | no direct append | yes | no | Deterministic replay scheduling logic exists but is not Prompt 27 scope. |
| `build_memory_micro`, `build_memory_meso`, `build_memory_macro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | consumes `ExperienceRecord` values | used by replay graph | no | Memory graph support is replay-adjacent and outside Micro builder scope. |

## 5. Candidate vs Emitted Semantics

| Concept | Semantics | May be archived? | May trigger replay? | May write Geist/ISM? | Notes |
|---|---|---:|---:|---:|---|
| `MinimalSpineMicroMilestoneCandidate` | Local derived candidate only; not a protocol milestone record; source for future Micro builder. | no | no | no | It may be inspected or digested locally, but is not emitted by itself. |
| `MicroMilestone` / future `MicroMilestoneRecord` | Protocol-facing emitted record if constructed by an explicit deterministic builder. | yes, only through explicit Evidence/Archive append contract | no | no | Prompt 27 may build it but must not append it. |
| `MesoMilestone` / future `MesoMilestoneRecord` | Protocol-facing aggregate record over emitted micro records. | yes, only through explicit Evidence/Archive append contract | no | no | Later Prompt 29/30 scope. |
| `MacroMilestone` / future macro finalized record | Protocol-facing macro candidate/finalization record. `Finalized` means consolidation-level finalization only. | yes, only through explicit Evidence/Archive append contract | no | no | Does not mean Geist/ISM identity finalization, replay completion proof, or identity anchor production. |
| Replay/Sleep records | Replay token/event/readiness artifacts. | yes, only in Replay Scheduler scope | yes, only in Replay Scheduler scope | no | Out of scope until a replay prompt. |
| Geist/ISM anchor | Identity or ISM anchor material. | no in consolidation prompts | no | yes, only in future Geist/ISM authority prompt | Out of scope for Prompt 27 and this document. |
| `ExperienceRecord` / `ProofEnvelope` | Canonical evidence/archive payload and proof surface. | yes | no by itself | no | Authority remains with Evidence/Archive append/readback contracts. |

## 6. Evidence / Archive Boundary

Evidence/Archive remains the canonical append/readback proof surface.
Consolidation may construct deterministic payloads, but append is allowed only through an explicit append contract in a future prompt.
Pure builders must not hide append, replay, sleep, Geist/ISM, Gateway write, or capability-issuance side effects.
There must be no second event log.
`ArchiveMilestoneSink` needs a boundary wrapper or strict tests before any new Micro/Meso/Macro pipeline uses it.

| Future artifact | Append allowed? | Required contract before append | Authority risk |
|---|---:|---|---|
| Micro milestone record | yes, later | Explicit Evidence/Archive append/readback contract, payload schema, proof expectations, and no hidden replay. | Medium: existing `ArchiveMilestoneSink::emit_micro` is too broad for pure builder use. |
| Meso milestone record | yes, later | Aggregation contract from emitted micro records and explicit append/readback tests. | Medium: direct Minimal Spine source must not bypass micro records. |
| Macro milestone candidate/finalized record | yes, later | Macro candidate/finalization semantics, event naming, and proof contract that excludes Geist/ISM identity finalization. | High: `finalized` can overclaim identity/replay completion. |
| Replay token/event | no for Prompt 27 | Replay Scheduler prompt with budget, redaction, archive kind, and Sleep boundary. | High: existing replay archive append path exists. |
| Geist/ISM anchor | no | Future Geist/ISM authority prompt with identity semantics and archive authority. | High: must not be implied by Macro finalization. |

## 7. Out-of-Scope Boundaries

- No Replay Scheduler.
- No Sleep Cycle Coordinator.
- No Geist/ISM writes.
- No Identity finalization.
- No Gateway writes.
- No Capability issuance.
- No real compute dependency.
- No second event log.
- No Macro finalization implementation.
- No Micro/Meso/Macro pipeline activation.
- No Evidence/Archive authority change.
- No Minimal Spine v1.x freeze change.

## 8. Prompt 27 Acceptance Criteria

UCF Prompt 27 -- Deterministic MicroMilestone Builder from Minimal Spine Links must satisfy all of the following:

1. Implement or document a pure deterministic Micro builder only.
2. Accept `MinimalSpineMicroMilestoneCandidate` or canonical Minimal Spine links as input.
3. Return `ucf_protocol::v1::spec::MicroMilestone` or a clearly named candidate-to-record result type.
4. Produce stable canonical bytes and a stable digest/commitment under repeated runs.
5. Preserve `CandidateSetRecord` and `OutputRecord` digest references in a deterministic provenance convention or explicitly document any schema insufficiency.
6. Perform no Evidence/Archive append.
7. Trigger no replay and create no replay token/event.
8. Perform no Sleep state update.
9. Write no Geist/ISM state or identity anchor.
10. Perform no Macro finalization and publish no macro-finalized event.
11. Depend on no real-compute backend or Gateway write API.
12. Include tests for determinism, candidate-to-record boundary, no append/replay/Geist/ISM side effects, and canonical protocol bytes.
13. Keep Minimal Spine v1.x record authority unchanged.

## 9. Open Questions

- Are existing protocol milestone records sufficient for emitted Micro/Meso/Macro records, or do they need additive fields or wrapper records?
- Do we need a new emitted `MicroMilestoneRecord`, or can existing `MicroMilestone` be used with a deterministic provenance convention?
- How should candidate-to-record provenance be represented without overloading the human-readable `label` field?
- Where should the Prompt 27 builder live: `ucf-consolidation`, `ucf-protocol` helpers, or a narrower bridge module?
- How do we avoid `ArchiveMilestoneSink` side effects in a pure builder while still planning a future append contract?
- What exactly does Macro `finalized` mean without Geist/ISM identity finalization or Macro identity anchor production?
- Should protocol-facing Micro/Meso/Macro records gain version fields in an additive schema update, or should versioning live in a wrapper?
- What proof/readback payload shape will Evidence/Archive require for milestone append in later prompts?
