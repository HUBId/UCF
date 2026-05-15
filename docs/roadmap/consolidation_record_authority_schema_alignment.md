# UCF Consolidation Record Authority and Schema Alignment

## 0. Purpose

- Establishes Micro/Meso/Macro record authority before implementation.
- Aligns candidate-vs-emitted semantics for existing consolidation, protocol, evidence, archive, replay, and Geist surfaces.
- Not a pipeline implementation.
- Not Replay/Geist/ISM readiness.
- Does not change Minimal Spine v1.x, Evidence/Archive authority, or any historical document.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `4d30820842a0107c1d0e3ff2494e05e3a32ca4eb` |
| HEAD short | `4d308208` |
| Dirty state at audit start | clean |
| Workspace package count | 192 |
| Consolidation roadmap present | yes |
| Micro hook test present | yes |
| `ucf-consolidation` present | yes |
| `ucf-protocol` present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -20`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and the requested file/directory presence checks.

Source links:

- [`docs/roadmap/full_consolidation_roadmap_boundary_audit.md`](full_consolidation_roadmap_boundary_audit.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/minimal_ucf_spine_v1.md`](../minimal_ucf_spine_v1.md)

## 2. Record Inventory

| Record / Type | Path | Module | Current role | Candidate or emitted? | Schema authority? | Notes |
|---|---|---|---|---|---|---|
| `MinimalSpineMicroMilestoneCandidate` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | `ucf_consolidation` | local-candidate | candidate-only | `ucf-consolidation` | Derived from Minimal Spine Evidence/Archive/Protocol links; not a canonical event-log record and not archived/emitted by itself. |
| `ucf_protocol::v1::spec::MicroMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing micro record with stable canonical encoding; current fields are id, timestamp, and label. |
| `ucf_protocol::v1::spec::MesoMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing aggregated meso record; references micro milestone ids. |
| `ucf_protocol::v1::spec::MacroMilestone` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Protocol-facing macro record; references meso milestone ids. The protocol spec form has no identity-anchor or replay-completion field. |
| `ucf_protocol::v1::MacroMilestone` | `protocol/crates/ucf-protocol/src/lib.rs` | `ucf_protocol::v1` | historical | unclear | `ucf-protocol` | Legacy/non-`spec` macro message with `MacroMilestoneState::Finalized`; do not use as the new pipeline authority without a later explicit migration decision. |
| `ucf_protocol::v1::MacroMilestoneAppend` | `protocol/crates/ucf-protocol/src/lib.rs` | `ucf_protocol::v1` | archive-payload | unclear | `ucf-protocol` | Append wrapper around the legacy macro message; broad-risky for Prompt 27 because it carries finalized naming. |
| `ucf_types::consolidation::MicroMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | not-applicable | `ucf-types` | Internal memory/replay graph node keyed by digests; not protocol-facing. |
| `ucf_types::consolidation::MesoMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | not-applicable | `ucf-types` | Internal memory/replay graph node keyed by micro commits; not protocol-facing. |
| `ucf_types::consolidation::MacroMilestone` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | builder-output | not-applicable | `ucf-types` | Internal memory/replay graph node keyed by meso commits; not protocol-facing and not identity finalization. |
| `ReplayToken` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | replay-token | emitted-record | `ucf-types` | Replay target token, explicitly content-free; out of scope until the Replay Scheduler line. |
| `ReplayScheduled` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | replay-token | emitted-record | `ucf-types` | Replay scheduling event derived by `ReplayCascade`; out of scope for Prompt 27. |
| `ReplayApplied` | `core/crates/ucf-types/src/lib.rs` | `ucf_types::consolidation` | replay-token | emitted-record | `ucf-types` | Replay effect event; out of scope for Prompt 27. |
| `ExperienceRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs` | `ucf_protocol::v1::spec`; `ucf_archive` | archive-payload | emitted-record | `ucf-protocol` | Canonical protocol payload container appended by Evidence/Archive; consolidation may only append through explicit append contracts. |
| `CandidateSetRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Minimal Spine commitment source; digest may be referenced by future micro builder provenance. |
| `OutputRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md` | `ucf_protocol::v1::spec` | protocol-record | emitted-record | `ucf-protocol` | Minimal Spine output commitment source; carries optional `evidence_id`. |
| `ProofEnvelope` | `protocol/crates/ucf-protocol/src/lib.rs`; `core/crates/ucf-evidence/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs` | `ucf_protocol::v1::spec`; `ucf_evidence`; `ucf_archive` | archive-payload | emitted-record | `ucf-protocol` | Proof wrapper/readback surface; remains Evidence/Archive authority, not consolidation authority. |
| `ArchiveRecord` / `RecordKind::{ReplayToken, ReplayApplied, IsmAnchor, OutputEvent}` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | `ucf_archive_store` | archive-payload | emitted-record | `ucf-archive-store` | Low-level archive-store records; not a second milestone event-log authority for Prompt 27. |
| `MacroMilestoneFinalized` | `domains/index/crates/ucf-vector-index` | `ucf_vector_index` | archive-payload | emitted-record | docs/code outside protocol | Index event naming only; must not be treated as Geist/ISM/identity finalization. |
| `GeistKernel::ingest_macro` derived record | `domains/geist/crates/ucf-geist/src/lib.rs` | `ucf_geist` | archive-payload | emitted-record | `ucf-geist` | Consumes protocol macro milestones and may upsert ISM anchors; out of scope for Prompt 27. |

## 3. Protocol Schema Alignment

| Protocol record | Exists? | Versioned? | Canonical encoding tested? | Minimal Spine link capable? | Overclaim risk | Gap |
|---|---:|---:|---:|---:|---|---|
| `ExperienceRecord` | yes | no explicit version field | yes | yes, as Evidence/Archive payload carrier | Low; generic payload container can obscure payload type without contract. | Future milestone append contract should state payload type and digest domain. |
| `CandidateSetRecord` | yes | yes | yes | yes, digestable directly as Minimal Spine source | Low; commitment-only and non-execution semantics are documented. | Candidate-to-micro provenance mapping is not yet encoded in a milestone record. |
| `OutputRecord` | yes | yes | yes | yes, via `candidate_set_digest`, `output_digest`, and optional `evidence_id` | Low; commitment/status only. | Future builder must define whether it stores output digest in label, id, proof envelope, or a new schema. |
| `MicroMilestone` | yes | no explicit version field | implementation exists; protocol test coverage is generic, not milestone-specific | indirectly; can encode a deterministic id/label from Minimal Spine links | Medium; current fields do not explicitly carry CandidateSet/Output/Evidence provenance. | Prompt 27 must decide whether existing fields are sufficient or whether a new emitted micro schema/version is needed. |
| `MesoMilestone` | yes | no explicit version field | implementation exists; protocol test coverage is generic, not milestone-specific | indirectly, through micro ids only | Low/medium; aggregation semantics are not specified beyond id references. | Prompt 29/30 must define aggregation and provenance. |
| `MacroMilestone` | yes | no explicit version field | implementation exists; protocol test coverage is generic, not milestone-specific | indirectly, through meso ids only | Medium; `macro` naming may be overread as identity anchor. | Prompt 31/32 must define consolidation-level finalization without Geist/ISM claims. |
| `ProofEnvelope` | yes | no explicit version field | implementation exists; protocol tests cover canonical message behavior for related records | yes, can wrap canonical bytes/digests | Low; proof metadata must be verified locally. | Future append contract must define proof envelope requirements. |

Alignment answers:

- Protocol-facing Micro/Meso/Macro records exist as `ucf_protocol::v1::spec::{MicroMilestone,MesoMilestone,MacroMilestone}` and are re-exported through `ucf-types::v1`.
- The protocol milestone records do not currently have explicit `version` fields; `CandidateSetRecord` and `OutputRecord` do.
- Deterministic canonical encoders exist for protocol milestone records. Existing canonical tests cover the canonical framework, `ExperienceRecord`, `CandidateSetRecord`, and `OutputRecord`; milestone-specific canonical roundtrip tests are not yet explicit.
- Existing milestone records can reference Minimal Spine material indirectly through deterministic ids/labels or proof/payload conventions, but do not have dedicated `CandidateSetRecord`, `OutputRecord`, `EvidenceId`, or archive-key fields.
- `CandidateSetRecord` and `OutputRecord` digests are directly usable as builder inputs and indirectly usable as milestone provenance commitments.
- The `spec` milestone records do not contain Geist, ISM, identity-anchor, replay-completion, or sleep-cycle fields.
- `protocol/crates/ucf-protocol/spec/messages_v1.md` mentions optional milestone `Digest commitment` fields that are not present in the current `spec` Rust structs or `spec/v1.md` milestone tables; this is a schema-documentation alignment gap to resolve before relying on those commitment fields.

## 4. Consolidation API Authority Audit

| API / Function | Path | Pure? | Side effects? | Uses Archive/Evidence? | Uses Replay? | Safe for Prompt 27? | Reason |
|---|---|---:|---:|---:|---:|---:|---|
| `MinimalSpineMicroMilestoneCandidate::from_minimal_spine_links` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | references ids/digests only | no | yes | Constructs local candidate deterministically without append or scheduler behavior. |
| `MinimalSpineMicroMilestoneCandidate::deterministic_bytes` | same | yes | no | no | no | yes | Local deterministic serialization for candidate digesting. |
| `MinimalSpineMicroMilestoneCandidate::digest` | same | yes | no | no | no | yes | Domain-separated candidate digest only. |
| `MinimalSpineMicroMilestoneCandidate::validate_links_nonzero` | same | yes | no | no | no | yes | Boundary validation only. |
| `build_micro` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | yes | no | consumes `ExperienceRecord` values | no | caution | Pure protocol milestone builder exists, but it builds from broad `ExperienceRecord` windows rather than Minimal Spine candidate links. |
| `build_meso` | same | yes | no | no | no | no | Later meso aggregation line; not needed for Prompt 27. |
| `build_macro` | same | yes | no | no | no | no | Later macro line; avoid macro finalization claims in Prompt 27. |
| `MilestoneSink::{emit_micro,emit_meso,emit_macro}` | same | no | yes | appends via sink implementations | possible via downstream integrations | no | Trait is append-oriented and too broad for pure builder prompt. |
| `ArchiveMilestoneSink::emit_micro` | same | no | yes | appends `ExperienceRecord`, may publish index event, may update sleep state | no | no | Requires explicit append contract and side-effect tests before use. |
| `ArchiveMilestoneSink::emit_meso` | same | no | yes | appends `ExperienceRecord`, may publish index event, may update sleep state | no | no | Later meso append contract only. |
| `ArchiveMilestoneSink::emit_macro` | same | no | yes | appends `ExperienceRecord`, publishes macro-finalized index event, may update sleep state | no | no | Macro finalized event is overclaim-prone and out of scope. |
| `ConsolidationKernel::run_one_cycle` | same | no | yes | reads source and appends micro/meso/macro records | no direct replay, but updates sleep state via sink | no | Broad pipeline behavior; not safe as Prompt 27 authority. |
| `ReplayCascade::schedule` | same | yes | no | no | yes | no | Pure replay selection exists, but Replay Scheduler is out of scope. |
| `ConsolidationKernel::run_sleep_replay` | same | no | yes | appends replay records to archive store | yes | no | Sleep/replay integration is explicitly deferred. |
| `build_memory_micro` / `build_memory_meso` / `build_memory_macro` | same | yes | no | consumes protocol `ExperienceRecord` digests | used by replay graph | no | Internal memory graph for replay selection, not protocol schema authority. |
| `derived_record_for_micro` / `derived_record_for_meso` / `derived_record_for_macro` | same | yes | no | constructs archive payload wrappers | no | no | Private helpers are tied to append sink and need an append/readback contract before promotion. |
| `GeistKernel::ingest_macro` | `domains/geist/crates/ucf-geist/src/lib.rs` | no | yes | appends derived record | uses replay effects elsewhere | no | Consumes macro milestones and can upsert ISM anchors; out of scope. |

## 5. Candidate vs Emitted Semantics

| Concept | Semantics | May be archived? | May trigger replay? | May write Geist/ISM? | Notes |
|---|---|---:|---:|---:|---|
| `MinimalSpineMicroMilestoneCandidate` | Local derived candidate only; not protocol milestone record; source for future deterministic MicroMilestone builder. | no | no | no | Candidate digest is local provenance, not an event-log record. |
| `MicroMilestone` / future `MicroMilestoneRecord` | Protocol-facing emitted record if a future prompt maps candidate links into a protocol record. | yes, only via explicit Evidence/Archive append contract | no | no | Prompt 27 may build it purely; append/readback belongs to a later prompt. |
| `MesoMilestone` / future `MesoMilestoneRecord` | Protocol-facing aggregated record from emitted micro milestones. | yes, only via explicit Evidence/Archive append contract | no | no | No direct Minimal Spine source except through micro aggregation. |
| `MacroMilestone` / future macro candidate/finalized record | Protocol-facing macro aggregation/finalization surface for later prompts. | yes, only via explicit Evidence/Archive append contract | no | no | `finalized` means consolidation-level finalization only. |
| `MacroMilestoneState::Finalized` and `MacroMilestoneFinalized` events | Legacy/event naming that may denote consolidation-level macro closure only if explicitly used later. | only under future macro contract | no | no | Does not mean Geist/ISM identity finalization, replay completion proof, or macro identity-anchor production. |
| `ReplayToken`, `ReplayScheduled`, `ReplayApplied` | Replay scheduler artifacts. | only in Replay Scheduler scope | yes, in Replay Scheduler scope only | no | Out of scope until replay prompt. |
| `Geist` / `ISM` anchors | Identity/self-model artifacts owned by Geist/ISM scope. | only under Geist/ISM contract | no | yes, in Geist/ISM scope only | Not produced by consolidation record builders. |

## 6. Evidence / Archive Boundary

Evidence/Archive remains the canonical append/readback proof surface. Consolidation may construct deterministic payloads, but pure builders must not append, publish events, update sleep state, trigger replay, write Geist/ISM, or create a second event log. Any future append must go through an explicit Evidence/Archive append contract with readback tests. `ArchiveMilestoneSink` needs a boundary wrapper or strict tests before use in the new Micro/Meso/Macro pipeline.

| Future artifact | Append allowed? | Required contract before append | Authority risk |
|---|---:|---|---|
| Micro milestone record | yes, later | Explicit Evidence/Archive append/readback contract, payload type/digest domain, id stability, no hidden replay/sleep/geist side effects. | Medium if `ArchiveMilestoneSink` is reused without narrowing. |
| Meso milestone record | yes, later | Aggregation contract from emitted micro records plus Evidence/Archive append/readback tests. | Medium; aggregation provenance could be under-specified. |
| Macro milestone candidate/finalized record | yes, later | Macro candidate/finalization contract defining consolidation-level finality, payload schema, event naming, and no Geist/ISM identity claim. | High because `finalized` can overclaim identity/replay completion. |
| Replay token/event | no for Prompt 27 | Replay Scheduler prompt with budget, redaction, archive-store, and sleep boundaries. | High if emitted as side effect of milestone building. |
| Geist/ISM anchor | no for Prompt 27 | Geist/ISM prompt with gate, consistency report, ISM upsert, and identity-finalization semantics. | Critical if macro finalization is confused with identity anchor production. |

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
- No Evidence/Archive authority change.
- No Minimal Spine v1.x change.

## 8. Prompt 27 Acceptance Criteria

UCF Prompt 27 — Deterministic MicroMilestone Builder from Minimal Spine Links must satisfy all of the following:

1. Implement or document a pure deterministic builder only.
2. Accept `MinimalSpineMicroMilestoneCandidate` or equivalent canonical Minimal Spine links as input.
3. Output a protocol `MicroMilestone` or a clearly named candidate-to-record result whose authority is explicit.
4. Produce stable canonical bytes and a stable digest/commitment across repeated runs.
5. Preserve `ucf-protocol` / `ucf-types::v1::spec` as protocol-facing schema authority unless a new schema/version is explicitly proposed and tested.
6. Perform no archive append and write no Evidence/Archive records.
7. Trigger no replay and create no replay tokens/events.
8. Write no Geist/ISM anchors and perform no identity finalization.
9. Perform no macro finalization and publish no macro-finalized events.
10. Depend on no real compute, Gateway write API, capability issuance, or sleep-cycle coordinator.
11. Include tests for determinism, canonical bytes/digest stability, candidate-to-record provenance, and negative boundary assertions for archive/replay/geist/macro side effects.
12. If existing `MicroMilestone` fields are insufficient, stop at a documented schema decision rather than overloading fields silently.

## 9. Open Questions

- Are existing protocol milestone records sufficient?
- Do we need a new emitted `MicroMilestoneRecord`, or can existing `ucf_protocol::v1::spec::MicroMilestone` be used?
- How should candidate-to-record provenance be represented without overloading `label`?
- Where should the builder live: `ucf-consolidation`, `ucf-protocol`, or a small bridging module?
- How do we avoid `ArchiveMilestoneSink` side effects in a pure builder while still preparing for a later append/readback contract?
- What does Macro finalized mean without Geist/ISM?
- Should `protocol/crates/ucf-protocol/spec/messages_v1.md` commitment-field text be reconciled with the Rust `spec` structs and `spec/v1.md` before Prompt 27 or as part of a later schema prompt?

## 11. Prompt 27 Deterministic MicroMilestone Builder Status

Prompt 27 is implemented as a pure, append-free builder in `ucf-consolidation`.

| Item | Status |
|---|---|
| Chosen option | Option B — builder output wrapper |
| Builder API | `build_micro_milestone_from_minimal_spine_candidate(&MinimalSpineMicroMilestoneCandidate) -> Result<MinimalSpineMicroMilestoneBuildOutput, ConsolidationError>` |
| Builder output | `MinimalSpineMicroMilestoneBuildOutput` containing a protocol-compatible `ucf_types::v1::spec::MicroMilestone` plus explicit Minimal Spine provenance digests/IDs |
| Deterministic bytes/digest | `MinimalSpineMicroMilestoneBuildOutput::deterministic_bytes` and `MinimalSpineMicroMilestoneBuildOutput::digest` |
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_micro_builder.rs` |
| Append behavior | none |
| Replay/Sleep/Geist/ISM behavior | none |
| Meso/Macro behavior | none |
| Minimal Spine v1.x changes | none |

### Schema gap recorded for Prompt 28 / schema follow-up

The current protocol `MicroMilestone` surface carries only milestone id, achieved-at timestamp, and label in code. It does not fully carry Minimal Spine provenance by itself: candidate digest, input digest, `CandidateSetRecord` digest, `OutputRecord` digest, `EvidenceId`, archive output key, and archive output event digest remain outside the protocol micro record. Prompt 27 therefore does not overclaim that the protocol `MicroMilestone` alone is the full Minimal Spine provenance container.

Prompt 28 should decide whether the provenance remains in an append payload/evidence wrapper, becomes a companion record, or requires a minimal protocol schema follow-up. Until that decision, the builder output wrapper is the honest deterministic handoff surface.
