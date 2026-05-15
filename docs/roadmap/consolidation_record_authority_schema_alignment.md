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

## 12. Prompt 28 Completion Note

Prompt 28 is complete. The Minimal Spine MicroMilestone Evidence/Archive append contract is now explicit and readback-tested.

| Concern | Decision | Reason |
|---|---|---|
| Append authority | Existing `ucf-evidence::EvidenceStore` and `ucf-archive-store::ArchiveStore` APIs | Evidence/Archive remain the canonical append/readback surfaces; `ucf-consolidation` only constructs the deterministic payload and invokes those APIs when the explicit helper is called. |
| Consolidation role | Payload constructor / explicit caller only | The pure builder remains append-free; append behavior lives in `append_minimal_spine_micro_milestone` and is not called by the builder. |
| `ArchiveMilestoneSink` usage | Avoided for Prompt 28 | `ArchiveMilestoneSink` is broader than this contract because it can publish index events, record sleep-state derived records, and emit meso/macro paths. The Prompt 28 helper uses only narrow Evidence/Archive append/readback APIs. |
| Payload provenance location | Append payload/wrapper | Protocol `MicroMilestone` still only carries id, timestamp, and label. Minimal Spine provenance is preserved in `MinimalSpineMicroMilestoneAppendPayload`, not in the protocol schema. |
| Readback authority | Existing Evidence/Archive stores | The helper reads back the appended evidence envelope by id and the archive-store record by key, then returns deterministic digests. |
| Replay/Geist side effects | Forbidden | The helper has no Replay Scheduler, Sleep Cycle, Geist, ISM, meso, macro, identity-finalization, capability, gateway, or real-compute integration. |

Implemented/tested surface:

- `MinimalSpineMicroMilestoneAppendPayload` preserves `build_output_digest`, `candidate_digest`, `micro_milestone_digest`, `input_digest`, `candidate_set_record_digest`, `output_record_digest`, source `EvidenceId`, `archive_output_key`, `archive_output_event_digest`, and source marker.
- `MinimalSpineMicroMilestoneAppendResult` returns the payload digest, build-output digest, micro milestone digest, appended evidence id, archive-store key, archive-record digest, and deterministic readback digest.
- `append_minimal_spine_micro_milestone` is the only new append helper and must be called explicitly.
- `RecordKind::Other(28)` is used as the narrow archive-store extension kind because archive-store has no canonical MicroMilestone `RecordKind`; no archive-store schema change was made.
- Tests live in `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_micro_append.rs`.

| Prompt | Status | Implemented surface | Boundary result | Recommended next prompt |
|---:|---|---|---|---|
| 28 | complete | Explicit Minimal Spine MicroMilestone append payload/result/helper plus readback tests | Micro append is opt-in, provenance-preserving, deterministic, Evidence/Archive-authoritative, and no Replay/Sleep/Geist/ISM/Meso/Macro path is activated | Prompt 29 — Deterministic MesoMilestone Aggregation |

## 13. Prompt 29 Deterministic MesoMilestone Aggregation Status

Prompt 29 is implemented as a pure, deterministic, append-free MesoMilestone aggregator in `ucf-consolidation`.

| Item | Status |
|---|---|
| Chosen input option | Option C — support micro build outputs and explicit micro append payload values |
| Primary builder API | `build_meso_milestone_from_minimal_spine_micro_payloads(&[MinimalSpineMicroMilestoneAppendPayload]) -> Result<MinimalSpineMesoMilestoneBuildOutput, ConsolidationError>` |
| Build-output convenience API | `build_meso_milestone_from_minimal_spine_micro_build_outputs(&[MinimalSpineMicroMilestoneBuildOutput]) -> Result<MinimalSpineMesoMilestoneBuildOutput, ConsolidationError>` |
| Builder output | `MinimalSpineMesoMilestoneBuildOutput` containing a protocol-compatible `ucf_types::v1::spec::MesoMilestone` plus micro payload digests, micro milestone digests, aggregation digest, count, and source marker |
| Ordering semantics | Inputs are normalized by micro payload digest, then micro milestone digest, then micro milestone id; reversed input produces the same output/digest. |
| Duplicate semantics | Duplicate micro payload digest or duplicate micro milestone digest is rejected. |
| Empty input semantics | Empty input is rejected. |
| Deterministic bytes/digest | `MinimalSpineMesoMilestoneBuildOutput::deterministic_bytes` and `MinimalSpineMesoMilestoneBuildOutput::digest` |
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_meso_builder.rs` |
| Append behavior | none |
| Replay/Sleep/Geist/ISM behavior | none |
| Macro behavior | none |
| Minimal Spine v1.x changes | none |

### Prompt 29 schema gap

The current protocol `MesoMilestone` surface carries milestone id, achieved-at timestamp, label, and micro milestone ids. It does not fully carry Minimal Spine micro provenance by itself: micro append payload digests, protocol micro milestone digests, aggregation digest, and source metadata remain outside the protocol meso record. Prompt 29 therefore uses the clearly named `MinimalSpineMesoMilestoneBuildOutput` wrapper rather than overloading the protocol `MesoMilestone` fields or changing protocol schema.

The Prompt 29 aggregator is Meso-only and append-free. It does not call `append_minimal_spine_micro_milestone`, `ArchiveMilestoneSink`, Evidence/Archive stores, Replay, Sleep, Geist, ISM, Macro builders, Gateway write APIs, capability issuance, or real-compute surfaces.

## 14. Prompt 30 Completion Note

Prompt 30 is complete. The roadmap now has a narrow Evidence/Archive append/readback contract for Minimal Spine MesoMilestone build outputs.

| Boundary | Prompt 30 result |
|---|---|
| Explicit append only | Append occurs only through `append_minimal_spine_meso_milestone`; the pure Minimal Spine meso builder does not append. |
| Builder purity | `build_meso_milestone_from_minimal_spine_micro_payloads` and `build_meso_milestone_from_minimal_spine_micro_build_outputs` remain deterministic and append-free. |
| Provenance | Full meso aggregation provenance remains in `MinimalSpineMesoMilestoneAppendPayload`: meso build-output digest, protocol meso milestone digest, aggregation digest, micro payload digests, micro milestone digests, micro count, and source marker. Protocol `MesoMilestone` is unchanged and still does not carry all provenance by itself. |
| Evidence/Archive authority | Existing `ucf-evidence::EvidenceStore` and `ucf-archive-store::ArchiveStore` APIs remain the append/readback authority. |
| Archive kind | `ucf-archive-store::RecordKind::Other(30)` is used as the Minimal Spine meso append extension kind because archive-store has no canonical `MesoMilestone` variant. |
| `ArchiveMilestoneSink` | Not used for Prompt 30 because it is broader than this meso-only contract and can couple to index publication, sleep-state derived-record tracking, and macro emission/finalization paths. |
| Replay/Sleep/Geist/ISM | Not integrated or triggered. |
| Macro | Not built, aggregated, emitted, or finalized. |
| Minimal Spine v1.x | Unchanged. |
| Protocol schema | Unchanged; the schema/provenance gap is documented as append-payload provenance. |
| Second event log | Not introduced; consolidation constructs deterministic payloads and delegates append/readback to Evidence/Archive stores. |
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_meso_append.rs` |

Recommended next prompt: **UCF Prompt 31 — MacroMilestone Candidate Builder**.

## 15. Prompt 31 MacroMilestone Candidate Builder Status

Prompt 31 is implemented as a pure, deterministic, candidate-only MacroMilestone builder in `ucf-consolidation`.

| Item | Status |
|---|---|
| Chosen input option | Option C — support explicit meso append payload values and meso build outputs |
| Primary builder API | `build_macro_milestone_candidate_from_minimal_spine_meso_payloads(&[MinimalSpineMesoMilestoneAppendPayload]) -> Result<MinimalSpineMacroMilestoneCandidate, ConsolidationError>` |
| Build-output convenience API | `build_macro_milestone_candidate_from_minimal_spine_meso_build_outputs(&[MinimalSpineMesoMilestoneBuildOutput]) -> Result<MinimalSpineMacroMilestoneCandidate, ConsolidationError>` |
| Builder output | `MinimalSpineMacroMilestoneCandidate` containing a protocol-compatible `ucf_types::v1::spec::MacroMilestone` plus meso payload digests, meso build-output digests, meso milestone digests, meso aggregation digests, macro aggregation digest, macro candidate digest, count, source marker, and explicit boundary booleans |
| Candidate-only flags | `finalized == false` and `identity_anchor == false` |
| Ordering semantics | Inputs are normalized by meso payload digest, then meso milestone digest, then meso aggregation digest, then meso milestone id; reversed input produces the same candidate/digest. |
| Duplicate semantics | Duplicate meso payload digest or duplicate meso milestone digest is rejected. |
| Empty input semantics | Empty input is rejected. |
| Deterministic bytes/digest | `MinimalSpineMacroMilestoneCandidate::deterministic_bytes` and `MinimalSpineMacroMilestoneCandidate::digest` |
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_macro_candidate.rs` |
| Append behavior | none |
| Replay/Sleep/Geist/ISM behavior | none |
| Identity-finalization behavior | none |
| Macro finalization behavior | none |
| Minimal Spine v1.x changes | none |

### Prompt 31 schema gap

The current protocol `MacroMilestone` surface carries milestone id, achieved-at timestamp, label, and meso milestone ids. It does not fully carry Minimal Spine meso/micro provenance by itself: meso append payload digests, meso build-output digests, protocol meso milestone digests, meso aggregation digests, macro aggregation digest, candidate-only status, and identity-anchor status remain outside the protocol macro record. Prompt 31 therefore uses `MinimalSpineMacroMilestoneCandidate` as the honest wrapper surface and does not change protocol schemas.

### Prompt 31 boundary statement

The MacroMilestone candidate builder is not finalization. It does not append Evidence or Archive records, does not use `ArchiveMilestoneSink`, does not publish `MacroMilestoneFinalized`, does not mark an identity anchor, does not trigger Replay/Sleep/Geist/ISM, does not issue capabilities, and does not change Minimal Spine v1.x authority. Future Prompt 32 should define the macro finalization boundary without Geist/ISM.

| Prompt | Status | Implemented surface | Boundary result | Recommended next prompt |
|---:|---|---|---|---|
| 31 | complete | `MinimalSpineMacroMilestoneCandidate` plus pure builders from meso payloads/build outputs | Macro candidate only, deterministic, provenance-preserving, append-free, replay/sleep/geist/ISM-free, no identity anchor, no Minimal Spine v1.x changes | Prompt 32 — MacroMilestone Finalization Boundary Without Geist/ISM |

## 16. Prompt 32 MacroMilestone Finalization Boundary Status

Prompt 32 is implemented as a local, deterministic, consolidation-level finalization boundary record. It deliberately does not use the existing broad `ArchiveMilestoneSink::emit_macro` / `MacroMilestoneFinalized` publication path.

| Item | Status |
|---|---|
| Chosen option | Option B — local finalization boundary record |
| Boundary API | `MinimalSpineMacroConsolidationFinalization::from_candidate(&MinimalSpineMacroMilestoneCandidate) -> Result<MinimalSpineMacroConsolidationFinalization, ConsolidationError>` |
| Boundary meaning | A deterministic local decision that a Macro candidate is structurally complete for the consolidation pipeline. |
| Deterministic bytes/digest | `MinimalSpineMacroConsolidationFinalization::deterministic_bytes` and `MinimalSpineMacroConsolidationFinalization::digest` |
| Required input state | Candidate must be unfinalized, non-identity, non-empty, and must carry valid non-zero candidate/milestone/aggregation digests. |
| Candidate mutation | none |
| Protocol schema changes | none |
| Evidence/Archive append | none |
| `ArchiveMilestoneSink` / broad macro-finalized publication | not used |
| `MacroMilestoneFinalized` event | not published |
| Replay/Sleep/Geist/ISM | not integrated, called, ingested, or triggered |
| Identity anchor | not created; `identity_anchor == false` |
| Gateway visibility | not enabled; `gateway_visible == false` |
| Capability issuance / real compute | none |
| Minimal Spine v1.x changes | none |
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_macro_finalization_boundary.rs` |

### Prompt 32 finalization semantics

| Term | Allowed meaning now | Explicitly not allowed |
|---|---|---|
| Macro candidate | Deterministic, append-free aggregation wrapper around a protocol-compatible MacroMilestone plus Minimal Spine meso provenance. | Finalized event, identity anchor, Replay completion proof, Geist/ISM input, Gateway-visible record, or Evidence/Archive append by itself. |
| Macro consolidation finalization | Local, deterministic completeness boundary for the consolidation pipeline; represented by `MinimalSpineMacroConsolidationFinalization` with `consolidation_finalized == true`. | Identity finalization, ISM anchor, Geist ingestion, Replay completion, Evidence/Archive authority update, Gateway write, capability issuance, real-compute activation, or `MacroMilestoneFinalized` publication. |
| Identity anchor | Out of scope. | Must not be inferred from a macro candidate, protocol MacroMilestone, or consolidation finalization boundary record. |
| Geist/ISM ingestion | Out of scope. | No call to `GeistKernel::ingest_macro`, no ISM write, no identity-finalization claim. |
| Replay completion | Out of scope. | No Replay Scheduler integration, replay token, replay completion proof, or sleep-cycle coupling. |
| Evidence/Archive authority | Separate explicit append/readback contract only. | The boundary record does not append, change archive/evidence authority, or introduce a second event log. |

### Prompt 32 boundary statement

`MinimalSpineMacroConsolidationFinalization` is consolidation-level only. It records `consolidation_finalized == true` while explicitly keeping `identity_anchor`, `geist_ingested`, `replay_completed`, `evidence_archive_appended`, and `gateway_visible` false. Existing broader APIs that append, publish index events, or update sleep state remain out of scope until a future audit explicitly narrows them.

| Prompt | Status | Implemented surface | Boundary result | Recommended next prompt |
|---:|---|---|---|---|
| 32 | complete | `MinimalSpineMacroConsolidationFinalization` local boundary record and deterministic boundary tests | Consolidation-level macro finalization only; no identity anchor, no Geist/ISM ingestion, no Replay completion, no Evidence/Archive append, no Gateway write, no broad macro-finalized publish path | Prompt 33 — Consolidation Pipeline E2E Determinism |

## 17. Prompt 33 Consolidation Pipeline E2E Determinism Status

Prompt 33 is implemented as an integration-test-only deterministic Micro→Meso→Macro→local-finalization test path. It does not introduce a new pipeline runtime, scheduler, sink, event-log authority, or protocol schema.

| Boundary | Prompt 33 result |
|---|---|
| Test path | `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_consolidation_pipeline_e2e.rs` |
| E2E sequence | Micro candidates → micro build outputs → explicit micro append/readback payloads → meso build output → explicit meso append/readback payload → macro candidate → local consolidation finalization boundary. |
| Determinism | Two runs with identical inputs and fresh `InMemoryEvidenceStore`, `InMemoryArchiveStore`, and `ArchiveAppender` instances compare equal micro build digests, micro payload/readback digests, meso build digest, meso payload/readback digest, macro candidate digest, and finalization digest. |
| Provenance continuity | Micro payload and milestone digests are asserted in meso output/payload; meso payload, build-output, milestone, and aggregation digests are asserted in the macro candidate; the macro candidate digest is asserted in the finalization boundary. |
| Explicit append/readback | Append/readback remains explicit for Micro and Meso only through `append_minimal_spine_micro_milestone` and `append_minimal_spine_meso_milestone`. |
| Builder purity | Micro, Meso, Macro candidate, and finalization builders remain append-free; builder-only tests assert empty Evidence/Archive stores. |
| Replay/Sleep/Geist/ISM | Still excluded; no scheduler, sleep-cycle integration, Geist ingestion, or ISM write is activated. |
| Gateway/capability/real compute | Still excluded; no Gateway write API, capability issuance, or real-compute activation is added. |
| Identity | No identity finalization and no identity anchor; macro candidate keeps `identity_anchor == false` and the local boundary keeps `identity_anchor == false`. |
| Evidence/Archive authority | Existing Evidence/Archive append/readback APIs remain the only append authority for Micro/Meso records. |
| `ArchiveMilestoneSink` / macro finalized event | Not used; no `MacroMilestoneFinalized` event is produced. |
| Minimal Spine v1.x | Unchanged. |
| Protocol schema | Unchanged. |
| Duplicate/invalid handling | Duplicate Micro payloads are rejected before Meso aggregation; duplicate Meso payloads are rejected before Macro candidate aggregation; invalid zero-link Micro candidates are rejected by the Micro builder path. |

Recommended next prompt: **UCF Prompt 34 — Consolidation Docs Overclaim Guard**.
