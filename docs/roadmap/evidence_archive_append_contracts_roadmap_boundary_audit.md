# UCF Evidence/Archive Append Contracts Roadmap and Boundary Audit

## 0. Purpose

- This document is a roadmap and boundary audit only.
- No append implementation is introduced here.
- No Replay, Sleep, Geist, ISM, runtime, identity, Gateway, capability, real-compute, or production semantics are introduced here.
- No Evidence/Archive authority is changed, no second event log is created, and Minimal Spine v1.x remains unchanged.
- Future append work is limited to explicit audit/provenance persistence plus deterministic readback tests unless a later prompt changes policy intentionally.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `c38a614a61da27150ac2389fcb2143c7eb22a124` |
| HEAD short | `c38a614a` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Post-Geist selection present | yes |
| Geist/ISM closure present | yes |
| Sleep closure present | yes |
| Replay closure present | yes |
| `ucf-evidence` present | yes |
| `ucf-archive` present | yes |
| `ucf-archive-store` present | yes |
| Geist/ISM E2E present | yes |
| Sleep E2E present | yes |
| Replay E2E present | yes |

Baseline links:

- [`docs/roadmap/post_geist_roadmap_selection.md`](post_geist_roadmap_selection.md)
- [`docs/roadmap/geist_ism_closure.md`](geist_ism_closure.md)
- [`docs/roadmap/sleep_closure.md`](sleep_closure.md)
- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

Baseline commands used for this audit: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -30`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, required file-presence checks, required crate-presence checks, the mandatory source/document reads, and the requested broad `rg` Evidence/Archive inventory query.

## 2. Evidence / Archive Code Inventory

| Concern | Existing API/type | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|---|
| Evidence envelope authority | `EvidenceEnvelope` | `core/crates/ucf-evidence/src/lib.rs` | Carries `EvidenceId`, optional proof envelope, optional fold proof, logical time, and wall time. | functional-prototype | Low if used as explicit append authority; medium if callers infer domain/runtime meaning from presence alone. |
| Evidence store API | `EvidenceStore::{append,get,len,is_empty}` | `core/crates/ucf-evidence/src/lib.rs` | Append returns the evidence id; `get` is a default no-op unless implemented; in-memory implementation supports append/get/list. | functional-prototype | Medium: default `get` is not readback proof unless the concrete store implements it. |
| Evidence append log API | `AppendLog::{append_bytes,read_at}` | `core/crates/ucf-evidence/src/lib.rs`; `core/crates/ucf-evidence/src/file_store.rs` | Byte append/read abstraction with hash-corruption checks in the file-store path. | functional-prototype | Medium: raw bytes are broad and need typed contracts to avoid schema drift. |
| File evidence store | `FileEvidenceStore::open`, `FileEvidenceStore` implementation of `EvidenceStore` | `core/crates/ucf-evidence/src/file_store.rs` | File-backed log plus manifest for evidence envelopes. | functional-prototype | Medium: good persistence candidate, but future prompt must prove typed readback for each append contract. |
| Experience archive appender | `ExperienceAppender::{append_with_proof,append}` | `domains/archive/crates/ucf-archive/src/lib.rs` | Appends `ExperienceRecord` into an evidence store, optionally with proof. | functional-prototype | Medium: broad `ExperienceRecord` surface can hide semantics unless wrappers are explicit. |
| In-memory evidence archive | `InMemoryArchive::{append_with_proof,append_and_fold,list}` | `domains/archive/crates/ucf-archive/src/lib.rs` | In-memory evidence archive with fold state. | functional-prototype | Medium: useful for tests, not a production-retention claim. |
| File evidence archive | `FileArchive::{open,append_and_fold}` | `domains/archive/crates/ucf-archive/src/lib.rs` | File evidence archive backed by evidence log/manifest and fold snapshot. | functional-prototype | Medium: persistence authority exists, but future append contracts must remain explicit. |
| Generic archive stores | `RecordStore`, `SnapshotStore`, `ArchiveStore` wrapper | `domains/archive/crates/ucf-archive/src/store.rs` | Key/value record and snapshot stores with in-memory/file/firewood backend selection. | partial | Medium: broad storage surface; should not become a second event log for bounded artifacts. |
| Archive record authority | `ucf_archive_store::ArchiveStore::{append,get,iter_kind,root_commit}` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Deterministic typed archive records with key, payload commit, metadata, readback, kind iteration, and root commit. | functional-prototype | Low when used with explicit kind/payload contracts; high if used as domain runtime authority. |
| Archive record metadata | `RecordMeta { cycle_id, tier, flags, boundary_commit }` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Records include cycle/tier/flags and a deterministic boundary commit. | functional-prototype | Medium: `flags` and `boundary_commit` need per-contract meaning to avoid overloading. |
| Archive record kinds | `RecordKind::{WorkspaceSnapshot,SelfState,IitReport,ConsistencyReport,StructuralParams,StructuralProposal,ReplayToken,ReplayApplied,IsmAnchor,CyclePlan,OutputEvent,Other(u16)}` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Fixed canonical variants plus an `Other(u16)` extension surface. | partial | High for Replay/Sleep/Geist until kind allocation is documented; `IsmAnchor` is especially risky because Identity/ISM write semantics are deferred. |
| Payload commit semantics | `ArchiveAppender::build_record`, `build_record_with_commit`, `digest_payload` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Payload bytes or supplied payload digest are bound into record key and root commit. | functional-prototype | Low if bytes are deterministic; high if payload serialization is ad hoc. |
| Boundary commit semantics | `RecordMeta::boundary_commit`; `write_meta` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Boundary commit participates in record-root hashing. | functional-prototype | Medium: each append contract must specify which artifact digest is the boundary commit. |
| Minimal Spine output append/readback | Output event archive append/readback path used by Minimal Spine and consolidation inputs. | `docs/minimal_spine_v1_freeze.md`; `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Existing output/archive records provide source links for bounded consolidation. | operational | Low for current v1.x use; not a template for new hidden appends. |
| Micro append/readback contract | `MinimalSpineMicroMilestoneAppendPayload`; `append_minimal_spine_micro_milestone`; `RecordKind::Other(28)` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_micro_append.rs` | Explicit deterministic payload, evidence append, archive record append, and readback verification. | operational | Low as pattern; medium if copied without reserving new record kind/payload semantics. |
| Meso append/readback contract | `MinimalSpineMesoMilestoneAppendPayload`; `append_minimal_spine_meso_milestone`; `RecordKind::Other(30)` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_meso_append.rs` | Explicit deterministic payload, evidence append, archive record append, and readback verification. | operational | Low as pattern; medium if future payload ordering is nondeterministic. |
| Broad consolidation runtime archive writes | `ConsolidationKernel::run_sleep_replay` appends `ReplayToken` and `ReplayApplied` records. | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Existing broad/prototype replay-side records from consolidation sleep replay. | unsafe/broad | High: not the bounded Replay append contract and must not be reused as Prompt 65 authority. |
| Sleep report append prototype | Sleep coordinator/RSA report path with archive-oriented report surfaces. | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `docs/module_implementation_depth_registry.md` | Existing sleep trigger/report paths are inventory/prototype surfaces, not bounded Sleep append authority. | unsafe/broad | High: could be mistaken for Sleep runtime completion or coordinator activation. |
| Replay bounded records | `MinimalSpineReplayTokenBuildOutput`, `MinimalSpineReplayScheduleBuildOutput`, `MinimalSpineReplayScheduleAudit`, `MinimalSpineReplayAppliedBoundary` | `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs` | Deterministic token/schedule/audit/local boundary values; no Evidence/Archive append. | functional-prototype | Medium: ready for explicit append planning, but runtime replay execution remains forbidden. |
| Sleep bounded records | `MinimalSpineSleepPlanCandidate`, `MinimalSpineSleepPlanAudit`, `MinimalSpineSleepAppliedBoundary` | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `core/crates/ucf-sleep-coordinator/tests/minimal_spine_sleep_e2e.rs` | Deterministic candidate/audit/local boundary values; no Evidence/Archive append. | functional-prototype | Medium: append must not imply SleepApplied/SleepCompleted runtime. |
| Geist/ISM bounded records | `MinimalSpineGeistProjectionCandidate`, `MinimalSpineGeistProjectionAudit`, `MinimalSpineIsmCandidateBoundary` | `domains/geist/crates/ucf-geist/src/lib.rs`; `domains/geist/crates/ucf-geist/tests/minimal_spine_geist_ism_e2e.rs` | Deterministic projection/audit/local candidate-boundary values; no Evidence/Archive append and no ISM write/upsert. | functional-prototype | High: append could be confused with ISM anchor or identity finalization unless named carefully. |
| ISM store prototype | `IsmStore::upsert_anchor` | `domains/geist/crates/ucf-geist/src/lib.rs` | Existing broad local ISM upsert method outside the bounded candidate-only line. | unsafe/broad | Critical: Prompt 64-67 must not call or authorize this for append contracts. |
| Protocol evidence/provenance structures | `ExperienceRecord`, `ProofEnvelope`, `ReplayRunEvidence`, generated protocol structs | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/*.md` | Protocol-level record/evidence schemas exist, but bounded append payloads are currently local contract bytes. | partial | Medium: promote schema only after concrete append/readback contracts stabilize. |
| Ops validation and reports | `docs lint`, `readiness-spine-check`, `workspace-test-check`, `readiness-gate`; CI/nightly workflows | `runtime/ucf-ops`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml` | Validation tooling and workflow evidence exist; generated root reports are freshness-bound. | operational | Low for validation; medium if stale `out/*.json` is treated as source truth. |

Inventory answers:

- EvidenceStore APIs exist for append, optional get, length, and emptiness, with in-memory and file-backed implementations.
- ArchiveStore APIs exist in two layers: broad key/value record/snapshot stores in `ucf-archive` and typed deterministic `ArchiveRecord` append/get/iter_kind/root_commit in `ucf-archive-store`.
- Append/readback helpers already exist for Minimal Spine output paths and explicit consolidation micro/meso append contracts.
- Canonical `RecordKind` values exist for workspace, self-state, IIT/consistency/structural records, replay token/applied, ISM anchor, cycle plan, and output event.
- Extension kinds exist via `RecordKind::Other(u16)`; current bounded consolidation uses `Other(28)` for micro and `Other(30)` for meso.
- `payload_commit` is a deterministic payload-byte digest or supplied commit bound into the archive record key and root commit; `boundary_commit` is metadata included in root hashing and must be contract-specific.
- No bounded Replay/Sleep/Geist/ISM append helpers exist yet; existing broad prototype surfaces must not be treated as those contracts.
- Prompt 65 safe APIs: deterministic bounded Replay structs and their `deterministic_bytes`/digest methods, explicit `EvidenceStore` append/get implementations, explicit `ArchiveAppender` and typed `ArchiveStore` append/get/iter_kind/root_commit, and the micro/meso pattern.
- Risky APIs: broad `ExperienceAppender` use without typed payloads, consolidation `run_sleep_replay` archive writes, sleep trigger/report append prototypes, `RecordKind::IsmAnchor` for candidate-only ISM, and `IsmStore::upsert_anchor`.

## 3. Existing Append Contract Inventory

| Contract | Existing? | Artifact | Append API | Readback API | Authority | Risk |
|---|---:|---|---|---|---|---|
| Minimal Spine output append/readback | yes | Minimal Spine output/archive output event records | Existing output event archive path | Output/archive readback links consumed by consolidation candidates | Evidence/Archive | Low for frozen v1.x; do not modify in this line. |
| Micro milestone append/readback | yes | `MinimalSpineMicroMilestoneAppendPayload` from `MinimalSpineMicroMilestoneBuildOutput` | `append_minimal_spine_micro_milestone` | Evidence `get` plus archive `get` verification | Evidence/Archive with consolidation adapter | Low; best pattern for explicit bounded contracts. |
| Meso milestone append/readback | yes | `MinimalSpineMesoMilestoneAppendPayload` from `MinimalSpineMesoMilestoneBuildOutput` | `append_minimal_spine_meso_milestone` | Evidence `get` plus archive `get` verification | Evidence/Archive with consolidation adapter | Low; best pattern for aggregate provenance. |
| Macro candidate/finalization append | no | Macro candidate and local finalization boundary | none for bounded contract | none | none yet | Medium; later only after Replay/Sleep/Geist append contracts are stable. |
| Compute audit metadata append | no explicit bounded contract | Compute audit/report metadata, where present | none identified as bounded Minimal Spine contract | none | none yet | Medium; should remain outside Prompt 65-67. |
| Broad Geist archive append prototype | no bounded contract | Existing Geist/ISM broad/runtime surfaces | no bounded explicit append helper | none | none for bounded line | Critical if reused; must stay out of Prompt 67. |
| Sleep report append prototype | no bounded contract | Sleep trigger/report/RSA surfaces | broad/prototype report append path, not bounded SleepPlan append | prototype read/report surfaces | prototype only | High; must not imply Sleep runtime activation or SleepCompleted. |
| Replay audit archive kind | partial/historical | `RecordKind::ReplayToken`, `RecordKind::ReplayApplied`; consolidation sleep replay writes | broad `ArchiveAppender` in consolidation runtime path | typed archive store `get`/`iter_kind`, if used | archive-store prototype, not bounded Replay authority | High; Prompt 65 must define explicit Replay bounded payloads instead of inheriting this surface. |

## 4. Target Artifact Inventory

| Artifact | Current authority | Current semantics | Append priority | Append risk | Notes |
|---|---|---|---|---|---|
| `MinimalSpineReplayTokenBuildOutput` | `ucf-replay` bounded builder | ReplayToken intent/reference over macro provenance; no runtime execution and no append. | now | low | Natural first Replay payload because it is intent/reference only. |
| `MinimalSpineReplayScheduleBuildOutput` | `ucf-replay` bounded schedule builder | Planned deterministic ordering; no scheduler/queue/worker and no append. | now | medium | Append must be provenance for planned order only, not scheduler readiness. |
| `MinimalSpineReplayScheduleAudit` | `ucf-replay` verify-only audit | PASS/FAIL consistency check; no runtime side effects. | now | low | Good append candidate if payload records status, failure reasons, and digest links. |
| `MinimalSpineReplayAppliedBoundary` | `ucf-replay` local boundary helper | Local replay-subsystem boundary marker from PASS audit; no runtime apply/execution. | now | medium | Name is risky; append record must state local-only boundary, not replay execution. |
| `MinimalSpineSleepPlanCandidate` | `ucf-sleep-coordinator` bounded candidate builder | Candidate-only SleepPlan from Replay metadata; no Sleep runtime. | later | medium | Prompt 66 after Replay append is stable. |
| `MinimalSpineSleepPlanAudit` | `ucf-sleep-coordinator` verify-only audit | PASS/FAIL consistency check; no SleepApplied/SleepCompleted. | later | low | Strong append candidate after Replay records exist. |
| `MinimalSpineSleepAppliedBoundary` | `ucf-sleep-coordinator` local boundary helper | Local sleep-subsystem boundary marker; no coordinator/runtime activation. | later | high | Name can be misread as Sleep applied/completed; payload must be explicit. |
| `MinimalSpineGeistProjectionCandidate` | `ucf-geist` bounded candidate builder | Candidate-only Sleep-derived projection; no Geist runtime, ISM write, identity, policy mutation, append, or Gateway. | later | medium | Prompt 67 after Sleep append is stable. |
| `MinimalSpineGeistProjectionAudit` | `ucf-geist` verify-only audit | PASS/FAIL consistency check over candidate side-effect flags. | later | medium | Safe if append payload remains audit/provenance only. |
| `MinimalSpineIsmCandidateBoundary` | `ucf-geist` local candidate-boundary helper | Local read-model/candidate-only ISM boundary; no `IsmStore::upsert_anchor`, no IdentityAnchor. | later | critical | Must avoid `RecordKind::IsmAnchor` unless explicitly redefined for candidate-only records; likely needs extension kind. |
| Macro candidate/finalization | `ucf-consolidation` bounded macro/local finalization line | Candidate and local finalization boundary; no identity or runtime authority. | later | high | Defer until Replay/Sleep/Geist records define cross-layer provenance needs. |
| Cross-layer bundle/manifest | none yet | Potential readback manifest linking Replay/Sleep/Geist/ISM append records. | later | medium | Prompt 68 planning target; do not create before per-layer append contracts exist. |
| Runtime scheduler/queue/worker records | none authorized | Runtime execution/control records. | never/currently forbidden | critical | Out of scope until a future scheduler prompt. |
| IdentityAnchor / IdentityFinalization records | none authorized | Identity authority records. | never/currently forbidden | critical | Out of scope until a future identity authority prompt. |
| Gateway write/API-visible mutation records | none authorized | Gateway mutation or production visibility. | never/currently forbidden | critical | Out of scope for the append contract line. |

## 5. Boundary Decisions

| Boundary | Decision | Reason |
|---|---|---|
| Append meaning | Audit/provenance persistence only. | Bounded Replay/Sleep/Geist/ISM records already define local candidate/audit/boundary semantics; append should only persist those facts. |
| Evidence authority | unchanged | Existing `ucf-evidence` and `ucf-archive` evidence surfaces remain the append/readback authority. |
| Archive authority | unchanged | Existing `ucf-archive-store` typed append/get/iter/root semantics remain the archive authority. |
| Runtime meaning | none | Append does not schedule, queue, execute, apply, complete, or activate Replay/Sleep/Geist runtime behavior. |
| Identity meaning | none | Append does not create `IdentityAnchor`, finalize identity, stabilize identity, or make persistent self-authority claims. |
| ISM write/upsert | none | Candidate-boundary append is not `IsmStore::upsert_anchor`, not `RecordKind::IsmAnchor` authority, and not ISM mutation. |
| Gateway meaning | none | Append does not expose Gateway write/read readiness, external visibility, or action authority. |
| Builders/audits/boundaries | append-free | Builders, audits, and local boundary constructors must remain pure and must not hide appends. |
| Explicit helper requirement | required | Each append contract must have a named helper and a named payload type; no generic broad append reuse as authority. |
| Original artifacts | immutable | Append helpers must derive payloads from existing artifacts without mutating them. |
| Readback | required | Each append contract needs deterministic evidence/archive readback tests before closure or readiness claims. |
| Second event log | forbidden | Archive records may index/persist provenance but must not become an independent domain event authority. |
| RecordKind choices | documented per prompt | Extension kind or canonical kind allocation must be explicit to avoid collisions and accidental semantic promotion. |
| Payload bytes | deterministic | Payloads must use stable byte order, sorted vectors where relevant, fixed versions, and domain-separated digests. |
| Gate criteria | unchanged | This roadmap does not change docs, readiness, workspace, or CI gate criteria. |
| Minimal Spine v1.x | unchanged | The freeze remains intact; append planning is additive and future-only. |

## 6. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| Second event log | critical | Archive-store has append/get/root_commit and existing broad record kinds. | Use Evidence/Archive as existing authority only; append records are provenance indexes, not domain-event authority. |
| Hidden append in builders | high | Current bounded builders/audits intentionally take no store/appender handles. | Keep builders/audits/boundaries append-free; append only through explicit helper functions. |
| Append interpreted as runtime applied/completed | critical | Replay and Sleep boundary names include `Applied`; closures explicitly exclude runtime execution/completion. | Payload names and docs must state local-only/audit-only; no scheduler, queue, worker, runtime, or completion flags. |
| Append interpreted as identity anchor | critical | Geist/ISM area includes ISM and identity-adjacent terminology. | Forbid `IdentityAnchor`, `IdentityFinalization`, stable identity claims, and any identity authority in Prompt 65-67. |
| Append interpreted as ISM write/upsert | critical | `IsmStore::upsert_anchor` exists and `RecordKind::IsmAnchor` exists. | Do not call `upsert_anchor`; prefer a clearly bounded extension kind for candidate-boundary provenance unless a later spec reserves otherwise. |
| Append interpreted as Gateway readiness | high | Future Gateway read surfaces are deferred in roadmap docs. | No Gateway write/read API changes and no Gateway visibility claim in append contract prompts. |
| RecordKind collision | high | Existing `Other(28)` and `Other(30)` are used; Replay/Sleep/Geist ranges are not reserved. | Prompt 65 must document extension-kind allocation; later prompts must update the allocation table before adding helpers. |
| Nondeterministic payload bytes | high | Archive keys/root commits bind payload commits; vectors and maps can drift if not sorted. | Use deterministic byte encoders, stable ordering, domain-separated digests, and readback tests. |
| Stale readback | medium | EvidenceStore default `get` returns `None` unless implemented; generated reports are freshness-bound. | Tests must use concrete stores that implement readback and must assert exact payload/record equality. |
| Broad prototype append surfaces accidentally reused | high | Consolidation sleep replay and Sleep report paths can append broad records. | Future prompts must build new bounded helper/payload types and classify broad surfaces as non-authority. |
| Protocol/schema drift | medium | Protocol structs exist, but append payloads currently live as local contract bytes. | Keep schema promotion deferred until after concrete append/readback contracts stabilize. |
| Overclaiming production readiness | high | CI/nightly validation exists but does not equal production retention or runtime authority. | Validation reports support only the checked bounded scope and current HEAD freshness. |
| Boundary commit ambiguity | medium | `RecordMeta::boundary_commit` is generic. | Each append contract must state which artifact digest populates `boundary_commit`. |
| Payload/record kind under-specification | medium | `RecordKind::Other(u16)` is intentionally broad. | Document version, contract string, domain digest, kind, and readback rules in each prompt. |

## 7. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 65 | Replay Evidence/Archive Append Contract | Add an explicit bounded append/readback contract for Replay token, schedule, audit, and local boundary records. | Replay-specific payload structs, helpers, tests, and docs only. | Deterministic payload bytes, record kind allocation, evidence append, archive append, exact readback tests, targeted Replay E2E, docs lint, fmt, clippy. | No Replay runtime execution, scheduler, queue, worker, Sleep/Geist/ISM activation, Gateway, identity, or hidden append. |
| 66 | Sleep Evidence/Archive Append Contract | Add explicit bounded append/readback for Sleep candidate, audit, and local boundary records. | Sleep-specific payload structs, helpers, tests, and docs only. | Deterministic payload bytes, Replay provenance links, evidence/archive readback, targeted Sleep and Replay E2E, docs lint, fmt, clippy. | No Sleep runtime activation, SleepCompleted, memory stabilization, Geist/ISM ingestion, Gateway, identity, or hidden append. |
| 67 | Geist/ISM Evidence/Archive Append Contract | Add explicit bounded append/readback for Geist projection/audit and ISM candidate-boundary provenance. | Geist/ISM-specific candidate-only payload structs, helpers, tests, and docs only. | Deterministic payload bytes, Sleep/Replay provenance links, evidence/archive readback, targeted Geist/Sleep/Replay E2E, docs lint, fmt, clippy. | No Geist runtime, no `IsmStore::upsert_anchor`, no `IdentityAnchor`, no `IdentityFinalization`, no `RecordKind::IsmAnchor` promotion unless explicitly approved. |
| 68 | Cross-Layer Evidence/Archive Readback E2E | Prove deterministic readback across Replay, Sleep, and Geist/ISM append records. | Cross-layer tests and, if needed, a readback-only manifest fixture. | Stable order, payload commits, boundary commits, root commits, and provenance links are asserted across layers. | No scheduler/runtime, no second event log, no Gateway write, no production readiness claim. |
| 69 | Evidence/Archive Docs Overclaim Guard | Audit docs for overclaims after append contracts land. | Documentation cleanup only. | Docs distinguish audit/provenance append from runtime, identity, ISM write/upsert, Gateway, and production authority. | No behavior changes and no closure claim without tests. |
| 70 | Evidence/Archive Readiness Refresh | Refresh validation evidence and closure-readiness docs for the append/readback line. | Validation runs and readiness/closure documentation only. | Fresh docs lint, readiness-spine, workspace-test-check where practical, readiness-gate, targeted tests, workspace tests, fmt, clippy, and diff hygiene are recorded. | Root `out/*.json` remains uncommitted unless policy requires; stale reports cannot support readiness. |
| 71 | Post-Archive Roadmap Selection: Runtime Scheduler vs Identity Anchor vs Prod-Profile | Select the next roadmap line after bounded append/readback contracts. | Roadmap selection only. | Primary, secondary, parallel, and deferred lines are selected from current evidence and open risks. | No runtime/identity/Gateway implementation unless explicitly selected in a later bounded prompt. |

## 8. Open Questions

- Which `RecordKind` ranges are reserved for Replay, Sleep, and Geist/ISM append records?
- Should append payloads live beside source modules or in an archive adapter layer?
- Should Replay/Sleep/Geist append helpers depend on archive/evidence directly or use adapter structs to reduce crate coupling?
- How can protocol schema promotion be avoided too early while keeping payload contracts discoverable?
- Should cross-layer readback include a manifest/bundle, or should Prompt 68 remain pure test aggregation over per-layer records?
- What remains out of scope until runtime scheduler or identity prompts, especially for ReplayApplied, SleepApplied, ISM candidate boundaries, and identity-adjacent names?
- Should `RecordKind::ReplayToken` and `RecordKind::ReplayApplied` be used for bounded Replay append, or should Replay append initially use extension kinds to avoid confusion with broad prototype writes?
- Should `RecordKind::IsmAnchor` remain entirely forbidden for Prompt 67 candidate-boundary records?
- How should `RecordMeta::flags` be partitioned so side-effect status bits are never inferred from append records?

## 9. Recommended Next Prompt

Recommended next prompt: **UCF Prompt 65 — Replay Evidence/Archive Append Contract**.

Prompt 65 should start with Replay because Replay is the upstream provenance source for Sleep and Geist/ISM, has bounded token/schedule/audit/local-boundary records, and can define the first post-consolidation append pattern without runtime activation, identity semantics, ISM mutation, or Gateway authority.
