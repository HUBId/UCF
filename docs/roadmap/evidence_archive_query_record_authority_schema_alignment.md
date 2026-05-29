# UCF Evidence/Archive Query Record Authority and Schema Alignment

## 0. Purpose
- Authority alignment only.
- No query implementation.
- No append/write.
- No Gateway/action authority.

## 1. Baseline
- HEAD: `ccfc602c527fcd195c55b26d13d86bbc6fbdc1be`.
- query roadmap present: yes (`docs/roadmap/evidence_archive_query_layer_roadmap_boundary_audit.md`).
- relevant crates present: `ucf-evidence`, `ucf-archive`, `ucf-archive-store`, `ucf-replay`, `ucf-sleep-coordinator`, `ucf-geist`.

## 2. Query Record / API Inventory

| Record/API | Path | Current role | Query relevance | Authority risk |
|---|---|---|---|---|
| `EvidenceStore::get` | `core/crates/ucf-evidence/src/lib.rs` | Evidence read primitive by `EvidenceId`. | Direct read-only lookup primitive. | Low if used read-only; high if conflated with append authority. |
| `EvidenceStore::append` | `core/crates/ucf-evidence/src/lib.rs` | Evidence append authority. | Not required for query. | High: forbidden for EAQ3 query line. |
| `ArchiveStore::get` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Archive record read by `ArchiveKey`. | Direct read-only lookup primitive. | Low if read-only. |
| `ArchiveStore::iter_kind` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Kind-scoped archive iteration. | Needed for bounded `Other(65/66/67)` reads. | Medium/high if unbounded kinds are queried without allowlist. |
| `ArchiveStore::root_commit` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Current archive root provenance primitive. | Useful for readback provenance checks. | Medium if mistaken as apply/approval signal. |
| `ArchiveStore::append` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Archive append authority. | Not required for query. | High: forbidden for EAQ3 query line. |
| `ArchiveAppender` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Write helper building archive records. | No read-only query need. | High: explicit append authority, forbidden for query ownership. |
| `RecordKind::Other(65)` | `runtime/ucf-replay/src/lib.rs` | Bounded Replay append/readback provenance payload kind. | Allowed bounded query source. | Medium if overread as replay runtime/apply authority. |
| `RecordKind::Other(66)` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Bounded Sleep append/readback provenance payload kind. | Allowed bounded query source. | Medium/high if overread as Sleep runtime/SleepCompleted. |
| `RecordKind::Other(67)` | `domains/geist/crates/ucf-geist/src/lib.rs` | Bounded Geist/ISM append/readback provenance payload kind. | Allowed bounded query source. | High if overread as ISM write/upsert or identity authority. |
| `RecordKind::ReplayApplied` / `RecordKind::ReplayToken` | `domains/archive/crates/ucf-archive-store/src/lib.rs`, replay/consolidation usages | Broad existing replay-facing kinds. | Out of bounded EAQ3 source set. | High: too broad for bounded query authority. |
| `RecordKind::SleepApplied` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Broad existing sleep-facing kind. | Out of bounded EAQ3 source set. | High: can imply completion/runtime. |
| `RecordKind::IsmAnchor` / `IdentityAnchor` references | archive-store enums + roadmap/test docs | Broad/deferred identity/ISM anchor semantics. | Out of bounded EAQ3 source set. | Critical: identity/anchor authority must remain deferred. |
| Gateway/read surfaces (`ucf-ops gateway ...`) | `runtime/ucf-ops/src/main.rs` and related tests/docs | Operational gateway tooling surface. | Deferred from query authority line. | High: coupling query visibility to action authority. |
| Existing dedicated Evidence/Archive query/read-model type | docs + target crates | Not implemented yet. | Confirms EAQ3 should define candidate type only. | Low if documented as candidate/read-model only. |

## 3. Authority Classification

| Record/API | Authority decision | Reason |
|---|---|---|
| `EvidenceStore::get` | read primitive/supporting only | Canonical read-only evidence lookup by id; required query primitive. |
| `ArchiveStore::get` | read primitive/supporting only | Canonical read-only archive-key lookup. |
| `ArchiveStore::iter_kind` | read primitive/supporting only | Required for bounded kind scans when constrained to allowlist. |
| `ArchiveStore::root_commit` | read primitive/supporting only | Provenance reference only; not action/apply authority. |
| `EvidenceStore::append` | append authority, forbidden for query | Query line must not mutate evidence. |
| `ArchiveStore::append` | append authority, forbidden for query | Query line must not mutate archive. |
| `ArchiveAppender` | append authority, forbidden for query | Explicit write helper; must stay out of query ownership. |
| `Other(65)` / `Other(66)` / `Other(67)` | bounded query source | Explicit bounded append/readback line already closed for provenance/audit persistence. |
| `ReplayApplied` / `ReplayToken` / `SleepApplied` / `IsmAnchor` / `IdentityAnchor` | broad/prohibited for EAQ3 | Too broad or identity-adjacent; not the bounded EAQ3 source set. |
| Future `QueryCandidate` / `QueryResultCandidate` type | future query candidate authority | Should encode read-model/candidate-only semantics with explicit bounded input scope. |
| Gateway read/action surfaces | gateway-deferred | Keep query ownership independent from gateway/action authority. |

## 4. Naming / Semantics Boundary

| Term | Allowed meaning | Forbidden meaning |
|---|---|---|
| Query | Read-only candidate retrieval and consistency reporting. | Any append/write/apply/approval/action authority. |
| Lookup | Deterministic fetch by key/id/kind within bounded scope. | Hidden mutation, side-effect trigger, or runtime activation. |
| ReadModel | Candidate-only representation of readback state/provenance. | Canonical runtime truth that authorizes actions. |
| QueryCandidate | Pre-authoritative candidate assembled from bounded reads. | Approval, commit, apply, or identity finalization signal. |
| QueryResultCandidate | Verify-only-friendly bounded output with mismatch/failure status. | Success token granting gateway/action authority. |
| Readback | Retrieval and integrity comparison of persisted evidence/archive entries. | Runtime execution proof or completion semantics. |
| Visibility | Operator-facing observability only. | Policy approval, identity acceptance, or authority grant. |
| GatewayRead | Deferred future read surface, if later explicitly scoped read-only. | Gateway action/write/control authority. |
| ArchiveIndex | Bounded lookup aid over existing archive authority. | Second event log or replacement authority source. |
| RecordKind | Explicit bounded selector, especially `Other(65/66/67)` for EAQ3. | Implicit license to query all broad kinds. |
| EvidenceId | Evidence lookup key. | Authorization token for writes/actions. |
| ArchiveKey | Archive record lookup key. | Apply/approval/execute handle. |
| Provenance | Digest/root/key/source lineage metadata and mismatch signaling. | Implicit guarantee of runtime execution or identity finalization. |
| Authority | Explicitly documented capability boundary. | Any inference from mere presence/visibility/readability. |

## 5. Schema Placement Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A. Local query schema in `archive-store`/archive domain | no | Too low-level; risks implying store layer owns cross-layer query semantics. | Authority drift into storage layer. |
| B. Query schema in `ucf-ops` | no (for now) | `ucf-ops` is broad/operational; can blur boundary with gateway/report/action surfaces. | Scope bleed to ops/gateway authority. |
| C. Query schema in terminal layer/test first or docs-first | **yes (EAQ2 docs decision)** | Preserves planning-only boundary, avoids premature ownership, and keeps EAQ3 free to pick high-level query-oriented placement intentionally. | Less canonical until EAQ3 concretizes crate ownership. |
| D. New crate later | deferred candidate | Clean long-term ownership if EAQ3+ proves stable query contract. | Added crate overhead and migration cost. |

EAQ2 decision: docs-first (Option C), with EAQ3 expected to implement a candidate in a high-level query-oriented location, not low-level archive-store by default.

## 6. EAQ3 Acceptance Criteria

| Criterion | Required? | Notes |
|---|---:|---|
| Query candidate type exists | yes | Candidate/read-model only; no authority promotion. |
| Reads/references only bounded `Other(65/66/67)` metadata | yes | Explicit allowlist; broad kinds remain excluded. |
| No append/write APIs | yes | Must not call/use `EvidenceStore::append`, `ArchiveStore::append`, `ArchiveAppender`. |
| No Gateway action | yes | Query result cannot imply control/action semantics. |
| No identity authority | yes | No IdentityAnchor/IdentityFinalization/ISM-upsert semantics. |
| Deterministic bytes/digest (if implemented) | yes | Stable encoding and digest derivation. |
| Missing/stale/mismatched records => candidate failure only | yes | Failure is report state; never mutation/fallback write. |
| Targeted tests only | yes | Query-line scoped tests only; no full-workspace overclaim. |

## 7. Current Status
- Bounded append/readback exists for Replay/Sleep/Geist/ISM (`Other(65/66/67)`).
- EAQ3 implemented in `domains/geist/crates/ucf-geist/src/lib.rs` with `EvidenceArchiveQueryableKindV1`, `EvidenceArchiveQueryRecordRefV1`, and `CrossLayerReadbackQueryCandidateV1` plus deterministic `deterministic_bytes()`/`digest()` read-model-only surface.
- Targeted coverage added in `domains/geist/crates/ucf-geist/tests/evidence_archive_query_candidate_v1.rs`.
- EAQ4 implemented with `CrossLayerReadbackQueryVerifyAuditV1` and `verify_cross_layer_readback_query_candidate_v1` in `domains/geist/crates/ucf-geist/src/lib.rs`, with targeted coverage in `domains/geist/crates/ucf-geist/tests/evidence_archive_query_audit_v1.rs`.
- EAQ5 query docs overclaim guard is complete in `docs/roadmap/evidence_archive_query_layer_roadmap_boundary_audit.md`.
- EAQ6 bounded query closure baseline is documented in `docs/roadmap/evidence_archive_query_layer_closure.md`.

## 8. Open Questions
- Which crate owns query candidates?
- Should query own cross-layer semantics or only store-level references?
- How to handle stale/missing records?
- How to handle `root_commit` / provenance mismatch?
- How to prevent gateway/action authority drift?
- How to avoid creating a second event log?
- How to keep identity semantics deferred?

## 9. EAQ6 Closure Link

- Current bounded Evidence/Archive query closure baseline: `docs/roadmap/evidence_archive_query_layer_closure.md`.
- Closure remains read-model-only and verify-only for the bounded `Other(65/66/67)` line.
- Closure does not claim Gateway Read API, append/write authority, action authority, identity/ISM authority, runtime scheduler execution, a second event log, or prod readiness.

## 10. Recommended Next Prompt

UCF Prompt EAQ7 — Post-Query Roadmap Selection
