# UCF Evidence/Archive Query Layer Roadmap and Boundary Audit

## 0. Purpose
- Inventory/roadmap only.
- No append/write behavior.
- No Gateway/action authority.
- No runtime execution.

## 1. Baseline
- HEAD: `06925cf942a26d513571d7716dcd81d570d26ae4`.
- Required docs/crates present:
  - `docs/roadmap/post_policy_roadmap_selection.md`
  - `docs/roadmap/evidence_archive_append_readback_closure.md`
  - `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`
  - `core/crates/ucf-evidence`
  - `domains/archive/crates/ucf-archive`
  - `domains/archive/crates/ucf-archive-store`
  - `runtime/ucf-replay`
  - `core/crates/ucf-sleep-coordinator`
  - `domains/geist/crates/ucf-geist`

## 2. Evidence/Archive Query Surface Inventory

| Concern | Path | Current behavior | Query relevance | Risk |
|---|---|---|---|---|
| Evidence readback API | `core/crates/ucf-evidence/src/lib.rs` | `EvidenceStore` exposes `put` and `get`; append-log hash is internal bookkeeping. | `get(evidence_id)` is direct read-only lookup candidate. | Query layer must not call `put` or imply append authority. |
| Archive readback API | `domains/archive/crates/ucf-archive-store/src/lib.rs` | `ArchiveStore` exposes `append`, `get`, `iter_kind`, `root_commit`; `ArchiveAppender` is explicit write helper. | `get/archive_key`, `iter_kind`, `root_commit` are query candidates. | Query layer must not expose/forward `append`/`ArchiveAppender`. |
| Replay bounded append/readback payload | `runtime/ucf-replay/src/lib.rs` | `MINIMAL_SPINE_REPLAY_APPEND_ARCHIVE_KIND = RecordKind::Other(65)` with immediate evidence/archive readback verification. | Replay record summaries can be read by bounded-kind iteration. | Overread as replay runtime apply/scheduler authority. |
| Sleep bounded append/readback payload | `core/crates/ucf-sleep-coordinator/src/lib.rs` | `MINIMAL_SPINE_SLEEP_APPEND_ARCHIVE_KIND = RecordKind::Other(66)` with immediate readback verification. | Sleep provenance summaries can be queried read-only. | Overread as Sleep runtime/coordinator activation. |
| Geist/ISM bounded append/readback payload | `domains/geist/crates/ucf-geist/src/lib.rs` | `MINIMAL_SPINE_GEIST_ISM_APPEND_ARCHIVE_KIND = RecordKind::Other(67)` with immediate readback verification. | Geist/ISM bounded provenance can be queried read-only. | Overread as ISM write/upsert or identity authority. |
| Cross-layer readback E2E | `domains/geist/crates/ucf-geist/tests/minimal_spine_cross_layer_archive_readback.rs` | E2E asserts readback for Replay/Sleep/Geist kinds and keeps non-bounded kinds empty. | Defines concrete bounded set for query candidate baseline. | Conflating bounded kinds with broad kinds (`ReplayApplied`, `IsmAnchor`). |
| Existing docs boundary | `docs/roadmap/evidence_archive_append_readback_closure.md` and `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md` | Explicitly constrains scope to audit/provenance append/readback and defers runtime/Gateway/identity authority. | Query docs can inherit these constraints as non-goals. | Loose wording could turn visibility into authority claim. |
| Gateway surfaces | `runtime/ucf-ops/src/alerts.rs`, roadmap docs | Gateway references are operational/guardrail oriented; query API not defined. | Confirms Gateway action authority remains out of scope. | Query-to-Gateway coupling would create authority drift. |

Required answers:
- Existing readback APIs: `EvidenceStore::get`, `ArchiveStore::get`, `ArchiveStore::iter_kind`, `ArchiveStore::root_commit`.
- Store read surfaces: evidence `get`; archive `get`/`iter_kind`/`root_commit`.
- Bounded append payloads query candidates: Replay `Other(65)`, Sleep `Other(66)`, Geist/ISM `Other(67)`.
- Allocated bounded record kinds: `RecordKind::Other(65/66/67)`.
- Existing dedicated query/index API for Evidence/Archive query line: not yet defined.
- Gateway read surface for this line: deferred; no query-authority bridge.
- Broad read APIs with authority risk: any unbounded `iter_kind`/broad kind use without scoped allowlist.
- Write APIs to avoid: `EvidenceStore::put`, `ArchiveStore::append`, `ArchiveAppender`.

## 3. Boundary Decisions

| Boundary | Decision | Reason |
|---|---|---|
| EvidenceStore | read-only `get` only | Supports evidence-id lookup without adding write authority. |
| ArchiveStore | read-only `get`/`iter_kind`/`root_commit` only | Supports archive-key lookup and bounded iteration deterministically. |
| Append APIs | forbidden | Query layer is read model only, not persistence authority. |
| Gateway | deferred/read-only later | Prevents query visibility from being interpreted as action authority. |
| Runtime | no execution | Query line is inventory/readback only, no scheduler/worker behavior. |
| Identity/ISM | no write/upsert | Maintains Geist/ISM deferred authority boundaries. |
| Policy | no mutation | Keeps policy line verify/candidate boundaries intact. |
| Query result | candidate/read model only | Results are informational and verify-only-audit-ready, not approvals. |

Allowed query semantics:
- read-only lookup by evidence id.
- read-only lookup by archive key.
- read-only iteration by bounded record kind.
- read-only query candidate over Replay/Sleep/Geist/ISM append records.
- deterministic query summary.
- verify-only query audit later.
- no mutation.

Forbidden query semantics:
- append/delete/mutate.
- scheduler activation and replay/sleep/geist execution.
- gateway write/action.
- identity or ISM write/upsert.
- policy mutation.
- production readiness claim.
- second event log creation.
- treating query visibility as authority.

## 4. Risk Matrix

| Risk | Severity | Guardrail |
|---|---|---|
| Query becomes second event log | High | Require read-only request/response contracts with no append path. |
| Query becomes Gateway/action authority | High | Keep Gateway integration deferred and explicitly non-authoritative. |
| Query visibility interpreted as approval | High | Label outputs as candidate/read-model only and add overclaim guard text. |
| Query layer mutates stores | High | Prohibit `put`/`append`/`ArchiveAppender` in query scope. |
| Query returns stale/mismatched provenance | Medium | Include provenance status fields and mismatch signaling in candidate results. |
| Query conflates Replay/Sleep/Geist/ISM semantics | Medium | Keep kind-scoped summaries per bounded kind and layer labels. |
| Query uses broad RecordKind (ReplayApplied/IsmAnchor) incorrectly | High | Enforce bounded allowlist: only `Other(65/66/67)` in EAQ line. |
| Query exposes identity semantics | High | Exclude identity fields and state identity as deferred/non-goal. |
| Query bypasses Evidence/Archive authority | High | All query reads must come from canonical EvidenceStore/ArchiveStore readback paths. |
| Query claims production readiness | Medium | Docs language guard: planning/inventory only, no prod-readiness claim. |

## 5. Proposed Architecture Shape

| Proposed component | Purpose | Inputs | Outputs | Non-goals |
|---|---|---|---|---|
| `EvidenceArchiveQueryScopeV1` | Declare bounded queryable record layers/kinds. | Static allowlist (`Other(65/66/67)`), layer names. | Deterministic scope descriptor. | No dynamic authority expansion; no runtime toggles. |
| `EvidenceArchiveQueryRequestV1` | Represent read-only query intent. | evidence id and/or archive key and/or bounded kind selector. | Canonical request object for query candidate. | No write, no action request, no scheduler trigger. |
| `EvidenceArchiveQueryResultCandidateV1` | Return read-only result candidate with provenance flags. | Readback records from evidence/archive stores. | Candidate result with status/mismatch markers. | No approval semantics, no mutation, no gateway action. |
| `CrossLayerReadbackQueryCandidateV1` | Summarize Replay/Sleep/Geist/ISM bounded readbacks. | Bounded kind slices + evidence/record references. | Deterministic cross-layer summary candidate. | No execution planning, no identity finalization. |
| `EvidenceArchiveQueryVerifyAuditV1` | Verify-only audit over query responses. | Query request/result candidate pair. | PASS/FAIL-style verify audit artifact. | No enforcement/action side effects. |
| `QueryOverclaimGuard` | Keep docs and contracts non-authoritative. | Prompt/docs language + boundary checklist. | Explicit forbidden-claim checklist. | No behavior implementation. |

Placement decision (current recommendation):
- Near-term planning ownership should remain docs-first with type ownership decision deferred to EAQ2.
- Preferred implementation direction to evaluate in EAQ2: a dedicated query-focused crate (or archive-adjacent crate) rather than Gateway, to keep authority boundaries explicit and avoid action coupling.

## 6. Prompt Series Plan

| Prompt | Title | Goal | Acceptance criteria | Guardrails |
|---|---|---|---|---|
| EAQ2 | Query Record Authority and Read-Only Semantics Alignment | Map query record ownership/types and read-only semantics across Evidence/Archive and bounded Replay/Sleep/Geist/ISM records. | Clear authority table for query inputs/outputs and explicit read-only semantics per field. | No policy mutation, no identity authority, no runtime execution semantics. |
| EAQ3 | Replay/Sleep/Geist/ISM Readback Query Candidate | Define bounded query-candidate schema over `Other(65/66/67)` records. | Deterministic candidate schema + bounded-kind allowlist documented. | Candidate/readback only; no scheduler/worker activation; no ISM write/upsert. |
| EAQ4 | Cross-Layer Query Verify-Only Audit Contract | Define deterministic verify-only audit contract for query outputs. | Verify audit failure taxonomy + deterministic rules documented. | Verify-only; no enforcement/action approval. |
| EAQ5 | Query Docs Overclaim Guard | Add explicit overclaim guard for query line wording. | Docs checklist preventing authority/runtime/prod overclaim. | Docs-only hardening; no behavior changes. |
| EAQ6 | Query Readiness Refresh | Run targeted docs/format validation for query artifacts. | Required docs checks pass with fresh report metadata. | No full-workspace claims; no prod-readiness claim. |
| EAQ7 | Post-Query Roadmap Selection | Re-rank next line after query-boundary closure. | Selection matrix + next prompt recommendation documented. | Planning-only; no runtime/authority rollout. |

## 7. Current Status
- Bounded append/readback exists for Replay/Sleep/Geist/ISM.
- Cross-layer readback E2E exists.
- EAQ2 alignment is now documented in `docs/roadmap/evidence_archive_query_record_authority_schema_alignment.md`.
- Query layer is not yet implemented.
- Next step: EAQ3.

## 8. Open Questions
- Which crate owns query types?
- Should query layer live in archive-store, `ucf-ops`, gateway, or a new crate?
- Should Gateway read API wait until query candidate/audit exists?
- What exact record kinds are in scope beyond `Other(65/66/67)` (if any)?
- How to prevent query from becoming authority?
- How to handle stale/missing records?
- How to represent provenance mismatch?
- How to avoid identity overclaim?

## 9. Recommended Next Prompt
UCF Prompt EAQ2 — Query Record Authority and Read-Only Semantics Alignment
