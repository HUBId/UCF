# UCF Replay Scheduler Roadmap and Boundary Audit

## 0. Purpose

- This document is a roadmap and boundary audit only.
- It does not implement Replay Scheduler behavior.
- It does not change existing replay verify/recompute behavior.
- It is not Sleep Cycle Coordinator readiness.
- It is not Geist/ISM readiness.
- It is not identity finalization, an identity anchor, Gateway write API work, capability issuance, real-compute activation, Evidence/Archive authority changes, a second event-log authority, or a Minimal Spine v1.x change.
- It treats historical replay/sleep/geist docs as historical unless linked through the current architecture index or current roadmap documents.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `eaa6f2d2199cc8d0ef8d2be849776b8c7a957f0b` |
| HEAD short | `eaa6f2d2` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Consolidation closure present | yes |
| Gate stability audit present | yes |
| `ucf-replay` present | yes |
| `ucf-consolidation` present | yes |
| `ucf-geist` present | yes |
| Consolidation E2E present | yes |

Baseline links:

- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/readiness_gate_timeout_stability_audit.md`](readiness_gate_timeout_stability_audit.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)

## 2. Replay Code Inventory

| Concern | Existing API/type | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|---|
| Replay package | `ucf-replay` crate | `runtime/ucf-replay/Cargo.toml` | Standalone runtime crate depending on compute, core, ESS, frames, and types; no consolidation/archive/geist dependency in this crate. | functional-prototype | Medium: current behavior is operational replay/audit, not a scheduler. |
| Replay mode/spec | `ReplayMode`, `ReplaySpec` | `runtime/ucf-replay/src/lib.rs` | Supports `compute_only`, `decision_scoring`, and `full_no_action` modes over tick ranges, with optional backend/seed/budget overrides. | functional-prototype | Medium: `decision_scoring` and `full_no_action` are explicitly non-actioning and mark scoring unavailable rather than executing decisions. |
| Verify/audit plan | `ReplayStrictness`, `ReplayPlan`, `ReplayReport`, `ReplayCounters`, `Divergence` | `runtime/ucf-replay/src/lib.rs` | Verify-only and recompute-stage audit surfaces report missing data, drift, and capped divergence details. | functional-prototype | Medium: good audit substrate, but not a schedule/token authority. |
| Recompute path | `replay_records` | `runtime/ucf-replay/src/lib.rs` | Recomputes compute summaries from nearby control records, compares digests/floats, and returns a report object. | functional-prototype | Medium: uses float epsilon for compute comparison; keep out of safety-critical scheduler policy. |
| Report write side effect | `write_report` and binary `--report` | `runtime/ucf-replay/src/lib.rs`, `runtime/ucf-replay/src/main.rs` | Serializes replay results to an explicit report path. | functional-prototype | Low if explicit; hidden appends remain forbidden. |
| Golden replay tests | `replay_golden.rs` | `runtime/ucf-replay/tests/replay_golden.rs` | Covers golden compute match, drift via seed override, full-no-action no execution, audit missing data, and missing chain digest drift. | functional-prototype | Low for current crate; does not prove scheduler readiness. |
| Golden fixture | `golden_replay_fixture.json` | `runtime/ucf-replay/fixtures/golden_replay_fixture.json` | Three decision fixtures with control/compute fields and digest inputs. | functional-prototype | Low; fixture intentionally lacks compute-chain digest for audit drift coverage. |
| Replay token records | `ReplayToken`, `ReplayScheduled`, `ReplayApplied` | `core/crates/ucf-types/src/lib.rs` | Bounded digest-only consolidation structs exist in `ucf_types::consolidation`. | skeleton | High: schema exists but not validated as scheduler authority. |
| Archive replay kinds | `RecordKind::ReplayToken`, `RecordKind::ReplayApplied` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Archive-store record kinds exist and deterministic tags are assigned. | skeleton | High: no `ReplayScheduled` archive kind; appending replay records needs explicit authority prompt. |
| Consolidation replay cascade | `ReplayCascade`, `ReplayOutcome` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Broader experimental path can build tokens, scheduled events, and applied effects from memory milestone graph plus sleep replay context. | broad-risky | Critical for this roadmap: not the Prompt 37 starting point and must not be treated as Minimal Spine scheduler readiness. |
| Consolidation bounded artifacts | Minimal Spine micro/meso append payloads, macro candidate, local finalization | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | Deterministic bounded pipeline artifacts exist and are validated by E2E tests. | partial | Medium: safe replay input candidates only if consumed read-only and without mutating finalization. |
| Evidence/Archive dependencies | `EvidenceStore`, `ArchiveStore`, `ExperienceAppender` | `core/crates/ucf-evidence/src/lib.rs`, `domains/archive/crates/ucf-archive/src/lib.rs`, `domains/archive/crates/ucf-archive-store/src/lib.rs` | Canonical append/read stores exist; replay crate currently writes only reports, while consolidation append/readback uses Evidence/Archive explicitly. | functional-prototype | High if replay creates hidden appends or a second log; low if explicitly using existing stores later. |
| Geist/ISM replay effects | `GeistKernel::apply_replay_effects` | `domains/geist/crates/ucf-geist/src/lib.rs` | Computes replay stabilization from `ReplayApplied` effects without being scheduler entry point. | partial | Critical: must not be implicitly activated by scheduler/token work. |
| Scheduler | No bounded Minimal Spine Replay Scheduler v1 API | `runtime/ucf-replay`, `domains/consolidation` | No current safe scheduler tied to bounded consolidation artifacts. Existing consolidation cascade is broader sleep/replay machinery. | missing | High: next prompts must align records before implementation. |
| Safe Prompt 37 surface | Replay records/types/docs only | `core/crates/ucf-types`, `domains/archive`, docs | Schema authority and boundary alignment can be audited without behavior changes. | docs-only | Low if limited to record authority and overclaim guards. |

Answers from inventory:

- `ReplayToken`, `ReplayScheduled`, and `ReplayApplied` exist in `ucf_types::consolidation`.
- `RecordKind::ReplayToken` and `RecordKind::ReplayApplied` exist in archive-store; no canonical `ReplayScheduled` archive kind exists today.
- `runtime/ucf-replay` has verify-only/recompute paths and golden/audit tests.
- `runtime/ucf-replay` currently has no dependency on consolidation, archive-store, Geist, ISM, or sleep crates.
- Existing pure/deterministic surfaces include digest encoders/builders, replay audit report construction, and fixture-driven replay comparisons; explicit side effects are report writes and existing Evidence/Archive append APIs outside `ucf-replay`.
- Existing broad-risky surfaces include consolidation `ReplayCascade` and Geist replay stabilization; they must remain out of scope for the first scheduler implementation prompt.

## 3. Replay Record / Type Authority

| Record / Type | Authority module | Existing fields | Used by replay? | Risk |
|---|---|---|---:|---|
| `ReplayToken` | `ucf_types::consolidation` | `tier`, `target`, `budget`, `redaction`, `commit` | Indirectly in consolidation cascade; not used by `runtime/ucf-replay`. | High: needs schema authority and commitment contract before scheduler use. |
| `ReplayScheduled` | `ucf_types::consolidation` | `tier`, `target`, `budget`, `redaction`, `commit` | Indirectly in consolidation cascade; not used by `runtime/ucf-replay`. | High: no archive-store record kind currently maps this as first-class archive authority. |
| `ReplayApplied` | `ucf_types::consolidation` | `tier`, `target`, `effect_digest` | Used by consolidation cascade and Geist replay stabilization; not used by `runtime/ucf-replay`. | Critical: name can overclaim effects, Geist ingestion, or identity finalization. |
| `ReplayPlan` | `runtime/ucf-replay` | `t0`, `t1`, optional backend-pack digest, strictness, stop-on-first-divergence | Yes. | Medium: audit plan, not scheduler token. |
| `ReplayReport` | `runtime/ucf-replay` | Range, status, first divergence, counters, details | Yes. | Low: audit-only if not appended as authority. |
| `ReplayResult` | `runtime/ucf-replay` | Totals, matched/drifted/unreplayable, items, truncation | Yes. | Low: report output only. |
| `DriftReason` / `Divergence` | `runtime/ucf-replay` | Digest/float/missing/backend/scoring drift details | Yes. | Medium: float drift must not become scheduler policy. |
| `MicroMilestone` | `ucf_types::consolidation`; protocol also has Minimal Spine milestone message | Memory struct has digest items/profile/commit; protocol struct has id/time/label | Consolidation, not `runtime/ucf-replay`. | Medium: replay should consume digest/payload read-only, not redefine milestone authority. |
| `MesoMilestone` | `ucf_types::consolidation`; protocol also has Minimal Spine milestone message | Memory struct has micro commits/topic/commit; protocol struct has id/time/label/micro ids | Consolidation, not `runtime/ucf-replay`. | Medium. |
| `MacroMilestone` | `ucf_types::consolidation`; protocol also has macro messages | Memory struct has mesos/trait_updates/commit; protocol Minimal Spine struct has id/time/label/meso ids | Consolidation and Geist, not `runtime/ucf-replay`. | High: replay must not mutate macro finalization or imply identity. |
| `EvidenceEnvelope` / `EvidenceId` | `ucf-evidence`, `ucf-types` | Evidence id, proof/fold proof, logical/wall time | Consolidation append/readback; not direct `runtime/ucf-replay`. | Medium: replay may reference or explicitly append later, but cannot create a new authority. |
| `ArchiveRecord` / `RecordKind` | `ucf-archive-store` | Kind, key, payload commit, meta | Consolidation append/readback and archive-store tests. | High: replay append semantics need a dedicated prompt. |
| Sleep records/context | `ucf-sleep-coordinator`, consolidation `SleepReplayContext` | Sleep replay context and summaries in broader consolidation path | Not in `runtime/ucf-replay`. | Critical for scope: scheduler v1 must not activate Sleep Cycle Coordinator. |
| Geist/ISM refs | `ucf-geist` | `GeistKernel`, `IsmStore`, replay stabilization | Not in `runtime/ucf-replay`. | Critical: no implicit Geist/ISM ingestion. |

Record decisions:

- Replay-facing authority candidates: `ReplayToken`, `ReplayScheduled`, and `ReplayApplied`, but only after Prompt 37 aligns schema, authority, archive mapping, and naming boundaries.
- Audit-only current records: `ReplayPlan`, `ReplayReport`, `ReplayResult`, `Divergence`, `DriftReason`.
- Historical/docs-only or deferred until revalidated: broad consolidation `ReplayCascade`, Sleep replay context, Geist replay stabilization, and any historical replay/sleep/geist docs not promoted through current roadmap docs.
- Missing or unresolved: canonical scheduler API, token-builder location, scheduled archive record kind/policy, explicit replay audit record schema, and exact replay-applied meaning without Geist/ISM.

## 4. Replay / Consolidation Boundary

| Boundary | Decision | Reason |
|---|---|---|
| Replay input source | Consume bounded consolidation artifacts read-only: micro/meso append payload digests, readback digests, macro candidate digest, or local finalization-boundary digest. | Consolidation already provides deterministic artifacts and provenance; replay must not require new consolidation writes. |
| Replay output | Initially token/plan/audit objects only; no hidden appends. | Keeps scheduler planning separate from Evidence/Archive mutation. |
| Evidence/Archive role | Existing stores remain the only append/readback authority; later replay records may be appended only by an explicit prompt and existing Archive/Evidence APIs. | Avoids a second event log and preserves current authority. |
| Geist/ISM role | Out of scope. | Replay tokens or applied reports are not Geist ingestion, ISM upsert, self-state authority, or identity finalization. |
| Sleep role | Later. | Replay Scheduler v1 must not instantiate a Sleep Cycle Coordinator or consume sleep-state feedback unless a later prompt scopes it. |
| Consolidation finalization | Unchanged. | Replay must not mutate macro candidate/finalization boundary or reinterpret local consolidation-level finalization as identity/final memory finalization. |
| Consolidation closure dependency | Replay is not required for consolidation closure. | Closure explicitly forbids Replay/Sleep/Geist/ISM/identity readiness claims. |
| Event-log authority | No second event log. | Any later replay append must use existing Evidence/Archive role and be explicit. |

## 5. Replay / Sleep / Geist Boundary

| Area | Existing API/type | Current behavior | Out-of-scope for next prompt? | Boundary |
|---|---|---|---:|---|
| Replay operational audit | `ReplayPlan`, `ReplayReport`, `replay_audit` | Verifies digest links and optional recompute-stage checks over ESS experience records. | No, for inventory and alignment only. | Audit-only; not scheduler authority. |
| Replay compute harness | `ReplaySpec`, `ReplayResult`, `replay_records` | Recomputes compute summaries and writes explicit reports. | Yes for scheduler implementation; may be referenced. | No action execution; no scheduler side effects. |
| Replay token schema | `ReplayToken`, `ReplayScheduled`, `ReplayApplied` | Digest-only types exist. | No for Prompt 37 schema alignment. | Do not append, schedule, or apply behavior yet. |
| Consolidation replay cascade | `ReplayCascade` | Selects micro/meso/macro candidates and emits token/scheduled/applied values using sleep replay context. | Yes. | Historical/broader path; not Minimal Spine Replay Scheduler v1. |
| Sleep replay context | `SleepReplayContext` and sleep coordinator references | Broader consolidation path can integrate sleep replay summaries. | Yes. | Sleep Cycle Coordinator remains later. |
| Geist replay stabilization | `GeistKernel::apply_replay_effects` | Computes stabilization digest from `ReplayApplied` effects. | Yes. | Replay applied is not Geist ingestion and does not upsert ISM. |
| ISM anchor | `IsmStore`, `InMemoryIsm`, `RecordKind::IsmAnchor` | Geist can manage anchors under policy gates. | Yes. | No identity anchor or identity-finalization semantics. |

## 6. Target Scope

| Layer | Goal | Required inputs | Outputs | Explicit non-goals |
|---|---|---|---|---|
| Replay token | Deterministic token schema and later token generation from bounded consolidation artifact digests. | Micro/meso append payload or readback digests, macro candidate or local finalization-boundary digest, bounded budget/redaction policy. | `ReplayToken` values and deterministic token digest/commit. | No Evidence/Archive append, no scheduler queue, no sleep/geist/ISM, no identity finalization, no Gateway/action trigger. |
| Replay schedule | Deterministic ordering semantics for token plans. | Stable token list, deterministic sort key, explicit range/profile inputs. | Schedule object or `ReplayScheduled` values after schema alignment. | No hidden apply, no second event log, no Sleep Cycle Coordinator, no mutation of consolidation finalization. |
| Replay applied/audit | Verify-only report of replay application boundary. | Token/schedule, replay report/audit evidence, optional existing Evidence/Archive references. | Audit-only report first; optional explicit archive record later. | No Geist ingestion, no ISM upsert, no identity anchor, no capability issuance, no real compute activation. |

## 7. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| Replay completion overclaim | high | Existing `ReplayApplied` type and consolidation cascade can sound like completed replay effects. | Define `ReplayApplied` as audit/effect-boundary only until a later explicit apply contract. |
| Sleep cycle overclaim | critical | Consolidation imports sleep coordinator and has `SleepReplayContext`. | Scheduler v1 must not instantiate or require sleep coordinator. |
| Geist/ISM ingestion overclaim | critical | Geist has `ingest_macro`, `IsmStore`, and `apply_replay_effects`. | No scheduler prompt may call Geist/ISM APIs unless explicitly scoped later. |
| Identity finalization overclaim | critical | Macro/finalization and ISM anchor terms can imply identity. | Keep replay completion separate from identity, anchor, and self-state claims. |
| Evidence/Archive authority confusion | high | Archive-store already has replay record kinds, and Evidence/Archive append APIs exist. | Any replay append must be explicit and use existing stores; no second log. |
| Replay drift/golden mismatch | medium | Replay golden fixture covers drift and missing digest behavior. | Add deterministic scheduler goldens only after schema alignment. |
| Nondeterministic replay ordering | high | Candidate selection and HashMap use can create ordering risk if not normalized. | Sort by canonical digest/order keys and cap outputs deterministically. |
| Hidden append side effects | high | Consolidation append/readback functions exist for milestones. | Token/schedule builders must be pure until append contract prompt. |
| Coupling to real compute | medium | `runtime/ucf-replay` can recompute compute summaries. | Scheduler must consume bounded consolidation artifacts, not activate optional real compute. |
| Coupling to Gateway/action | critical | Runtime modules include gateway/client/action surfaces outside replay. | Replay Scheduler v1 has no Gateway write or action trigger. |
| Historical replay docs overclaim | medium | Current index lists replay harness/audit docs and historical consolidation docs. | Link current audit as planning doc and treat older docs as historical unless current-indexed. |
| Minimal Spine freeze drift | critical | Minimal Spine v1.x is frozen. | No schema/runtime behavior change to Minimal Spine v1.x without explicit future policy. |

## 8. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 37 | Replay Record Authority and Token Schema Alignment | Decide replay-facing authority for `ReplayToken`, `ReplayScheduled`, `ReplayApplied`, archive kinds, and audit-only records. | Docs/schema audit first; minimal code only if existing records need comments/tests. | Complete: [`replay_record_authority_schema_alignment.md`](replay_record_authority_schema_alignment.md) chooses split authority, keeps scheduler/apply/append deferred, and defines Prompt 38 acceptance criteria. | No append behavior, no Sleep/Geist/ISM, no identity finalization. |
| 38 | Deterministic Replay Token Builder from Consolidation Artifacts | Add pure token builder from bounded consolidation artifact digests. | Read-only inputs from micro/meso/macro payloads or digests. | Deterministic unit tests; stable ordering; no store writes. | No macro finalization mutation; no Gateway/action; no real compute activation. |
| 39 | Replay Schedule Builder and Ordering Semantics | Add deterministic schedule builder over replay tokens. | Sort/cap semantics, duplicate handling, schedule digest. | Golden tests prove stable ordering and caps. | No Sleep Cycle Coordinator and no hidden apply. |
| 40 | Replay Audit Record / Verify-Only Contract | Define verify-only replay audit object and optional report bridge. | Audit/report only, not authoritative apply. | Audit tests cover missing, drift, and no-append defaults. | Evidence/Archive append remains deferred and explicit. |
| 41 | Replay Applied Boundary Without Geist/ISM | Clarify and test `ReplayApplied` meaning without Geist/ISM. | Boundary docs and pure applied-effect digest if needed. | Tests prove no Geist/ISM dependencies and no ISM upsert. | Replay applied is not identity finalization or Geist ingestion. |
| 42 | Replay E2E Determinism | Build bounded deterministic E2E over token, schedule, and audit. | Fixture/golden over bounded consolidation artifacts. | Two fresh runs produce identical digests and reports. | No archive mutation unless a prior prompt explicitly added it. |
| 43 | Replay Docs Overclaim Guard | Add docs lint/registry guard for replay claims. | Current index, registry, roadmap docs. | Docs lint catches forbidden overclaims or docs explicitly caveat them. | Historical docs remain preserved; no deletion. |
| 44 | Replay Readiness Refresh | Integrate replay checks into readiness planning without weakening workspace evidence. | Readiness docs/checklist; optional `ucf-ops` verify hook if scoped. | Fresh replay tests and readiness docs pass with existing strict evidence policy. | Replay readiness does not imply Sleep/Geist/ISM, Gateway, or identity readiness. |
| 45 | Explicit Replay Archive Append Contract | If approved, add explicit append/readback for replay audit/token records via existing Evidence/Archive. | Existing archive/evidence APIs only. | Append/readback tests; no second event log. | No hidden appends; no authority change. |
| 46 | Replay/Sleep Integration Roadmap Gate | Plan, not implement, the later Sleep Cycle Coordinator bridge. | Boundary audit only. | Sleep integration prerequisites documented. | No sleep implementation in scheduler prompts. |

## 9. Open Questions

- Are existing `ReplayToken`, `ReplayScheduled`, and `ReplayApplied` records sufficient for bounded Replay Scheduler v1, or do they need version fields and explicit provenance?
- Where should the token builder live: `runtime/ucf-replay`, `domains/consolidation`, or a narrower shared crate?
- Does replay consume Macro candidate digests, local finalization-boundary digests, or both?
- What does replay-applied mean without Geist/ISM, and should the term be constrained to audit/effect-boundary semantics?
- What gets archived and when: token, schedule, audit report, applied boundary, or none until explicit append prompt?
- How does replay verify/recompute relate to `ucf-ops` readiness checks without making workspace evidence less strict?
- What remains out of scope until Sleep/Geist prompts: sleep replay summaries, Geist ingestion, ISM upsert, identity anchor, and identity finalization?

## 10. Recommended Next Prompt

Recommended next prompt: **UCF Prompt 38 — Deterministic Replay Token Builder from Consolidation Artifacts**. Prompt 37 is complete in [`docs/roadmap/replay_record_authority_schema_alignment.md`](replay_record_authority_schema_alignment.md).

## 11. Prompt 38 Completion Update — Token Builder Only

Prompt 38 is complete as a token intent/reference builder only.

| Prompt | Status | Implemented surface | Boundary retained |
|---:|---|---|---|
| 38 | complete | `MinimalSpineReplayTokenInput`, `MinimalSpineReplayTokenBuildOutput`, and `build_replay_token_from_minimal_spine_input` in `runtime/ucf-replay`; tests in `runtime/ucf-replay/tests/minimal_spine_replay_token_builder.rs` | No scheduler, no schedule builder, no `ReplayApplied`, no Evidence/Archive append, no Sleep/Geist/ISM, no identity anchor, no Gateway write, no Minimal Spine v1.x change. |

Design decision: the builder uses a small replay-owned digest input struct rather than a direct consolidation-crate dependency. This avoids broad dependency coupling while still preserving bounded consolidation provenance: macro candidate digest, macro milestone digest, meso aggregation/provenance digest, macro finalization digest, meso count, and source marker.

Schema-gap note: `ReplayToken` remains the existing compact token primitive. Full consolidation provenance is preserved by `MinimalSpineReplayTokenBuildOutput`, not by adding scheduler/apply semantics to `ReplayToken`.

Recommended next prompt: **UCF Prompt 39 — Replay Schedule Builder and Ordering Semantics**.

## 11. Prompt 39 Completion Update — Schedule Builder Boundary

Prompt 39 is complete as a schedule-construction-only step.

| Prompt | Status | Implemented surface | Boundary retained | Recommended next prompt |
|---:|---|---|---|---|
| 39 | complete | Pure deterministic schedule builder from `MinimalSpineReplayTokenBuildOutput` values; deterministic digest/order; duplicate rejection; optional deterministic cap; schedule-provenance wrapper around `ReplayScheduled`. | Planned ordering only. No `ReplayApplied`, no replay execution, no Sleep Cycle Coordinator, no Geist/ISM ingestion, no identity finalization/anchor, no Gateway write/API, no capability issuance, no real compute activation, no Evidence/Archive append, no second event-log authority, no runtime queue/background worker, and no Minimal Spine v1.x change. | **UCF Prompt 40 — Replay Audit Record / Verify-Only Contract** |

Prompt 39 ordering decision: normalize by ascending replay-token digest. Prompt 39 cap decision: no cap by default; optional positive cap truncates after sorting and sets truncation metadata. Prompt 39 duplicate decision: reject duplicate replay-token digests.

The `ReplayScheduled` schema remains a compact scheduled-record shell and cannot by itself carry Prompt 38 token-builder provenance or explicit schedule ordering metadata. Prompt 39 intentionally uses a replay-local schedule build output wrapper to document and preserve that schema gap without changing shared record schemas.

## 12. Prompt 40 Completion Update — Verify-Only Schedule Audit

Prompt 40 is complete and keeps the replay scheduler roadmap in audit-only territory.

| Prompt | Status | Implemented surface | Tests | Boundary retained | Recommended next prompt |
|---:|---|---|---|---|---|
| 40 | complete | `MinimalSpineReplayScheduleAudit` plus `verify_minimal_spine_replay_schedule` in `runtime/ucf-replay`. | `runtime/ucf-replay/tests/minimal_spine_replay_audit_contract.rs` covers PASS/FAIL, deterministic audit digest, token count/order, duplicate detection, truncation metadata, verify-only non-mutation, no `ReplayApplied`, no Sleep/Geist/identity flags, and no Evidence/Archive append flag. | Audit/report only. No actual replay execution, no runtime scheduler/background queue, no Sleep Cycle Coordinator, no Geist/ISM integration, no identity finalization/anchor, no Gateway write/API, no capability issuance, no real compute activation, no Evidence/Archive append, no second event-log authority, no Minimal Spine v1.x behavior change, and no consolidation E2E behavior change. | **UCF Prompt 41 — ReplayApplied Boundary Without Geist/ISM** |

Prompt 41 defined the `ReplayApplied` boundary without connecting it to Geist/ISM, identity finalization, Evidence/Archive append, runtime queues, or real replay execution.

## 13. Prompt 41 Completion Update — ReplayApplied Boundary Without Geist/ISM

Prompt 41 is complete and keeps ReplayApplied semantics local to replay bookkeeping.

| Prompt | Status | Implemented surface | Boundary retained | Recommended next prompt |
|---:|---|---|---|---|
| 41 | complete | `MinimalSpineReplayAppliedBoundary` and `build_replay_applied_boundary_from_audit` in `runtime/ucf-replay`, plus deterministic boundary tests in `runtime/ucf-replay/tests/minimal_spine_replay_applied_boundary.rs`. | The boundary can only be derived from a PASS verify-only schedule audit. It rejects FAIL audits and does not call Geist, ISM, Sleep, Evidence, Archive, Gateway, scheduler queue, real compute, or identity code. It does not mutate token, schedule, or audit records and does not construct the broad `ReplayApplied` runtime/type value. | **UCF Prompt 42 — Replay E2E Determinism** |

ReplayApplied remains explicitly not Geist ingestion, not ISM write/upsert, not identity finalization, not a memory/identity anchor, not sleep completion, not Evidence/Archive append, not Gateway visibility/action, and not actual replay runtime apply.

## 14. Prompt 42 Completion Update — Replay E2E Determinism

Prompt 42 is complete and proves the bounded replay path is deterministic end-to-end without
promoting it to runtime replay execution.

| Prompt | Status | Implemented surface | Tests | Boundary retained | Recommended next prompt |
|---:|---|---|---|---|---|
| 42 | complete | Replay-local E2E test over token build outputs, deterministic schedule build output, verify-only schedule audit, and local applied-boundary marker. | `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs` covers fresh-run digest determinism, token-to-schedule-to-audit-to-boundary provenance continuity, PASS-audit precondition for boundary creation, FAIL-audit rejection, false Sleep/Geist/ISM/identity/Evidence/Gateway flags, duplicate token rejection, zero digest input rejection, tampered schedule audit failure, and absence of Evidence/Archive append or runtime queue markers. | E2E determinism only. No actual replay runtime apply, no runtime scheduler/background queue, no Sleep Cycle Coordinator, no Geist/ISM integration, no identity finalization/anchor, no Gateway write/API, no capability issuance, no real-compute activation, no Evidence/Archive append, no second event-log authority, no Minimal Spine v1.x behavior change, and no consolidation E2E behavior change. | **UCF Prompt 43 — Replay Docs Overclaim Guard** |

Prompt 43 should harden replay documentation against overclaiming: the current E2E proves bounded
record determinism and provenance continuity, not real replay execution, not archive authority, and
not Sleep/Geist/ISM/identity/Gateway readiness.
