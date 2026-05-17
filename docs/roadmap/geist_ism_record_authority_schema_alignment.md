# UCF Geist/ISM Record Authority and Schema Alignment

## 0. Purpose

- This document is record-authority and schema-alignment only.
- It introduces no Geist/ISM implementation.
- It now records the Prompt 57 implementation of a candidate-only Geist projection wrapper.
- It does not implement a Geist projection audit.
- It does not implement an ISM candidate boundary.
- It does not write, upsert, apply, or persist ISM state.
- It does not create an IdentityAnchor.
- It does not finalize identity.
- It does not activate unbounded recursion, runtime Geist, runtime Sleep, Gateway writes, action authority, capability issuance, real compute, Evidence/Archive append, or a second event-log authority.
- It does not alter Minimal Spine v1.x, bounded Consolidation, bounded Replay, bounded Sleep, or gate criteria.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `564706f00cbd66238fd8b34e2610e2d99d6bf56a` |
| HEAD short | `564706f0` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Geist/ISM roadmap present | yes |
| Sleep closure present | yes |
| `ucf-geist` present | yes |
| `ucf-sleep-coordinator` present | yes |
| Sleep E2E present | yes |

Baseline links:

- [`docs/roadmap/geist_ism_roadmap_boundary_audit.md`](geist_ism_roadmap_boundary_audit.md)
- [`docs/roadmap/sleep_closure.md`](sleep_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)

## 2. Geist / ISM Record and API Inventory

| Record / Type / API | Path | Fields / role summary | Current use | Maturity | Risk |
|---|---|---|---|---|---|
| `SelfState` | `domains/geist/crates/ucf-geist/src/lib.rs` | `cycle_id`, five digest commits, bounded `consistency`, and derived `commit`. | Deterministic in-memory record built from caller-supplied commits. It is not derived from bounded Sleep provenance. | functional-prototype | Medium: name can imply persistent self, memory stabilization, or identity unless future use is strictly scoped. |
| `SelfStateBuilder` / `encode_self_state` | `domains/geist/crates/ucf-geist/src/lib.rs` | Builder clamps consistency to `0..=10000`; encoder emits deterministic big-endian bytes excluding the stored commit. | Prototype helper for `SelfState`. | functional-prototype | Medium: usable as a deterministic primitive, but not yet a bounded post-Sleep projection authority. |
| `GeistConfig` | `domains/geist/crates/ucf-geist/src/lib.rs` | `recursion_depth`, `per_cycle_steps`, `consistency_threshold`. | Existing Geist kernel configuration. | partial | High: recursion naming and runtime knobs are too broad for Prompt 57 unless bounded separately. |
| `GeistLoopState` | `domains/geist/crates/ucf-geist/src/lib.rs` | `level`, `anchor`, `context`. | Existing prototype loop state built from Macro commitments and prior loop anchors. | functional-prototype | Critical: `anchor` is not an IdentityAnchor and must not be promoted as one. |
| `ReplayStabilization` / `apply_replay_effects` | `domains/geist/crates/ucf-geist/src/lib.rs` | `drift_reduction`, `commit`; hashes legacy `ReplayApplied` effects. | Prototype replay-stabilization helper. | partial | High: stabilization wording can overclaim memory stabilization and does not use bounded Sleep artifacts. |
| `IsmStore` / `InMemoryIsm` | `domains/geist/crates/ucf-geist/src/lib.rs` | Read `anchors()` plus mutating `upsert_anchor()`. | Prototype in-memory ISM mutation surface. | unsafe/broad | Critical: write/upsert authority exists and remains forbidden for the next bounded line. |
| `GeistKernel::ingest_macro` | `domains/geist/crates/ucf-geist/src/lib.rs` | Consumes `MacroMilestone`, derives loop states, consults `GeistGate`, upserts ISM on accept, appends archive record, optionally updates Sleep state. | Existing prototype integration path. | unsafe/broad | Critical: combines macro ingestion, ISM write, Archive append, and Sleep mutation; not safe for Prompt 57. |
| `GeistGate::allow_ism_upsert` | `core/crates/ucf-policy-ecology/src/lib.rs` | Verify/decision trait returning a boolean for a consistency report. | Existing policy read used by the broad Geist kernel. | partial | High: may be read-only, but any upsert it authorizes remains out of scope. |
| `PolicyEcology` | `core/crates/ucf-policy-ecology/src/lib.rs` | Deterministic policy rules including ISM-upsert, replay, and sleep-phase decisions. | Gate/read decision layer. | functional-prototype | Medium: read-only use is acceptable later; mutation or new policy authority is forbidden. |
| `ExperienceAppender` / `InMemoryArchive` / `FileArchive` | `domains/archive/crates/ucf-archive/src/lib.rs` | Archive append contract and memory/file implementations. | Existing Archive authority and broad Geist append dependency. | operational for archive, unsafe/broad for Geist use | Critical for Geist: Prompt 57 must not append derived Geist records. |
| `ArchiveStore` / `ArchiveAppender` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Store/list/get/append archive records by kind and metadata. | Archive store authority. | functional-prototype | High for Geist: read-only provenance may be allowed later; append authority is unchanged and out of scope. |
| `EvidenceStore` / `AppendLog` | `core/crates/ucf-evidence/src/lib.rs` | Evidence append/read envelope APIs. | Evidence authority. | functional-prototype | High for Geist: no Evidence append is allowed in Prompt 57. |
| `MinimalSpineSleepPlanCandidate` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Deterministic Sleep plan candidate from bounded Replay metadata. | Bounded Sleep candidate-only record. | operational | Low/medium if read-only: valid future input, not Sleep runtime or identity. |
| `MinimalSpineSleepPlanAudit` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Verify-only audit with status, failure reasons, forbidden side-effect flags, and digest. | Bounded Sleep audit. | operational | Low if `Pass` is used as prerequisite only; not an apply or identity signal. |
| `MinimalSpineSleepAppliedBoundary` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Local-only boundary tying candidate/audit/replay digests and enforcing no forbidden side effects. | Bounded Sleep local applied boundary. | operational | Medium: may be provenance only; must not become IdentityAnchor or SleepCompleted. |
| `SleepStateHandle` / `SleepStateUpdater` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Mutable coordinator handle with derived-record and verdict update hooks. | Existing coordinator prototype and broad Geist coupling. | unsafe/broad | Critical: future Geist must not mutate Sleep coordinator state. |
| `MinimalSpineReplay*` records | `runtime/ucf-replay/src/lib.rs` and tests | Bounded replay token/schedule/audit/applied-boundary records. | Bounded Replay provenance already consumed by Sleep E2E. | operational | Low as indirect provenance through Sleep records; direct Geist integration is deferred. |
| `MacroConsolidation*` / `MacroFinalizationBoundary` docs and tests | `domains/consolidation/crates/ucf-consolidation` and docs | Bounded consolidation candidate/finalization boundary surfaces. | Bounded consolidation line. | functional-prototype | Medium/high: Macro finalization must not become a Geist/Identity anchor. |
| `GatewayService` and Gateway records | `runtime/ucf-gateway/src/lib.rs` | Gateway config, auth, control frame, readback, and access records. | Runtime Gateway surface. | partial/functional-prototype | High: no Gateway write/action authority belongs to Geist/ISM Prompt 57. |
| `IdentityAnchor` | docs and boundary-audit references only; no active record authority found for this line | Deferred identity-anchor concept. | Deferred/historical references only. | docs-only/deferred | Critical: no current prompt may create or infer identity anchoring. |
| `MinimalSpineGeistProjectionInput` / `MinimalSpineGeistProjectionCandidate` | `domains/geist/crates/ucf-geist/src/lib.rs` | Prompt 57 local wrapper carrying Sleep audit digest, Sleep candidate digest, optional SleepApplied boundary digest, Replay provenance, token count, deterministic projection digest, and hard false authority flags. | Candidate-only read model from bounded Sleep metadata. | operational-candidate | Low/medium: intentionally not a `SelfState`; it preserves provenance without claiming runtime Geist, ISM persistence, identity, policy, archive, or Gateway authority. |
| `GeistProjectionAudit` | not found | Missing verify-only audit name. | Not implemented. | missing | Should remain verify-only; no apply, acceptance, or identity semantics. |
| `ISMCandidateBoundary` | not found | Missing local candidate/read-model boundary name. | Not implemented. | missing | Must not call existing `upsert_anchor` or create persistent ISM. |

Inventory answers:

- SelfState types exist in `ucf-geist`, but they are functional prototypes, not bounded post-Sleep authority.
- `GeistProjectionCandidate`, `GeistProjectionAudit`, and `ISMCandidateBoundary` do not exist yet.
- IdentityAnchor is present as deferred documentation language and failure-flag vocabulary, not as an implemented Geist/ISM record authority.
- ISM write/upsert APIs exist through `IsmStore::upsert_anchor`, `InMemoryIsm::upsert_anchor`, `GeistGate::allow_ism_upsert`, and `GeistKernel::ingest_macro`; these remain forbidden for the next bounded line.
- Recursive/self-recursion surfaces exist through `GeistConfig::recursion_depth` and `GeistLoopState` construction; these are not acceptable as unbounded runtime recursion.
- Verify-only/audit surfaces exist for Sleep and Replay; Geist verify-only audit does not yet exist.
- Evidence/Archive append surfaces exist and are used by the broad Geist kernel; they remain unchanged and forbidden for Prompt 57.
- Policy mutation was not found as a required Geist/ISM path; Policy Ecology may be consulted read-only only if a future prompt authorizes it.
- Gateway/action surfaces exist elsewhere in the repo and remain out of scope.
- Sleep/Replay/Macro ingestion references exist, but Prompt 57 should use only bounded Sleep artifacts as read-only inputs.

## 3. Authority Decisions

| Concern | Decision | Reason |
|---|---|---|
| SelfState authority | Prompt 57 uses a new local `ucf-geist` projection candidate wrapper instead of reusing `SelfState`. | Existing `SelfState` is deterministic but not tied to bounded Sleep provenance and can overclaim identity/self persistence. |
| GeistProjectionCandidate authority | Option B implemented: new local bounded Geist projection candidate in `ucf-geist`. | Keeps candidate records local and avoids premature `ucf-types` or `ucf-protocol` promotion; audit remains a Prompt 58 follow-up. |
| GeistProjectionAudit authority | Option B later: local verify-only `ucf-geist` audit, not apply. | Existing Geist code has no verify-only audit surface; future audit must not write ISM or Archive. |
| ISMCandidateBoundary authority | Option B later: local candidate/read-model boundary in `ucf-geist`, not existing ISM store authority. | Existing `IsmStore` mutates anchors, so it is too broad for the first bounded step. |
| ISM write/upsert authority | deferred | Current write/upsert APIs exist but are explicitly outside Prompt 56 and Prompt 57 acceptance. |
| IdentityAnchor authority | deferred | No current identity-anchor record authority is allowed; Sleep or Macro boundaries must not become anchors. |
| Sleep input role | read-only bounded provenance | Future projection may consume `Pass` Sleep audit and candidate/applied-boundary digests only; no Sleep coordinator/runtime mutation. |
| Evidence/Archive role | unchanged | Evidence and Archive remain their own authorities; Geist/ISM Prompt 57 must not append. |
| Policy Ecology role | read-only | Policy Ecology can remain a decision/read layer; no mutation or new authority is introduced here. |
| Protocol role | deferred/current | No `ucf-protocol` promotion until record semantics are proven locally and need a protocol-facing contract. |

Prompt 56 chooses documentation-only authority alignment. It does not add record skeletons, builders, audits, or tests because even minimal local skeletons would become schema authority before Prompt 57's pure builder contract is explicit.

## 4. Naming / Semantics Boundary

| Term | Allowed meaning now | Explicitly not allowed |
|---|---|---|
| SelfState | Bounded projection input/output descriptor or prototype deterministic digest container. | Persistent self, identity, identity anchor, memory stabilization, runtime-applied state, or final self record. |
| GeistProjectionCandidate | Future deterministic candidate derived from bounded Sleep provenance only. | Applied Geist state, ISM write, Archive append, Gateway-visible action, identity finalization, or identity acceptance. |
| GeistProjectionAudit | Future verify-only check over a projection candidate. | Applying projection, accepting identity, upserting ISM, writing Evidence/Archive, or mutating Policy/Sleep. |
| ISMCandidateBoundary | Future local candidate/read-model boundary summarizing what could be considered by ISM without writing it. | Persistent ISM upsert, identity anchor, finalization, capability issuance, or second event log. |
| ISM write/upsert | deferred | Use of `upsert_anchor`, persistent ISM mutation, runtime apply, or hidden write through policy/gateway/archive. |
| IdentityAnchor | deferred | Any SleepAppliedBoundary, MacroFinalizationBoundary, Geist anchor, ISM candidate, or digest being treated as identity anchor. |
| IdentityFinalization | forbidden/deferred | Any current acceptance, persistence, capability, or Gateway action based on identity finalization. |
| Self-recursion | bounded/deferred vocabulary only. | Unbounded recursion, autonomous loop activation, runtime scheduler activation, or recursive identity proof. |

## 5. Sleep Input Boundary

| Sleep artifact | Future role | Restrictions |
|---|---|---|
| `MinimalSpineSleepPlanCandidate` | Read-only provenance and candidate input for a future pure projection builder. | Must remain candidate-only; no mutation, no SleepCompleted assumption, no runtime/coordinator call. |
| `MinimalSpineSleepPlanAudit` | Future prerequisite: projection may require `Pass` audit. | Verify-only; failure/pass status is not identity, apply, Archive append, or ISM write authority. |
| `MinimalSpineSleepAppliedBoundary` | Optional read-only provenance tying candidate/audit/replay digests. | Local boundary only; not identity anchor, not SleepCompleted, not Geist apply, not ISM boundary by itself. |
| Sleep runtime/coordinator | none | No `SleepStateHandle`, `SleepStateUpdater`, WAL coordinator, trigger/report runner, journal, or mutation in Prompt 57. |

## 6. Evidence / Archive / Policy / Gateway Boundary

| Boundary | Decision | Reason |
|---|---|---|
| Evidence/Archive | Read-only provenance only in the first Geist/ISM line; no append. | Existing append authorities are broad and already used by the prototype kernel, so bounded projection must not create a second event log or hidden publication path. |
| Policy Ecology | read-only | Policy can be a deterministic decision source later, but no policy mutation, new rule authority, or upsert authorization is introduced here. |
| Gateway | deferred | Gateway read/write/action semantics are not needed to prove local deterministic projection. |
| Action authority | none | Geist/ISM candidate work must not issue capabilities, trigger compute, publish actions, or authorize identity-driven behavior. |

## 7. Prompt 57 Acceptance Criteria

Prompt 57 must satisfy all of the following if it implements code:

- Add a pure deterministic Self-State / Geist projection candidate builder.
- Use input from bounded Sleep artifacts only, preferably `MinimalSpineSleepPlanAudit` plus candidate and optional applied-boundary provenance.
- Do not run or activate `GeistKernel` as the bounded path.
- Do not call `GeistKernel::ingest_macro`.
- Do not call `IsmStore::upsert_anchor` or any ISM write/upsert API.
- Do not create persistent ISM state.
- Do not create an IdentityAnchor.
- Do not perform identity finalization.
- Do not mutate Policy Ecology.
- Do not append to Evidence or Archive.
- Produce a stable digest using deterministic canonical encoding.
- Add boundary tests proving no broad kernel/upsert/archive/sleep-runtime path is needed by the candidate builder.
- Keep all names candidate/audit/boundary-only and avoid identity, finalization, stabilization, completed, applied, or anchor overclaims.

## 8. Open Questions

- Are existing `SelfState`/Geist/ISM records sufficient, or should bounded v1 use new explicitly named records?
- Should the projection candidate live in `ucf-geist` only until proven stable?
- Should Geist consume `MinimalSpineSleepPlanAudit`, `MinimalSpineSleepAppliedBoundary`, or both?
- What does `ISMCandidate` mean without write/upsert?
- What gets archived and when, if a later append-contract prompt authorizes publication?
- How is IdentityAnchor authority kept deferred and mechanically guarded?
- How is recursion bounded if a later prompt adds recursive projection behavior?
- How does Policy Ecology remain read-only while still documenting future gate semantics?

## 9. Recommended Next Prompt

**UCF Prompt 58 — Geist Projection Verify-Only Audit Contract**

## Prompt 57 Implementation Note - Candidate-Only Geist Projection

Prompt 57 adds the first bounded Geist projection candidate surface in `domains/geist/crates/ucf-geist/src/lib.rs` and tests it in `domains/geist/crates/ucf-geist/tests/minimal_spine_geist_projection_candidate.rs`. The chosen design is a **new local projection candidate wrapper**, not reuse of the existing `SelfState`, because `SelfState` cannot honestly carry the required Sleep audit, optional SleepApplied boundary, Replay provenance, token count, and forbidden-authority markers. This documents the schema gap: `SelfState` remains a functional prototype and is not the bounded post-Sleep projection candidate record.

Chosen input model: **PASS `MinimalSpineSleepPlanAudit` plus optional `MinimalSpineSleepAppliedBoundary`**. The direct builder rejects FAIL audits, recomputes/verifies the audit digest, requires no audit failure reasons, checks that the audit candidate digest matches its recomputed candidate digest, and rejects forbidden audit flags. When a SleepApplied boundary is supplied, it must match the Sleep audit digest, Sleep candidate digest, Replay audit digest, Replay schedule digest, and token count, and it must not carry forbidden runtime/identity/archive/Gateway side-effect flags. A lower-level digest input builder exists for deterministic candidate construction from already-validated bounded Sleep metadata; it rejects zero digests, zero token counts, and empty sources.

The Prompt 57 API is:

- `MinimalSpineGeistProjectionInput`
- `MinimalSpineGeistProjectionCandidate`
- `GeistProjectionError`
- `build_geist_projection_candidate_from_sleep_input`
- `build_geist_projection_candidate_from_sleep_audit`

The candidate hard-codes these boundary markers: `candidate_only = true`, `geist_applied = false`, `ism_written = false`, `identity_anchor = false`, `identity_finalized = false`, `policy_mutated = false`, `evidence_archive_appended = false`, and `gateway_visible = false`. Prompt 57 does **not** call `GeistKernel::ingest_macro`, does **not** accept stores, appenders, Gateways, `GeistKernel`, ISM, policy mutation, scheduler, queue, worker, or runtime handles, and does **not** alter Minimal Spine v1.x, bounded Consolidation, bounded Replay, bounded Sleep, Evidence/Archive authority, or gate criteria.
