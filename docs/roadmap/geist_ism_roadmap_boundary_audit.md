# UCF Geist/ISM Roadmap and Boundary Audit

## 0. Purpose

- This is a roadmap and boundary audit only.
- This introduces no Geist/ISM implementation.
- This is not Self-State projection implementation.
- This is not ISM write or upsert implementation.
- This is not identity finalization.
- This is not identity anchor readiness.
- This does not activate unbounded recursion.
- This does not mutate Policy Ecology.
- This does not add Gateway write or action authority.
- This does not issue capabilities or activate real compute.
- This does not change Evidence/Archive authority, append behavior, or event-log authority.
- This does not alter Minimal Spine v1.x, bounded Consolidation, bounded Replay, bounded Sleep, or gate criteria.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `1c012519bd66767e8671c7c820e57e9759d3435b` |
| HEAD short | `1c012519` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Post-Sleep selection present | yes |
| Sleep closure present | yes |
| Replay closure present | yes |
| `ucf-geist` present | yes |
| `ucf-sleep-coordinator` present | yes |
| Sleep E2E present | yes |

Baseline links:

- [`docs/roadmap/post_sleep_roadmap_selection.md`](post_sleep_roadmap_selection.md)
- [`docs/roadmap/sleep_closure.md`](sleep_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

## 2. Geist / ISM Code and Docs Inventory

| Concern | Existing API/type/doc | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|---|
| SelfState record | `SelfState`, `SelfStateBuilder`, `encode_self_state`, canonical self-state hash domain | `domains/geist/crates/ucf-geist/src/lib.rs` | Builds a deterministic in-memory SelfState commit from supplied digest fields and a bounded score; does not consume bounded Sleep records. | functional-prototype | Medium: the name can be overread as stabilized memory or identity unless scoped as a candidate/prototype record. |
| Geist configuration | `GeistConfig { recursion_depth, per_cycle_steps, consistency_threshold }` | `domains/geist/crates/ucf-geist/src/lib.rs` | Holds knobs for existing prototype recursion and consistency logic. | partial | High: `recursion_depth` has no audited Prompt-55 boundary contract and must not become unbounded runtime recursion. |
| Geist loop/projection-like state | `GeistLoopState { level, anchor, context }` plus private `build_self_states` and `build_self_state` | `domains/geist/crates/ucf-geist/src/lib.rs` | Derives deterministic loop states from Macro milestone commits and prior loop anchors. | functional-prototype | Critical: `anchor` vocabulary can be confused with IdentityAnchor; Prompt 56 must rename/scope authority or add strict boundary docs before new use. |
| Macro ingestion | `GeistKernel::ingest_macro` | `domains/geist/crates/ucf-geist/src/lib.rs` | Consumes a `MacroMilestone`, derives loop states, gates ISM upsert, appends a derived archive record, and may update a sleep-state handle. | unsafe/broad | Critical: performs archive append and ISM upsert in prototype code; not safe as the next bounded Geist/ISM v1 entry point. |
| ISM store | `IsmStore`, `InMemoryIsm`, `upsert_anchor`, `anchors` | `domains/geist/crates/ucf-geist/src/lib.rs` | Provides in-memory anchor storage and mutation through upsert. | functional-prototype | Critical: existing upsert is a write surface and must remain outside Prompt 56 candidate-only scope. |
| Replay stabilization | `ReplayStabilization`, `GeistKernel::apply_replay_effects` | `domains/geist/crates/ucf-geist/src/lib.rs` | Computes a deterministic drift-reduction digest from legacy `ReplayApplied` effects. | partial | High: stabilization vocabulary can overclaim memory stabilization and uses legacy replay effect types, not bounded Replay/Sleep provenance. |
| Policy Ecology gate | `GeistGate`, `DefaultPolicyEcology`, `allow_ism_upsert` usage | `domains/geist/crates/ucf-geist/src/lib.rs`; `core/crates/ucf-policy-ecology/src/lib.rs` | Existing kernel asks the gate whether an ISM upsert may occur. | partial | High: must remain read-only for the next line; no policy mutation or hidden authority changes. |
| Evidence/Archive append | `ExperienceAppender` dependency and `archive.append(record)` in Geist kernel | `domains/geist/crates/ucf-geist/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs`; `domains/archive/crates/ucf-archive-store/src/lib.rs` | Existing prototype appends derived records to an archive appender. | unsafe/broad | Critical: future Geist/ISM candidate work must not append to Evidence/Archive unless an explicit append-contract prompt authorizes it. |
| Sleep coupling | `SleepStateHandle`, `SleepStateUpdater` usage in Geist kernel | `domains/geist/crates/ucf-geist/src/lib.rs`; `core/crates/ucf-sleep-coordinator/src/lib.rs` | Prototype optionally records consistency verdict and derived record ID into sleep state. | unsafe/broad | Critical: not compatible with the bounded Sleep closure claim; future Geist must consume Sleep records read-only, not mutate coordinator state. |
| Bounded Sleep inputs | `MinimalSpineSleepPlanCandidate`, `MinimalSpineSleepPlanAudit`, `MinimalSpineSleepAppliedBoundary` | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Candidate-only plan, verify-only audit, and local-only applied boundary exist for bounded Sleep. | operational | Medium: safe only as read-only projection input; not identity, memory stabilization, runtime Sleep, or append authority. |
| Bounded Replay inputs | `MinimalSpineReplayScheduleAudit`, `MinimalSpineReplayAppliedBoundary`, replay E2E tests | `runtime/ucf-replay/src/lib.rs`; `runtime/ucf-replay/tests/minimal_spine_replay_e2e.rs` | Bounded replay supports deterministic token/schedule/audit/local-boundary flow. | operational | Medium: usable only as provenance/reference input, not runtime replay apply or scheduler authority. |
| Bounded Consolidation inputs | Micro/Meso append/readback, Macro candidate, `MinimalSpineMacroFinalizationBoundary` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_consolidation_pipeline_e2e.rs` | Bounded pipeline exists through local consolidation-level finalization boundary. | operational | High: Macro finalization boundary must not be promoted to identity finalization. |
| Geist tests | Unit tests inside `src/lib.rs`; no `domains/geist/crates/ucf-geist/tests/` directory at baseline | `domains/geist/crates/ucf-geist/src/lib.rs` | Tests cover deterministic anchors, consistency report, macro ingestion append/upsert, gate denial, and SelfState determinism. | partial | High: tests validate prototype write/append behavior, not the future bounded candidate/audit boundary. |
| Historical Geist/ISM docs | Architecture/index/registry and historical roadmap references | `docs/current_state_architecture_index.md`; `docs/module_implementation_depth_registry.md`; historical `docs/*` | Current index marks Geist/Self-State/ISM/Recursion as partial prototype and warns conceptual claims outrun integration. | historical | High: historical docs can overclaim current readiness unless linked to this boundary audit. |
| Gateway/action references | Gateway remains a deferred or read-only future surface for bounded state visibility. | `docs/roadmap/post_sleep_roadmap_selection.md`; `runtime/ucf-gateway/src/lib.rs` | No Prompt-55 Gateway change. | skeleton | High: Gateway visibility must not imply write/action authority. |
| Identity anchor APIs | No dedicated bounded `IdentityAnchor` authority found in current Geist crate; prototype uses generic `anchor` names. | `domains/geist/crates/ucf-geist/src/lib.rs`; docs search results | Anchor vocabulary exists, but no authorized identity-anchor line exists. | unknown | Critical: identity anchor remains deferred and must not be inferred from existing anchor fields. |
| Verify-only/audit APIs | Sleep and Replay have verify-only audits; Geist projection audit does not yet exist. | `core/crates/ucf-sleep-coordinator/src/lib.rs`; `runtime/ucf-replay/src/lib.rs`; `domains/geist/crates/ucf-geist/src/lib.rs` | Existing verify-only pattern exists outside Geist; Geist needs a new bounded audit contract later. | partial | Medium: Prompt 56 should align record authority before any implementation. |

Inventory answers:

- SelfState types exist in `ucf-geist`, but they are prototype-level and not a bounded Sleep/Replay/Consolidation projection contract.
- Geist projection APIs, as a named bounded `GeistProjectionCandidate`, do not yet exist; only macro-derived loop-state helpers exist.
- ISM store/upsert APIs exist in `ucf-geist` and are too broad for Prompt 56 implementation.
- Identity anchor authority does not exist; existing `anchor` fields are not identity anchors.
- Recursion-related state exists through `recursion_depth` and loop levels; it is not yet a safe bounded recursion contract.
- Verify-only/audit APIs exist for Replay and Sleep, but not yet for Geist projection.
- Evidence/Archive append references exist in Geist prototype code and must be treated as unsafe/broad for the next line.
- Policy mutation risk exists if the gate is interpreted as mutable authority; Prompt 56 must keep Policy Ecology read-only.
- Gateway/action references remain deferred; no Gateway write or action authority may be introduced.
- Sleep/Replay/Macro ingestion references exist, but future Prompt 56 should only align records and authority, not implement ingestion.
- Safe Prompt-56 material: record authority inventory, schema naming, provenance input classification, forbidden flags, and docs/test acceptance criteria.
- Too-broad material: `GeistKernel::ingest_macro`, `IsmStore::upsert_anchor`, archive append, sleep-state mutation, identity-anchor language, and stabilization claims.

## 3. Geist / Sleep Boundary

| Boundary | Decision | Reason |
|---|---|---|
| Geist input source | Later Geist work may consume bounded Sleep provenance as read-only input from `MinimalSpineSleepAppliedBoundary` or `MinimalSpineSleepPlanAudit`; Prompt 56 should decide record authority before implementation. | Bounded Sleep closure is candidate/audit/local-boundary only and does not authorize runtime coupling. |
| SleepAppliedBoundary role | Local-only Sleep bookkeeping and optional future projection input; never an identity anchor, memory stabilization, or persistent ISM authority. | The Sleep closure forbids Geist/ISM integration, identity finalization, and memory stabilization claims. |
| SleepPlanAudit role | Verify-only validation input that may support future Geist projection audits. | Audit semantics are safer than applied-boundary promotion and align with candidate-only first steps. |
| SleepPlanCandidate role | Candidate-only input metadata; may seed a future projection candidate only after authority alignment. | Candidate records cannot imply action, identity, write, or archive authority. |
| Evidence/Archive role | Read-only provenance reference only for this line; no Geist/ISM append until an explicit append-contract prompt. | Prevents Evidence/Archive authority confusion and avoids a second event-log authority. |
| Runtime coordinator role | No runtime Sleep Coordinator calls, triggers, reports, WAL, journal, or state mutation in Geist/ISM v1 planning. | Bounded Sleep explicitly has no runtime coordinator activation. |

## 4. Geist / ISM / Identity Boundary

| Area | Existing API/type | Current behavior | Boundary |
|---|---|---|---|
| Geist | `GeistKernel`, `GeistConfig`, `GeistLoopState` | Prototype derives loop states from Macro milestones and can append/archive/upsert. | Future v1 must start with candidate-only projection records and verify-only audits, not the broad kernel path. |
| ISM | `IsmStore`, `InMemoryIsm`, `upsert_anchor` | In-memory mutable anchor set with upsert. | Persistent or hidden ISM writes remain deferred; an `ISMCandidateBoundary` must be non-persistent and non-finalizing. |
| SelfState | `SelfState`, `SelfStateBuilder`, `encode_self_state` | Deterministic digest record from supplied components. | Treat as candidate/prototype schema material until record authority and provenance inputs are aligned. |
| Recursive Self / recursion | `recursion_depth`, loop `level`, previous anchor chaining | Prototype builds a finite number of loop states from a numeric depth. | Recursion must be bounded, deterministic, and not runtime-expanded; no unbounded or self-triggering recursion. |
| Identity Anchor | No authorized identity-anchor API; generic `anchor` fields exist. | Prototype anchors are digest labels for loop/ISM behavior. | IdentityAnchor remains deferred; no current field may be promoted to identity authority. |
| Policy Ecology | `GeistGate` and policy ecology crate | Gate can allow/deny prototype ISM upsert. | Read-only decision input only; no policy mutation or authority expansion. |
| SleepAppliedBoundary | `MinimalSpineSleepAppliedBoundary` | Local-only bounded Sleep boundary. | Projection input only if later authorized; not identity, memory, ISM, runtime, or append authority. |
| MacroFinalizationBoundary | `MinimalSpineMacroFinalizationBoundary` | Local consolidation-level boundary in bounded pipeline. | Not identity finalization and not an identity anchor; usable later only as provenance. |
| Evidence/Archive | `ExperienceAppender`, archive crates, evidence crate | Archive append exists in prototype Geist and consolidation/archive surfaces. | No Prompt-55 or Prompt-56 append; future append requires explicit authority and readback contract. |

## 5. Target Scope

| Layer | Goal | Required inputs | Outputs | Explicit non-goals |
|---|---|---|---|---|
| Geist projection candidate | Define a deterministic candidate record that projects Self-State/Geist facts from bounded Sleep/Replay/Consolidation provenance. | Read-only Sleep audit or applied boundary, Replay audit/applied boundary metadata, Macro/Meso/Micro provenance digests, policy digest references if needed. | `GeistProjectionCandidate`-like record with canonical digest, provenance references, bounded recursion metadata, and forbidden authority flags. | No ISM write, no identity anchor, no finalization, no Gateway authority, no archive append, no runtime scheduler. |
| Geist projection audit | Define a verify-only audit that validates candidate provenance, boundedness, digest consistency, and forbidden claims. | Projection candidate plus source provenance records and expected policy/read-only constraints. | `GeistProjectionAudit`-like pass/fail record with deterministic failure reasons. | No mutation, no upsert, no append, no runtime coordinator calls, no policy mutation, no readiness claim. |
| ISM candidate boundary | Define an optional non-persistent boundary that summarizes what would be eligible for ISM review without writing it. | Passing projection audit, candidate digest, provenance digests, explicit non-finalization flags. | `ISMCandidateBoundary`-like local/candidate-only record. | No persistent ISM store write, no `upsert_anchor`, no identity anchor, no identity finalization, no Evidence/Archive append. |

## 6. Risk / Boundary Matrix

| Risk | Severity | Evidence | Guardrail |
|---|---|---|---|
| Identity anchor overclaim | critical | Existing `anchor` fields and `upsert_anchor` names in Geist prototype. | State that anchors are prototype digests only; keep `IdentityAnchor` deferred until a separate authority roadmap. |
| Identity finalization overclaim | critical | Macro finalization and ISM vocabulary could be conflated with identity. | Require all Geist/ISM records to carry non-finalization language and forbid identity-finalization acceptance criteria. |
| Hidden ISM write | critical | `IsmStore::upsert_anchor` and `GeistKernel::ingest_macro` mutate ISM. | Do not use the broad kernel for Prompt 56; first implementation stage must be candidate-only and write-negative-tested. |
| Unbounded recursion | high | `GeistConfig::recursion_depth` and recursive/self-state vocabulary exist. | Require explicit max depth, deterministic iteration, no runtime expansion, and negative tests for oversized depth. |
| Policy mutation | high | Existing Geist gate depends on Policy Ecology. | Treat Policy Ecology as read-only input; no pack, overlay, rule, or gate-criteria mutation. |
| SleepApplied hidden promotion to identity | critical | Sleep boundary could appear to stabilize memory or identity. | SleepAppliedBoundary is only local Sleep bookkeeping and optional read-only projection input. |
| MacroFinalization hidden promotion to identity | critical | Consolidation has local finalization boundary vocabulary. | Macro finalization remains consolidation-local and not identity finalization. |
| Evidence/Archive authority confusion | high | Geist prototype appends derived records to archive. | No append in Prompt 55/56; future append requires explicit contract and no second event-log authority. |
| Gateway/action authority confusion | high | Future read surfaces could expose bounded states. | Gateway remains deferred/read-only; no write/action/capability authority. |
| Memory stabilization overclaim | high | `ReplayStabilization` and Sleep terminology can imply stabilized memory. | Use projection/candidate language only; forbid memory stabilization claims. |
| Historical Geist docs overclaim | high | Historical architecture docs reference Geist/ISM/Self-State/recursion broadly. | Link this audit from current index and post-Sleep selection as the current planning boundary. |

## 7. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 56 | Geist/ISM Record Authority and Schema Alignment | Decide authoritative record names, provenance inputs, forbidden flags, and schema placement. | Docs and schema-alignment only unless small compile-only declarations are explicitly authorized. | Inventory reconciles existing `SelfState`, `GeistLoopState`, ISM store, Sleep/Replay/Consolidation provenance, Evidence/Archive authority, and test plan. | No projection builder, no ISM write, no identity anchor, no append, no Policy mutation. |
| 57 | Self-State Projection Candidate from Sleep Boundary | Implement or specify the first deterministic candidate builder from bounded Sleep provenance if Prompt 56 authorizes it. | Candidate-only builder and negative tests. | Stable digest, sorted inputs, bounded recursion metadata, rejects identity/append/write flags. | No audit pass authority, no ISM upsert, no runtime Sleep Coordinator call. |
| 58 | Geist Projection Verify-Only Audit Contract | Add verify-only audit for projection candidates. | Audit type, deterministic failure reasons, tests. | Audit verifies candidate digest/provenance and rejects forbidden authority claims. | Verify-only; no writes, no append, no finalization. |
| 59 | ISM Candidate Boundary Without Identity Finalization | Define an ISM candidate boundary that remains a read-model candidate. | Candidate boundary and negative tests. | Boundary can reference a passing audit but cannot persist or upsert. | No persistent ISM write, no anchor authority, no identity finalization. |
| 60 | Geist/ISM E2E Determinism | Validate deterministic candidate -> audit -> candidate-boundary flow. | Bounded E2E tests only. | Fresh-run determinism and stable canonical digests across repeated runs. | No runtime scheduler, no Evidence/Archive append, no Gateway visibility. |
| 61 | Geist/ISM Docs Overclaim Guard | Align docs with bounded Geist/ISM claims after implementation steps. | Docs-only cleanup. | Docs distinguish projection candidates from identity, ISM persistence, Sleep completion, and production readiness. | No readiness overclaim, no historical-doc deletion. |
| 62 | Geist/ISM Readiness Refresh | Refresh validation evidence for the bounded line. | Validation and closure evidence only. | Fmt, docs lint, readiness spine, targeted tests, regression E2E tests, workspace tests and clippy where practical. | Do not commit generated `out/*.json`; stale reports cannot support readiness. |
| 63 | Post-Geist Roadmap Selection: Runtime Scheduler vs Evidence Append vs Prod-Profile | Select next line after bounded Geist/ISM candidate work. | Roadmap selection only. | Primary, secondary, parallel, and deferred lines are explicit. | No runtime activation, Gateway write, or append unless explicitly selected later. |
| 64 | Optional ISM Evidence/Archive Append Contract, if Authorized | Define append/readback authority for ISM or Geist records only if later selected. | Contract and tests for append/readback. | Append authority is single-source, deterministic, and does not create a second event log. | Optional only; no identity anchor or finalization. |
| 65 | Optional Identity Anchor Authority Roadmap, not Implementation | Audit what identity-anchor authority would require. | Roadmap/boundary audit only. | Preconditions, authorities, schemas, evidence, and negative boundaries are listed. | No IdentityAnchor implementation, no finalization, no capability issuance. |

## 8. Open Questions

- Are existing SelfState/Geist/ISM records sufficient, or should v1 use new explicitly bounded candidate/audit names?
- Where should the projection candidate builder live: `domains/geist`, a shared core crate, or an ops-only planning surface?
- Should Geist consume `MinimalSpineSleepAppliedBoundary`, `MinimalSpineSleepPlanAudit`, or both?
- What does an ISM candidate mean without write/upsert behavior?
- What gets archived and when, if Evidence/Archive append is later authorized?
- How is identity anchor kept deferred in type names, docs, tests, and failure reasons?
- How is recursion bounded by schema, config, tests, and docs?
- What remains out of scope until identity-anchor prompts: persistent identity, identity finalization, capability issuance, Gateway action authority, policy mutation, and production readiness?

## 9. Recommended Next Prompt

Recommended next prompt: **UCF Prompt 56 — Geist/ISM Record Authority and Schema Alignment**.

Prompt 56 should remain an authority/schema alignment step. It should not implement Self-State projection, ISM write/upsert, identity anchor, identity finalization, Evidence/Archive append, Gateway write/action authority, runtime coordinator calls, Policy Ecology mutation, or Minimal Spine v1.x changes.
