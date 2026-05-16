# UCF Replay Record Authority and Token Schema Alignment

## 0. Purpose

- This document is a record-authority and schema/naming alignment only.
- It does not implement a Replay Scheduler.
- It does not build replay tokens from consolidation artifacts.
- It does not implement a schedule builder, applied replay runtime, Sleep Cycle Coordinator, Geist/ISM integration, identity finalization, identity anchor, Gateway write API, capability issuance, real-compute activation, Evidence/Archive authority change, second event-log authority, or Minimal Spine v1.x change.
- It treats `ReplayApplied` as a boundary term that must not imply Geist ingestion, ISM writes, identity finalization, memory anchoring, or runtime completion.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `f91c5259290765a5de822f2d41d5797885d88fa7` |
| HEAD short | `f91c5259` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Replay roadmap present | yes |
| `ucf-replay` present | yes |
| `ucf-types` present | yes |
| `ucf-protocol` present | yes |
| `ucf-consolidation` present | yes |

Baseline links:

- [`docs/roadmap/replay_scheduler_roadmap_boundary_audit.md`](replay_scheduler_roadmap_boundary_audit.md)
- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/consolidation_record_authority_schema_alignment.md`](consolidation_record_authority_schema_alignment.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

Baseline commands captured before edits: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -25`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and presence checks for the replay roadmap, `runtime/ucf-replay`, `core/crates/ucf-types`, `protocol/crates/ucf-protocol`, and `domains/consolidation/crates/ucf-consolidation`.

## 2. Replay Record Inventory

| Record / Type | Path | Fields summary | Current use | Maturity | Risk |
|---|---|---|---|---|---|
| `ReplayToken` | `core/crates/ucf-types/src/lib.rs`; committed by `core/crates/ucf-commit/src/lib.rs`; currently constructed in broad consolidation replay cascade | `tier`, `target`, `budget`, `redaction`, `commit` | Shared digest-only token primitive; `commit_replay_token` has deterministic BLAKE3 commitment over tier/target/budget/redaction with domain `ReplayToken`; not used by `runtime/ucf-replay`. | scheduler-candidate primitive | Medium: usable as a primitive, but missing explicit builder provenance/version and must not be treated as scheduled/applied execution. |
| `ReplayScheduled` | `core/crates/ucf-types/src/lib.rs`; broad consolidation cascade only | `tier`, `target`, `budget`, `redaction`, `commit` | Candidate schedule entry shape; local broad consolidation helper has a private deterministic byte encoder, but no public canonical commit helper and no archive kind. | skeleton / scheduler-candidate | High: name can imply a scheduler exists; no scheduler v1 authority or stable public digest domain is defined yet. |
| `ReplayApplied` | `core/crates/ucf-types/src/lib.rs`; broad consolidation cascade; `domains/geist/crates/ucf-geist/src/lib.rs` | `tier`, `target`, `effect_digest` | Candidate applied-effect boundary; broad consolidation can create effects; Geist can compute replay stabilization from effects. Not used by `runtime/ucf-replay`. | unsafe/broad | Critical: term can overclaim completed replay, Geist ingestion, ISM write, identity finalization, or memory anchor. |
| `ReplayPlan` | `runtime/ucf-replay/src/lib.rs` | `t0`, `t1`, optional expected backend-pack digest, strictness, stop-on-first-divergence | Verify/audit plan for existing replay audit path. | audit-only functional-prototype | Medium: audit plan only; not a scheduler plan or queue. |
| `ReplayReport` / `ReplayCounters` | `runtime/ucf-replay/src/lib.rs` | range, overall status, first divergence, missing/mismatch/degraded counters, details | Verify/audit report over ESS experience records; report is returned/written only when explicitly requested. | audit-only functional-prototype | Low if kept report-only; must not become an append authority without explicit future scope. |
| `ReplayResult` / `ReplayItem` / `DiffSummary` | `runtime/ucf-replay/src/lib.rs` | item totals, match/drift/unreplayable counters, persisted/recomputed summaries, drift reasons, truncation flag | Recompute/report surface for golden replay fixture and explicit report writes. | functional-prototype / audit-only | Medium: float epsilon comparison is acceptable for audit/reporting but not scheduler safety policy. |
| `Divergence` | `runtime/ucf-replay/src/lib.rs` | tick, component, expected digest string, observed digest string, hint | Audit divergence detail for verify-only report. | audit-only functional-prototype | Low/medium: human-readable strings are report evidence, not canonical scheduler records. |
| `DriftReason` | `runtime/ucf-replay/src/lib.rs` | digest mismatch, float mismatch, missing field, backend unavailable, decision scoring unavailable | Recompute drift explanation in `ReplayResult`. | audit-only functional-prototype | Medium: includes floats and backend availability; must not drive scheduler policy directly. |
| `ReplayRunEvidence` | `protocol/crates/ucf-protocol/src/lib.rs` | run id/digest, replay plan ref, asset manifest ref, micro-config evidence, steps/timing, summary digest | Protocol-facing evidence for replay runs/microcircuit evidence, not a Replay Scheduler token/schedule/applied schema. | partial protocol evidence | Medium: protocol replay naming exists, but not aligned to scheduler records. |
| `RecordKind::ReplayToken` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | deterministic kind tag `7` within archive record metadata | Existing archive-store kind for possible future replay token append. | skeleton archive hook | High: existence of kind is not permission to append; Evidence/Archive authority remains unchanged. |
| `RecordKind::ReplayApplied` | `domains/archive/crates/ucf-archive-store/src/lib.rs` | deterministic kind tag `8` within archive record metadata | Existing archive-store kind for possible future applied replay append. | skeleton archive hook | Critical: no Prompt 37 append; applied archive semantics require explicit future contract. |
| `ReplayCascade` / `ReplayOutcome` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs` | cascade config plus tokens, scheduled entries, applied entries, selected micro/meso/macro digests | Broad consolidation replay/sleep path can select targets, build tokens/schedules/effects, and append via archive store when invoked. | unsafe/broad historical implementation surface | Critical: not the scheduler v1 authority; must not be activated by Prompt 37/38. |
| `ExperienceRecord` / `ProofEnvelope` | `core/crates/ucf-types/src/lib.rs` via protocol; `protocol/crates/ucf-protocol/src/lib.rs`; Evidence/Archive crates | existing Minimal Spine / protocol evidence fields and proof references | Evidence and protocol records remain current authorities for Minimal Spine evidence/archive paths. | functional-prototype / current-core | High if replay tried to become a second event-log authority; unchanged here. |

Inventory conclusions:

- `ReplayToken`, `ReplayScheduled`, and `ReplayApplied` are defined in `ucf_types::consolidation`, not in `runtime/ucf-replay`, `ucf-protocol`, archive-store, or docs-only.
- `ReplayToken` has a deterministic commitment helper in `ucf-commit`; `ReplayScheduled` and `ReplayApplied` do not have public canonical commitment helpers yet.
- `ReplayScheduled` and `ReplayApplied` have private encoders in the broad consolidation crate, but those are not scheduler v1 public authority.
- Existing replay tests cover `runtime/ucf-replay` audit/recompute behavior; they do not prove scheduler-token/schedule/applied canonical encoding.
- Readiness/replay audit uses `ReplayPlan`, `ReplayReport`, `ReplayResult`, `Divergence`, and `DriftReason`, not `ReplayToken`, `ReplayScheduled`, or `ReplayApplied`.
- Current `ReplayApplied` code paths are broad consolidation/Geist surfaces; they are not an applied replay runtime contract.
- Naming overlaps exist with Sleep (`run_sleep_replay`, sleep coordinator references), Geist (`apply_replay_effects`), and ISM/vector anchor surfaces; these remain out of scope.

## 3. Authority Decisions

Prompt 37 chooses split authority (Option D) for now.

| Concern | Decision | Reason |
|---|---|---|
| `ReplayToken` authority | `ucf-types` owns the primitive shape; `ucf-commit` owns the current deterministic token commitment; Prompt 38 may add a pure builder in `ucf-replay` or a narrow shared location after explicit review. | The type already exists as a shared digest-only primitive and already has a domain-separated commitment. Scheduler construction policy is not yet implemented and should remain separate from the primitive. |
| `ReplayScheduled` authority | Deferred scheduler-facing authority; existing `ucf-types` struct is only a candidate primitive until a Prompt 39 schedule schema/digest contract is added. | No public canonical commitment helper, archive kind, or scheduler contract exists. Treating it as authoritative now would overclaim scheduler readiness. |
| `ReplayApplied` authority | Deferred; current struct is a boundary placeholder only and must not be scheduler v1 completion authority. | Existing broad/Geist use makes the name risky. Prompt 41 must define applied-boundary semantics without Geist/ISM before any promotion. |
| `ReplayPlan` / `ReplayReport` / `ReplayResult` authority | `runtime/ucf-replay` owns audit/recompute report authority only. | These are functional audit/report surfaces and are already tested in the replay crate. They are not scheduler tokens, schedules, or append records. |
| `Divergence` / `DriftReason` authority | `runtime/ucf-replay` owns audit-only divergence and drift-report semantics. | They explain verification/recompute differences. Float/backend reasons make them unsuitable as scheduler safety records. |
| Evidence/Archive role | unchanged | Evidence/Archive remain the only append/readback authorities. Prompt 37 does not append replay records and does not add a second event log. |
| Archive replay kinds | reserved/skeleton hooks only | `ReplayToken` and `ReplayApplied` kinds exist in archive-store, but no future append is allowed without an explicit append contract. No `ReplayScheduled` archive kind is currently present. |
| Protocol schema role | deferred for scheduler records | `ReplayRunEvidence` exists, but scheduler token/schedule/applied protocol schemas are not needed until an external/wire boundary is explicitly scoped. |
| Geist/ISM role | out of scope | `ReplayApplied` is not Geist ingestion, not an ISM upsert, and not identity finalization. |
| Sleep role | out of scope | `ReplayScheduled` is not a Sleep cycle plan or Sleep Cycle Coordinator record. |

## 4. Naming / Semantics Boundary

| Term | Allowed meaning now | Explicitly not allowed |
|---|---|---|
| `ReplayToken` | Deterministic replay intent/reference token over bounded digests and bounded metadata. | Replay execution, scheduled queue entry, applied replay, Geist/ISM input, identity anchor, capability issuance, Evidence/Archive append side effect, Gateway/action trigger. |
| `ReplayScheduled` | Candidate deterministic schedule entry or future plan inclusion record after Prompt 39 defines ordering, duplicate, cap, and digest semantics. | Actual replay completion, Sleep cycle plan, Sleep coordinator operation, hidden append, Gateway/action trigger, runtime apply. |
| `ReplayApplied` | Boundary placeholder for a replay-subsystem/audit effect only after a future explicit applied-boundary prompt. | Geist ingestion, ISM write/upsert, identity finalization, memory anchor, archived completion, successful runtime execution, consolidation finalization. |
| `ReplayPlan` | Audit/verify plan for `runtime/ucf-replay` ranges and strictness. | Scheduler plan, token builder output, queue authority, Sleep plan. |
| `ReplayReport` | Audit report returned or explicitly written by replay tooling. | Canonical append authority, schedule completion record, Evidence/Archive replacement. |
| `ReplayResult` | Recompute/audit result containing match/drift/unreplayable details. | Scheduler readiness proof, real-compute activation proof, action execution proof. |
| `Divergence` / `DriftReason` | Diagnostic explanation for audit/recompute drift. | Policy decision authority, scheduler safety scalar, evidence append decision by itself. |
| Archive replay record kind | Reserved kind tag for possible future explicit append/readback contract. | Authorization to append in Prompt 37/38, second log, hidden write. |
| Protocol replay evidence | Existing protocol-facing run evidence surface. | Scheduler token/schedule/applied schema unless a future prompt adds it explicitly. |

## 5. Evidence/Archive Boundary

- Prompt 37 performs no Evidence append and no Archive append.
- Future replay appends, if any, must use existing Evidence/Archive authorities and must be explicitly scoped, tested, and documented.
- `RecordKind::ReplayToken` and `RecordKind::ReplayApplied` are reserved archive-store hooks only; their existence does not grant write authority.
- No second event-log authority is introduced.
- No current Minimal Spine v1.x Evidence/Archive authority is changed.

## 6. Sleep / Geist / ISM Boundary

- Sleep is out of scope for Prompt 37 and Prompt 38.
- `ReplayScheduled` is not a sleep-cycle record and not a Sleep Cycle Coordinator plan.
- Geist and ISM are out of scope for Prompt 37 and Prompt 38.
- `ReplayApplied` is not Geist ingestion, not an ISM upsert, not identity finalization, and not a memory/identity anchor.
- Existing broad consolidation and Geist replay-effect functions remain historical/broad surfaces, not scheduler v1 authority.

## 7. Prompt 38 Acceptance Criteria

Prompt 38 must satisfy all of the following before any token-builder claim is allowed:

1. Add a pure deterministic replay token builder only.
2. Accept input only from bounded consolidation artifacts or their digests that are already within the current bounded consolidation line.
3. Do not append to Evidence, Archive, or any other store.
4. Do not schedule tokens.
5. Do not apply replay.
6. Do not call Sleep, Geist, ISM, Gateway write/action, capability issuance, identity finalization, or real-compute activation paths.
7. Produce stable token bytes/commitments across repeated runs.
8. Define duplicate handling and deterministic ordering semantics for builder inputs.
9. Define budget/redaction bounds and any version/domain separation needed for builder output.
10. Include tests for stable digest, duplicate/ordering semantics, no appends, and no scheduler/apply/Sleep/Geist/ISM behavior.
11. Keep `ReplayScheduled` and `ReplayApplied` out of the implementation unless a future prompt explicitly promotes them.
12. Preserve Minimal Spine v1.x freeze boundaries and current Evidence/Archive authority.

## 8. Open Questions

- Are existing `ReplayToken` fields enough for the Prompt 38 builder, or is an explicit version/provenance wrapper needed outside the primitive?
- Should protocol scheduler schemas be added later, or should v1 remain internal until replay append/readback is designed?
- Should `ReplayApplied` be deferred completely until Prompt 41 instead of appearing in scheduler v1 code?
- What exactly is archived later: token, schedule, audit report, applied boundary, or none until a release workflow requires it?
- Should `ReplayScheduled` receive a public canonical digest helper only in Prompt 39, after schedule ordering semantics are defined?
- Should archive-store add a `ReplayScheduled` kind later, or should schedule records remain report-only?

## 9. Schema/Test Hardening Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| Option A — docs-only authority alignment | yes | The current safe work is to decide authority, naming, and future acceptance criteria without promoting scheduler behavior. `ReplayToken` already has a commitment helper; `ReplayScheduled`/`ReplayApplied` require future semantics before hardening. | Low: no behavior change; leaves tests for Prompt 38/39/41. |
| Option B — add tests for existing records only | no | Tests for `ReplayScheduled`/`ReplayApplied` would imply a stable digest/schema that is not yet authorized. `ReplayToken` tests can be added with the Prompt 38 builder. | Medium: premature tests could freeze the wrong boundary. |
| Option C — add minimal deterministic bytes/digest helpers | no | Public helpers for scheduled/applied records would promote schema authority before schedule/applied semantics are settled. | High for `ReplayApplied` overclaim; medium for schedule. |
| Option D — move/create protocol records | no | Protocol-facing scheduler schemas are unnecessary and too invasive before an external protocol boundary is required. | High schema churn and Minimal Spine freeze risk. |

No code hardening is implemented in Prompt 37. The required hardening is deferred to the prompt that owns each boundary: Prompt 38 for token builder/token digest tests, Prompt 39 for schedule bytes/digest tests, and Prompt 41 for applied-boundary tests.

## 10. Recommended Next Prompt

Recommended next prompt: **UCF Prompt 38 — Deterministic Replay Token Builder from Consolidation Artifacts**.

## 11. Prompt 38 Implementation Note — Deterministic Replay Token Builder

Prompt 38 is implemented as a pure deterministic builder in `runtime/ucf-replay`:

- API: `MinimalSpineReplayTokenInput`, `MinimalSpineReplayTokenBuildOutput`, and `build_replay_token_from_minimal_spine_input` in `runtime/ucf-replay/src/lib.rs`.
- Test path: `runtime/ucf-replay/tests/minimal_spine_replay_token_builder.rs`.
- Input decision: the builder consumes a bounded digest-only input copied from the macro consolidation finalization boundary and macro candidate provenance, rather than depending directly on `ucf-consolidation`. Callers should provide the macro candidate digest, macro milestone digest, meso aggregation/provenance digest, macro finalization digest, meso count, and source marker.
- Schema gap: the existing `ReplayToken` only carries `tier`, `target`, `budget`, `redaction`, and `commit`. It cannot honestly carry all consolidation provenance fields. `MinimalSpineReplayTokenBuildOutput` wraps the existing `ReplayToken` and preserves the missing macro/meso/finalization provenance plus explicit false side-effect flags.
- Digest behavior: the existing token commitment remains the deterministic `ucf-commit::commit_replay_token` digest. The token target is a deterministic digest of all bounded input links, so a change to macro candidate, macro milestone, meso aggregation, macro finalization, meso count, or source changes the token commitment. The wrapper also exposes its own deterministic digest over token plus provenance.
- Boundary: the builder is intent/reference only. It does not create `ReplayScheduled`, does not create `ReplayApplied`, does not schedule replay, does not apply replay, does not append Evidence/Archive data, does not call Gateway write APIs, does not trigger Sleep/Geist/ISM, does not create an identity anchor, and does not change Minimal Spine v1.x.

Prompt 39 remains the next scheduler prompt. Prompt 38 intentionally stops before ordering, queue, schedule, apply, audit-append, and runtime replay semantics.

## 12. Prompt 39 Completion Update — Schedule Builder Only

Prompt 39 is complete as a pure deterministic planned-order schedule builder over Prompt 38 replay-token build outputs.

| Prompt | Status | Implemented surface | Boundary retained |
|---:|---|---|---|
| 39 | complete | `MinimalSpineReplayScheduleConfig`, `MinimalSpineReplayScheduleBuildOutput`, `MinimalSpineReplayScheduledTokenProvenance`, and `build_replay_schedule_from_minimal_spine_tokens` in `runtime/ucf-replay`; tests in `runtime/ucf-replay/tests/minimal_spine_replay_schedule_builder.rs` | Schedule is planned ordering only: no applied replay, no `ReplayApplied` emission, no Evidence/Archive append, no Sleep Cycle, no Geist/ISM ingestion, no identity anchor/finalization, no Gateway write, no runtime queue/background worker, and no Minimal Spine v1.x change. |

Ordering and cap semantics:

| Concern | Decision | Reason |
|---|---|---|
| Input ordering | Normalize by ascending `replay_token_digest` before schedule construction. | Keeps the builder deterministic and ergonomic while making reversed or otherwise shuffled equal token sets produce the same schedule digest. |
| Duplicates | Reject duplicate `replay_token_digest` values. | A planned schedule must not silently schedule the same replay intent/reference token twice. |
| Cap/limit | Optional `max_tokens`; absent means no cap. If present, zero is rejected and truncation happens after deterministic sorting. | Cap behavior remains deterministic and records `truncated = true` when it drops otherwise valid sorted tokens. |
| Empty input | Reject. | An empty planned replay schedule would be ambiguous and carries no useful replay-token provenance. |

Schema-gap note: `ReplayScheduled` is reused as the scheduler-facing record shape for tier/target/budget/redaction/commit, but it still cannot carry Minimal Spine token provenance or ordering metadata. Prompt 39 therefore records that gap explicitly in `MinimalSpineReplayScheduleBuildOutput` with `scheduled_token_provenance`, `replay_token_digests`, `token_build_output_digests`, `schedule_digest`, boundary flags, count, truncation metadata, and a source marker.

Recommended next prompt: **UCF Prompt 40 — Replay Audit Record / Verify-Only Contract**.
