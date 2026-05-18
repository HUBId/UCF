# UCF Post-Geist Roadmap Selection

## Historical status note

Historical status note: this selection predates Prompts 65-68. The selected Evidence/Archive append/readback line now has bounded Replay, Sleep, Geist/ISM audit/provenance persistence and cross-layer readback E2E; this does not imply runtime scheduler readiness, identity authority, Gateway authority, production readiness, Evidence/Archive authority changes, a second event log, or Minimal Spine v1.x changes.

## 0. Purpose

- Select the next line after bounded Geist/ISM closure.
- This is a planning and boundary-selection document only.
- No implementation is introduced here.
- No Evidence/Archive append readiness claim is introduced here.
- This does not activate a runtime scheduler, queue, worker, Geist runtime, Sleep runtime, Replay runtime execution, Gateway write API, capability issuance, real compute, IdentityAnchor, IdentityFinalization, ISM write/upsert, or Evidence/Archive append behavior.
- Minimal Spine v1.x remains frozen and unchanged.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `43e086731a5bbf1eb99279eaeed7c0b7f23bc3d9` |
| HEAD short | `43e08673` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Geist/ISM closure present | yes |
| Sleep closure present | yes |
| Replay closure present | yes |
| Consolidation closure present | yes |
| Compute closure present | yes |
| Freeze doc present | yes |
| `ucf-evidence` present | yes |
| `ucf-archive` present | yes |
| `ucf-archive-store` present | yes |

Baseline links:

- [`docs/roadmap/geist_ism_closure.md`](geist_ism_closure.md)
- [`docs/roadmap/sleep_closure.md`](sleep_closure.md)
- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

Baseline commands used for this selection: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -30`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, required file-presence checks, required crate-presence checks, the mandatory roadmap/code inventory read, and the requested broad `rg` inventory across docs, runtime, core, domains, protocol, workflows, and the root README.

## 2. Candidate Inventory

| Candidate | Relevant paths | Current maturity | Tests present | Boundary risk | Dependency on completed lines | Can remain bounded? | Difficulty | Notes |
|---|---|---|---|---|---|---|---:|---|
| A. Evidence/Archive Append Contracts for Replay/Sleep/Geist/ISM | `core/crates/ucf-evidence/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs`; `domains/archive/crates/ucf-archive-store/src/lib.rs`; `runtime/ucf-replay/src/lib.rs`; `core/crates/ucf-sleep-coordinator/src/lib.rs`; `domains/geist/crates/ucf-geist/src/lib.rs`; existing closure and record-authority docs | partial | Evidence/Archive crates are covered by workspace tests; bounded Replay/Sleep/Geist/ISM E2E tests exist; no explicit append/readback tests for Replay/Sleep/Geist/ISM audit or boundary records yet | medium | Directly depends on bounded Consolidation explicit append/readback and bounded Replay/Sleep/Geist audit/boundary records | yes | L | Safest next architecture block because it can make audit/provenance records explicit without adding runtime, identity, Gateway, or ISM write semantics. Must preserve Evidence/Archive as the authority and avoid a second event log. |
| B. Runtime Replay/Sleep Scheduler / Queue | `runtime/ucf-replay/src/lib.rs`; `core/crates/ucf-sleep-coordinator/src/lib.rs`; `runtime/ucf-ops`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml`; Replay and Sleep closure docs | partial | Bounded ReplaySchedule and SleepPlan tests exist; no approved runtime scheduler/queue/worker tests for this post-Geist line | high | Builds on ReplaySchedule and SleepPlanCandidate, but also depends on append/readback clarity to avoid ambiguous runtime authority | yes, only as a later deterministic design or explicit local contract | XL | Important but premature. Scheduler, queue, worker, or background behavior could couple planning records to runtime action before provenance and authority are stable. |
| C. Prod-profile / Workspace Evidence Stability | `runtime/ucf-ops`; `docs/readiness_gate.md`; `docs/artifact_convention_v0.md`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml`; root `out/*.json` when generated | mixed | `docs lint`, `readiness-spine-check`, `workspace-test-check`, `readiness-gate`, workspace tests, clippy, CI and nightly workflows | medium | Operationally spans all completed lines because gate evidence freshness must cover the current HEAD and bounded closures | yes | M | Strong parallel validation line. It hardens evidence freshness and CI reliability without adding UCF functional behavior. |
| D. Identity Anchor Authority Roadmap | `domains/geist/crates/ucf-geist/src/lib.rs`; `docs/roadmap/geist_ism_closure.md`; `docs/roadmap/geist_ism_record_authority_schema_alignment.md`; registry entries referencing ISM/identity | docs-only | No bounded IdentityAnchor implementation or tests are authorized by the Geist/ISM closure | critical | Depends on stable append/provenance, protocol, policy, and governance boundaries after Geist/ISM candidate records | yes, as roadmap-only, but not as implementation yet | XL | Semantically highest risk. It must wait until audit/provenance records and authority ownership are explicit enough to avoid accidental identity finalization. |
| E. Protocol Schema / Provenance Evolution | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/v1.md`; `protocol/crates/ucf-protocol/spec/messages_v1.md`; `core/crates/ucf-types`; wrapper provenance docs for Consolidation, Replay, Sleep, and Geist/ISM | partial | Protocol tests and workspace tests exist; no dedicated post-Geist cross-layer provenance-evolution test plan yet | high | Cross-cuts bounded Consolidation, Replay, Sleep, Geist/ISM, Evidence, and Archive records | yes, if additive-only, versioned, deterministic, and fixture/golden-aware | XL | Strategically useful as a secondary line after append/readback contracts reveal which provenance fields are stable and worth promoting. |
| F. Gateway Read API Expansion | Gateway and status-surface docs/code; bounded state records in Consolidation, Replay, Sleep, Geist/ISM crates; `runtime/ucf-ops` docs visibility | skeleton | No dedicated bounded-state read-only Gateway expansion tests identified for this line | high | Depends on stable readback surfaces so Gateway visibility cannot be mistaken for write/action authority | yes, if read-only and non-authoritative | L | Useful later for observability. It should wait until append/readback surfaces are stable and authority language is unambiguous. |

## 3. Selection Criteria Score

Scoring uses 0-5 for each underlying criterion. The displayed total is the sum of all ten requested criteria: freeze safety, natural fit after Consolidation -> Replay -> Sleep -> Geist/ISM, overclaim-risk reduction, architectural truthfulness, future unlock value, prompt granularity, determinism/CI fit, runtime-coupling avoidance, authority-confusion avoidance, and movement toward the original architecture.

| Candidate | Freeze safety | Builds on completed lines | Reduces overclaim risk | Unlocks future work | CI-friendliness | Authority clarity | Strategic value | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Evidence/Archive Append Contracts for Replay/Sleep/Geist/ISM | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 46 |
| B. Runtime Replay/Sleep Scheduler / Queue | 2 | 4 | 2 | 5 | 2 | 2 | 4 | 27 |
| C. Prod-profile / Workspace Evidence Stability | 5 | 3 | 4 | 3 | 5 | 4 | 4 | 39 |
| D. Identity Anchor Authority Roadmap | 3 | 3 | 2 | 5 | 3 | 1 | 4 | 27 |
| E. Protocol Schema / Provenance Evolution | 4 | 4 | 4 | 5 | 3 | 3 | 5 | 38 |
| F. Gateway Read API Expansion | 3 | 3 | 2 | 3 | 3 | 2 | 3 | 28 |

## 4. Roadmap Decision

| Decision | Selected line | Reason | Risks | Guardrails |
|---|---|---|---|---|
| Primary next line | A. Evidence/Archive Append Contracts for Replay/Sleep/Geist/ISM | It is the most natural next step after bounded Consolidation, Replay, Sleep, and Geist/ISM because it strengthens auditability and provenance while preserving bounded semantics. | Could be mistaken for a second event log, runtime apply, identity stabilization, or production readiness. | Start with roadmap/boundary audit only; append must be explicit; Evidence/Archive remain authority; no runtime, identity, Gateway write, ISM write/upsert, or Minimal Spine v1.x change. |
| Secondary next line | E. Protocol Schema / Provenance Evolution | Append/readback contracts will reveal which provenance surfaces are stable enough for later protocol/schema treatment. | Broad schema work can cause compatibility churn or imply stronger authority than local records have. | Additive-only, versioned, deterministic, fixture-aware, and only after append/readback contracts are clear. |
| Parallel validation line | C. Prod-profile / Workspace Evidence Stability | Workspace evidence freshness remains a watch item; hardening it improves operational confidence without changing UCF behavior. | Operational work could be mistaken for functional roadmap progress or could weaken gates if handled poorly. | Do not weaken gates; stale or missing workspace evidence fails; profile expectations must be explicit. |
| Deferred line | B. Runtime Replay/Sleep Scheduler / Queue | Runtime scheduling is important but should not precede clear append/readback authority. | Background worker/action/runtime coupling, nondeterminism, and authority confusion. | No scheduler/queue/worker implementation until append/readback contracts and deterministic tests are explicit. |
| Deferred line | D. Identity Anchor Authority Roadmap | Identity semantics are too risky before provenance and governance boundaries are explicit. | Accidental IdentityAnchor, IdentityFinalization, ISM write/upsert, or memory stabilization claim. | Keep roadmap-only until append/provenance, policy, and governance boundaries are explicit. |
| Deferred line | F. Gateway Read API Expansion | Gateway visibility should not precede stable readback surfaces. | Read-only visibility could be mistaken for write/action authority or production readiness. | Strictly read-only, non-authoritative, and later than append/readback surface stabilization. |

## 5. Guardrails for Selected Line

The selected primary line is Evidence/Archive Append Contracts for Replay/Sleep/Geist/ISM. It must obey these guardrails:

- The first prompt is Roadmap and Boundary Audit only.
- Append must be explicit.
- No hidden append in builders, audits, or boundaries.
- Evidence/Archive remain the append/readback authority.
- No second event log is created.
- No identity meaning is introduced.
- No runtime scheduler meaning is introduced.
- No Gateway write is introduced.
- No production readiness claim is introduced.
- Append records remain audit/provenance only.
- Replay, Sleep, Geist, and ISM semantics remain bounded.
- Any later `ISMCandidateBoundary` append authorization is not an ISM write/upsert.
- `IdentityAnchor` remains deferred.
- Stable readback tests are required before any closure claim.
- Minimal Spine v1.x remains unchanged.

## 6. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 64 | Evidence/Archive Append Contracts Roadmap and Boundary Audit | Select exact append/readback authority boundaries for Replay, Sleep, Geist, and ISM candidate records. | Docs-only roadmap and boundary audit. | Candidate records, authority owners, forbidden claims, test strategy, and prompt order are documented. | No implementation; no append readiness claim; no runtime, identity, Gateway write, or ISM upsert. |
| 65 | Replay Evidence/Archive Append Contract | Add an explicit bounded append/readback contract for Replay audit and applied-boundary records if authorized by Prompt 64. | Replay-specific contract and tests only. | Deterministic explicit append and readback tests pass without runtime replay execution. | No scheduler/queue/worker; no Sleep/Geist/ISM activation; no hidden append. |
| 66 | Sleep Evidence/Archive Append Contract | Add an explicit bounded append/readback contract for Sleep audit and applied-boundary records if authorized by Prompt 64. | Sleep-specific contract and tests only. | Deterministic explicit append and readback tests pass without Sleep runtime activation. | No coordinator runtime activation; no memory stabilization; no hidden append. |
| 67 | Geist/ISM Evidence/Archive Append Contract | Add an explicit bounded append/readback contract for Geist audit and ISM candidate-boundary records if authorized by Prompt 64. | Geist/ISM-specific contract and tests only. | Deterministic explicit append and readback tests pass while records remain candidate-only and verify-only. | No Geist runtime; no ISM write/upsert; no IdentityAnchor; no IdentityFinalization. |
| 68 | Cross-Layer Evidence/Archive Readback E2E | Prove deterministic readback across Replay, Sleep, and Geist/ISM append records. | Cross-layer tests only; no runtime activation. | Stable E2E readback order, digests, and provenance assertions pass. | No second event log; no Gateway write; no production readiness claim. |
| 69 | Evidence/Archive Docs Overclaim Guard | Remove or qualify documentation that overstates append/readback semantics. | Docs cleanup only. | Docs distinguish audit/provenance append from runtime, identity, and production authority. | No new feature behavior; no claim of final closure unless tests justify it. |
| 70 | Evidence/Archive Readiness Refresh | Refresh validation evidence for the append/readback line after implementation prompts. | Validation and closure-readiness docs only. | Fresh docs lint, readiness-spine, targeted tests, workspace-test evidence where practical, readiness-gate, fmt, and clippy are recorded. | Generated root `out/*.json` remains uncommitted unless policy requires; stale reports cannot support readiness. |
| 71 | Post-Archive Roadmap Selection: Runtime Scheduler vs Identity Anchor vs Prod-Profile | Select the next roadmap line after bounded append/readback contracts. | Roadmap selection only. | Primary, secondary, parallel, and deferred lines are selected from current evidence. | No runtime or identity activation unless explicitly selected in a later bounded prompt. |

## 7. Deferred Lines

Runtime Replay/Sleep Scheduler / Queue is deferred because bounded Replay and Sleep currently stop at deterministic local planning, audit, and boundary records. A scheduler, queue, worker, or background runtime would introduce action coupling and possible nondeterminism before append/readback authority is stable.

Identity Anchor Authority Roadmap is deferred because bounded Geist/ISM closure explicitly excludes IdentityAnchor, IdentityFinalization, ISM write/upsert, stable identity, and memory stabilization. That semantic authority should wait until append/provenance and policy/governance boundaries are explicit.

Protocol Schema / Provenance Evolution is secondary rather than primary because it should be informed by concrete append/readback contracts. Schema work should follow stable provenance needs instead of guessing them up front.

Gateway Read API Expansion is deferred because Gateway visibility can be mistaken for action or authority. It should wait for stable readback surfaces and remain strictly read-only and non-authoritative if later selected.

Prod-profile / Workspace Evidence Stability should proceed as a parallel validation line. It is operationally important but does not replace the next architecture block.

## 8. Revalidation Rules

Before each prompt in the selected line:

- Capture `pwd`, branch, `git status --short`, full and short HEAD, recent log, workspace package count, and relevant file/crate presence checks.
- If `git status --short` is non-empty at baseline, stop and report dirty paths before making changes.
- For docs-only prompts, run at minimum:
  - `cargo fmt --check`
  - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
  - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
  - targeted package tests for affected crates when practical
  - `git diff --check`
- For implementation prompts, run targeted Replay/Sleep/Geist/ISM tests and append/readback tests relevant to the touched crate.
- Before any closure/readiness claim, attempt:
  - `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`
  - `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --workspace-test-report ./out/workspace_test_report.json --out ./out/gate_report.json` only when the workspace-test report exists and is fresh
  - `cargo test -p ucf-geist --test minimal_spine_geist_ism_e2e -- --nocapture`
  - `cargo test -p ucf-sleep-coordinator --test minimal_spine_sleep_e2e -- --nocapture`
  - `cargo test -p ucf-replay --test minimal_spine_replay_e2e -- --nocapture`
  - `cargo test -p ucf-consolidation --test minimal_spine_consolidation_pipeline_e2e -- --nocapture`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Root `out/*.json` reports are current evidence only for the HEAD/run whose embedded metadata matches the evaluated HEAD. They should normally remain uncommitted.
- Workspace evidence stability remains an operational caveat across environments. A timeout, stale report, missing report, dirty-state mismatch, or non-PASS workspace-test report must not be treated as a pass.
- Update this roadmap selection if Minimal Spine v1.x boundaries change, if Gate criteria change, if append/readback authority is explicitly rejected, if protocol schema debt blocks append contracts, if runtime scheduler work is selected first by explicit policy, or if identity/ISM authority semantics are explicitly selected by a later roadmap prompt.

## 9. Next Prompt

Recommended next prompt title: **UCF Prompt 64 — Evidence/Archive Append Contracts Roadmap and Boundary Audit**.

Reason: Prompt 64 keeps the first post-Geist step planning-only while defining authority, append/readback boundaries, test requirements, and forbidden claims before any Replay, Sleep, Geist, or ISM candidate record can gain explicit Evidence/Archive append/readback support.

## Prompt 64 Roadmap Audit Note

Evidence/Archive append contract planning is now available as [`docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`](evidence_archive_append_contracts_roadmap_boundary_audit.md). The audit is roadmap/boundary-only: it does not implement Replay, Sleep, Geist/ISM, runtime, identity, Gateway, capability, real-compute, Evidence/Archive-authority, second-event-log, gate-criteria, or Minimal Spine v1.x behavior changes. It recommends **UCF Prompt 65 — Replay Evidence/Archive Append Contract** as the next implementation prompt, with explicit readback tests and the existing Evidence/Archive authority preserved.


- Evidence/Archive bounded append/readback closure baseline is available at `docs/roadmap/evidence_archive_append_readback_closure.md` (with workspace-evidence caveat handling).


## Supersession Note
- After bounded Evidence/Archive append/readback closure, `docs/roadmap/post_archive_roadmap_selection.md` supersedes earlier next-line planning in this document.
