# UCF Post-Sleep Roadmap Selection

## 0. Purpose

- Select the next roadmap line after bounded Sleep closure.
- This is a planning and boundary-selection document only.
- No implementation is introduced here.
- This is not Geist/ISM readiness.
- This is not identity finalization.
- This is not an identity anchor.
- This does not activate a runtime scheduler, queue, worker, Sleep runtime coordinator, Replay runtime scheduler, Gateway write API, capability issuance, real compute, or Evidence/Archive append behavior.
- Minimal Spine v1.x remains frozen and unchanged.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `f70378353a60258785a2d56f7cbfe543fd420fbc` |
| HEAD short | `f7037835` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Sleep closure present | yes |
| Replay closure present | yes |
| Consolidation closure present | yes |
| Compute closure present | yes |
| Freeze doc present | yes |
| `ucf-geist` present | yes |
| `ucf-replay` present | yes |
| `ucf-sleep-coordinator` present | yes |

Baseline links:

- [`docs/roadmap/sleep_closure.md`](sleep_closure.md)
- [`docs/roadmap/replay_closure.md`](replay_closure.md)
- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)
- [`docs/current_state_architecture_index.md`](../current_state_architecture_index.md)
- [`docs/module_implementation_depth_registry.md`](../module_implementation_depth_registry.md)

## 2. Candidate Inventory

| Candidate | Relevant paths | Current maturity | Tests present | Boundary risk | Dependency on completed lines | Can remain bounded? | Difficulty | Notes |
|---|---|---|---|---|---|---|---:|---|
| A. Geist/ISM Roadmap and Boundary Audit | `domains/geist/crates/ucf-geist/src/lib.rs`; `docs/current_state_architecture_index.md`; `docs/module_implementation_depth_registry.md`; closure docs for Consolidation, Replay, and Sleep | mixed | `cargo test -p ucf-geist --all-targets`; no dedicated post-Sleep Geist/ISM projection tests yet | critical | High dependency on bounded Consolidation -> Replay -> Sleep because those lines define the inputs that must not be promoted into identity authority | yes, only if first prompts are docs-only, projection-only, read-only, candidate-only, and verify-only | XL | Existing Geist code contains SelfState/ISM-like surfaces and anchor/upsert vocabulary. The next safe step is a boundary audit before any implementation. |
| B. Runtime Replay/Sleep Scheduler / Queue | `runtime/ucf-replay/src/lib.rs`; `core/crates/ucf-sleep-coordinator/src/lib.rs`; Replay and Sleep E2E tests; `.github/workflows/ci.yml` | partial | bounded Replay and Sleep tests exist, but no runtime scheduler/queue/worker tests for these lines | high | Depends directly on completed ReplayToken/ReplaySchedule and SleepPlanCandidate surfaces | yes, but only as deterministic planned runtime design with no worker activation | L | Technically useful, but selecting it now risks hidden Gateway/action/runtime coupling before Geist/ISM and append authority are clarified. |
| C. Evidence/Archive Append Contracts for Replay/Sleep | `core/crates/ucf-evidence/src/lib.rs`; `domains/archive/crates/ucf-archive/src/lib.rs`; `domains/archive/crates/ucf-archive-store/src/lib.rs`; Replay/Sleep audit and boundary records | partial | Evidence and Archive crates have tests through workspace coverage; no explicit Replay/Sleep append/readback contract tests yet | high | Builds on ReplayAudit, ReplayAppliedBoundary, SleepPlanAudit, and SleepAppliedBoundary | yes, if append is explicit, local, deterministic, and does not create a second event-log authority | L | A strong secondary line because it improves auditability and can prepare read-only provenance for later Geist/ISM work. |
| D. Prod-profile / Workspace Evidence Stability | `runtime/ucf-ops`; `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml`; readiness reports under `out/` when generated | mixed | docs lint, readiness-spine, workspace-test-check, readiness-gate, workspace tests, clippy, CI and nightly workflows | medium | Operationally depends on all completed lines because evidence freshness and workspace stability must cover them | yes | M | Important parallel validation line. It improves release confidence without advancing UCF cognitive behavior. |
| E. Protocol Schema / Provenance Evolution | `protocol/crates/ucf-protocol`; `protocol/crates/ucf-protocol/spec`; Consolidation/Replay/Sleep wrapper provenance docs and records | partial | protocol canonical tests and workspace tests exist; no dedicated post-Sleep provenance-evolution plan yet | high | Cross-cuts Consolidation, Replay, Sleep, Evidence, and future Geist/ISM projection inputs | yes, if additive-only, versioned, deterministic, and fixture-aware | XL | Important but broad. It should be deferred unless schema debt blocks the selected Geist/ISM boundary audit. |
| F. Gateway Read API Expansion | Gateway-related docs and ops surfaces; current bounded state records in Consolidation/Replay/Sleep crates | skeleton | no dedicated bounded-state read-only Gateway expansion tests identified in this pass | high | Could read completed bounded states, but should not drive core roadmap selection | yes, if strictly read-only and non-authoritative | L | Useful later for observability. It is not the primary cognition line and could accidentally imply visibility/authority claims. |

## 3. Selection Criteria Score

Scoring uses 0-5 for each criterion. The displayed total is the sum across all ten criteria: freeze safety, natural fit after Consolidation -> Replay -> Sleep, overclaim-risk reduction, architectural truthfulness, future unlock value, small-prompt feasibility, determinism/CI fit, low runtime coupling, authority clarity, and movement toward the original architecture.

| Candidate | Freeze safety | Builds on completed lines | Reduces overclaim risk | Unlocks future work | CI-friendliness | Authority clarity | Strategic value | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Geist/ISM Roadmap and Boundary Audit | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 45 |
| B. Runtime Replay/Sleep Scheduler / Queue | 3 | 4 | 2 | 4 | 3 | 2 | 3 | 31 |
| C. Evidence/Archive Append Contracts for Replay/Sleep | 4 | 5 | 4 | 4 | 4 | 3 | 4 | 40 |
| D. Prod-profile / Workspace Evidence Stability | 5 | 3 | 4 | 3 | 5 | 5 | 3 | 40 |
| E. Protocol Schema / Provenance Evolution | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 38 |
| F. Gateway Read API Expansion | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 29 |

Detailed criterion notes:

| Candidate | Preserves freeze | Natural build | Reduces overclaim | Architectural truthfulness | Future unlock | Small prompts | Deterministic / CI-friendly | Avoids runtime coupling | Avoids authority confusion | Original architecture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 4 | 5 |
| B | 3 | 4 | 2 | 3 | 4 | 3 | 3 | 2 | 2 | 5 |
| C | 4 | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 4 |
| D | 5 | 3 | 4 | 5 | 3 | 4 | 5 | 5 | 5 | 1 |
| E | 4 | 4 | 4 | 5 | 4 | 2 | 3 | 4 | 4 | 4 |
| F | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 |

## 4. Roadmap Decision

| Decision | Selected line | Reason | Risks | Guardrails |
|---|---|---|---|---|
| Primary next line | A. Geist/ISM Roadmap and Boundary Audit | It best uses the completed bounded Consolidation -> Replay -> Sleep chain while reducing future self-state and identity-adjacent overclaim risk before any implementation proceeds. | Critical semantic risk around SelfState, ISM, anchors, stabilization, recursion, and identity-adjacent language. | Start with roadmap/boundary audit only; projection-only/read-only/candidate-only first; no identity anchor, no identity finalization, no ISM write, no Gateway/action authority. |
| Secondary next line | C. Evidence/Archive Append Contracts for Replay/Sleep | It can strengthen auditability and archived provenance before Geist/ISM consumes read-only inputs later. | Could accidentally create a second event-log authority or imply identity meaning for Replay/Sleep records. | Explicit append only; no second event log; no Gateway write; no identity meaning; Evidence/Archive authority unchanged. |
| Parallel validation line | D. Prod-profile / Workspace Evidence Stability | It hardens release and CI evidence without adding UCF behavior. | Can absorb roadmap time without advancing architecture; workspace evidence can vary across environments. | Keep it operational; do not change gate criteria casually; generated reports remain freshness-bound validation artifacts. |
| Deferred line | B. Runtime Replay/Sleep Scheduler / Queue | Runtime scheduling is important but would introduce queue/worker/action coupling before semantic and append boundaries are clearer. | Hidden runtime activation, Gateway/action coupling, background nondeterminism. | No worker activation; no Gateway action; no Geist/ISM write; deterministic tests before runtime behavior. |
| Deferred line | E. Protocol Schema / Provenance Evolution | Provenance/schema evolution is broad and should not block Geist/ISM boundary planning unless concrete schema debt is found. | Large cross-cutting scope, fixture/golden churn, versioning risk. | Additive-only; versioned; deterministic; fixture/golden-aware; no Minimal Spine v1.x weakening. |
| Deferred line | F. Gateway Read API Expansion | Read API expansion can help observability later but is not the next cognition line. | Gateway visibility can be mistaken for authority or readiness. | Read-only only; no write API; no action authority; no production claim. |

## 5. Guardrails for Selected Line

Because the primary selected line is Geist/ISM Roadmap and Boundary Audit, the following guardrails are mandatory:

- First prompt is Roadmap and Boundary Audit only.
- No identity anchor implementation.
- No identity finalization.
- No persistent ISM write in the first implementation prompt.
- No unbounded recursion.
- No policy mutation.
- No Gateway/action authority.
- No hidden promotion from `SleepAppliedBoundary` to identity.
- No hidden promotion from Macro finalization to identity.
- Geist inputs must be projection-only/read-only at first.
- ISM semantics must be candidate/read-model first.
- Any future anchor/upsert requires an explicit authority prompt.
- Negative tests are required before any write behavior.
- Evidence/Archive authority remains unchanged.
- Minimal Spine v1.x remains unchanged.
- Bounded Replay, bounded Consolidation, and bounded Sleep behavior remain unchanged.
- Gate criteria remain unchanged.

Secondary-line guardrails for Evidence/Archive append, if selected later:

- Append behavior must be explicit only.
- It must not create a second event log.
- It must not introduce a Gateway write.
- It must not assign identity meaning to Replay/Sleep records.

Deferred runtime-scheduler guardrails, if selected later:

- No Geist/ISM write.
- No Gateway action.
- No background worker activation before deterministic tests.

## 6. Prompt Series Plan

| Prompt | Title | Goal | Scope | Acceptance criteria | Boundary guardrails |
|---:|---|---|---|---|---|
| 55 | Geist/ISM Roadmap and Boundary Audit | Inventory current Geist/ISM surfaces and define safe next boundaries after Sleep closure. | Docs-only audit and roadmap selection for Geist/ISM. | New or updated Geist/ISM boundary doc; no code changes; explicit mapping from Consolidation/Replay/Sleep inputs to read-only projections. | No identity anchor; no finalization; no ISM write; no runtime activation; no Evidence/Archive authority change. |
| 56 | Geist/ISM Record Authority and Schema Alignment | Clarify record authority, schema ownership, and naming for projection-only Geist/ISM candidates. | Docs/schema planning only unless a later prompt explicitly authorizes code. | Authority table distinguishes SelfState, ISM candidate, Replay/Sleep provenance, Evidence, Archive, Gateway, and policy. | No persistent writes; no anchor/upsert; no protocol breaking changes; no Minimal Spine v1.x change. |
| 57 | Self-State Projection Candidate from Sleep Boundary | Plan or implement a deterministic candidate projection from bounded Sleep provenance only if authorized by Prompt 56. | Candidate/read-model surface; projection-only input digest mapping. | Candidate has deterministic encoding, negative tests for forbidden authority flags, and no store/appender handles. | No identity meaning; no hidden promotion from SleepAppliedBoundary; no Gateway/action authority. |
| 58 | Geist Projection Verify-Only Audit Contract | Define verify-only audit for Geist projection candidates. | Pure verification contract and tests if implementation is authorized. | Audit verifies candidate provenance and rejects identity, anchor, policy mutation, runtime, Gateway, and append flags. | Verify-only; no write behavior; no finalization. |
| 59 | ISM Candidate Boundary Without Identity Finalization | Define an ISM candidate boundary that remains a read-model candidate. | Candidate boundary and negative tests if implementation is authorized. | Boundary record is local/candidate-only and cannot upsert anchors or finalize identity. | No persistent ISM write; no anchor/upsert; explicit future authority prompt required. |
| 60 | Geist/ISM E2E Determinism | Validate deterministic candidate -> audit -> candidate-boundary flow. | Bounded E2E tests only. | Fresh-run determinism, canonical ordering, and stable digests; negative tests for hidden writes. | No runtime scheduler; no Evidence/Archive append; no Gateway visibility; no production readiness claim. |
| 61 | Geist/ISM Docs Overclaim Guard | Remove or qualify overclaims in docs after bounded Geist/ISM planning/implementation. | Docs-only cleanup. | Docs distinguish projection candidates from identity finalization, ISM persistence, Sleep completion, and production readiness. | No readiness claim; no anchor/finalization claim. |
| 62 | Geist/ISM Readiness Refresh | Refresh validation evidence for the bounded Geist/ISM line. | Validation and docs evidence only. | Fresh docs lint, readiness-spine, targeted Geist/ISM tests, Replay/Sleep/Consolidation regression tests, workspace tests where practical, fmt, clippy. | Generated `out/*.json` not committed unless explicitly required; stale reports cannot support readiness. |
| 63 | Post-Geist Roadmap Selection: Runtime Scheduler vs Evidence Append vs Prod-Profile | Select the next line after bounded Geist/ISM candidate work. | Roadmap selection only. | Clear primary/secondary/parallel/deferred choices based on actual Geist/ISM closure evidence. | No runtime activation or Gateway/API write unless explicitly selected and bounded. |

## 7. Deferred Lines

- Runtime Replay/Sleep Scheduler / Queue is deferred because bounded Replay and bounded Sleep currently stop at deterministic local records. Scheduler, queue, worker, and runtime coordinator behavior would introduce action coupling and possible nondeterminism before Geist/ISM and append authority boundaries are clear.
- Evidence/Archive Append Contracts for Replay/Sleep is secondary, not primary, because append/readback can improve auditability but still needs strict authority language to avoid creating a second event log or identity interpretation. It may move ahead of implementation prompts if Geist/ISM needs archived read-only provenance first.
- Prod-profile / Workspace Evidence Stability should run in parallel because it hardens CI and release evidence without adding feature behavior. It should not be treated as a cognition-line substitute.
- Protocol Schema / Provenance Evolution is deferred unless the Geist/ISM boundary audit identifies blocking schema debt. Any schema work must be additive-only, versioned, deterministic, and fixture/golden-aware.
- Gateway Read API Expansion is deferred because bounded states can remain internal until read-only exposure is explicitly justified. Gateway visibility must never imply write/action authority or production readiness.

## 8. Revalidation Rules

Before each prompt in the selected line:

- Capture `pwd`, branch, `git status --short`, full and short HEAD, recent log, workspace package count, and relevant file-presence checks.
- If `git status --short` is non-empty at baseline, stop and report the dirty paths before making changes.
- For docs-only prompts, run at minimum:
  - `cargo fmt --check`
  - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
  - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
  - targeted package tests for affected crates when practical
  - `git diff --check`
- Before closing a multi-prompt Geist/ISM planning block, attempt:
  - `timeout 600s cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`
  - `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --workspace-test-report ./out/workspace_test_report.json --out ./out/gate_report.json` only when the workspace-test report exists and is fresh
  - `cargo test -p ucf-sleep-coordinator --test minimal_spine_sleep_e2e -- --nocapture`
  - `cargo test -p ucf-replay --test minimal_spine_replay_e2e -- --nocapture`
  - `cargo test -p ucf-geist --all-targets`
  - `cargo test -p ucf-consolidation --test minimal_spine_consolidation_pipeline_e2e -- --nocapture`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Root `out/*.json` reports are current evidence only for the HEAD/run whose embedded metadata matches the evaluated HEAD. They should normally remain uncommitted.
- Workspace evidence stability remains an operational caveat across environments. A timeout or missing workspace-test report must not be treated as a pass.
- Update this roadmap selection if Geist/ISM implementation is authorized, if Evidence/Archive append contracts become a prerequisite, if Runtime Scheduler/Queue is explicitly selected, if Minimal Spine v1.x boundaries change, if CI/gate criteria change, or if protocol schema debt blocks the selected line.

## 9. Next Prompt

Recommended next prompt title: **UCF Prompt 55 — Geist/ISM Roadmap and Boundary Audit**.

Reason: bounded Consolidation, bounded Replay, and bounded Sleep now provide a safer basis for a self-state and ISM-adjacent boundary audit, but the next step must remain docs-only and must not claim Geist/ISM readiness, identity finalization, identity anchoring, runtime activation, Gateway authority, or Evidence/Archive append behavior.
