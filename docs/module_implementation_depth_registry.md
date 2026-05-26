# UCF Module Implementation-Depth Registry

## 0. Purpose

Minimal UCF Spine v1 is specified at `docs/minimal_ucf_spine_v1.md`; Prompt 5 must follow that spec.

- This document classifies the implementation depth of the most important UCF modules from the current code, tests, feature lanes, integration surfaces, and operational gates.
- It is the input document for Minimal Spine planning, roadmap sequencing, and the prompt series after the current-state architecture index.
- It does not replace source-code inspection, generated spec snapshots, or CI/readiness reports.
- It intentionally avoids feature implementation, production-code refactors, architectural rewrites, and historical-doc deletion.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `92a353e4e997103ba1979bf3c02542c07fa749c0` |
| HEAD short | `92a353e4` |
| Dirty state at audit start | clean |
| Workspace package count | 192 |
| Cargo.toml count | 280 |
| Test file count | 113 Rust files under `*/tests/` outside `target/` |
| Architecture index | `docs/current_state_architecture_index.md` |

Baseline commands: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -10`, `cargo metadata --no-deps --format-version 1`, `find . -name Cargo.toml -not -path "./target/*" | sort`, `find core domains runtime crates protocol ucf-sdk chip4 chip-3 adapters app -maxdepth 4 -type f \( -name "Cargo.toml" -o -name "README.md" -o -name "lib.rs" -o -name "main.rs" \) 2>/dev/null | sort`, `find . -path "*/tests/*" -type f -name "*.rs" -not -path "./target/*" | sort`, and marker search with `rg` over current code/doc roots.

## 2. Classification Rules

### Implementation-depth categories

- `docs-only`: only documentation exists; no load-bearing code path was found.
- `skeleton`: crate/module exists, but is mostly types, placeholders, or structure.
- `stub`: intentionally simplified implementation, not a real production path.
- `mock`: test, simulation, or mock path; not a real production path.
- `toy`: small working demo or toy backend, but not production-near.
- `partial`: substantial parts exist, but integration, boundaries, tests, or production path are missing.
- `functional-prototype`: works for defined use-cases and has tests, but is not production-hard.
- `production-leaning`: relatively stable, tested, integrated, has a clear API/boundary, and is visible in CI/gates.
- `deferred`: intentionally postponed; not currently active.
- `historical`: audit/legacy/context material; not the current implementation base.
- `vendor-only`: vendored or external reference; not primary UCF implementation.

### Role categories

- `spine-core`
- `spine-supporting`
- `operational`
- `code-near-spec`
- `advisory-boundary`
- `research-experimental`
- `integration-surface`
- `peripheral-adapter`
- `vendor-reference`
- `historical-context`
- `unknown`

### Current-status categories

- `current-core`
- `current-supporting`
- `current-operational`
- `code-near-spec`
- `partial-prototype`
- `advisory-only`
- `experimental`
- `deferred`
- `historical`
- `vendor-only`
- `unknown`

### Evidence rules

1. Code, tests, and gates outrank docs.
2. Feature existence alone does not imply implementation maturity.
3. Historical, deferred, advisory, and vendor docs are not current implementation evidence.
4. In-memory stores, toy compute lanes, mock producers, and diagnostic-only Blue-Brain surfaces must not be represented as production persistence, real compute, or complete E2E cognition.
5. A module can be eligible for Minimal Spine even if not production-leaning, but only when the included path is deterministic, testable offline, bounded, and explicit about missing pieces.

## 3. Workspace Group Summary

| Group | Package count | Main purpose | Maturity spread | Notes |
|---|---:|---|---|---|
| `core` | 55 | Shared primitives, bus, routing, evidence, policy ecology, ports, cognitive support crates. | `skeleton` to `production-leaning` | Strongest Minimal Spine candidates live here, but many AI/neuro ports remain partial or feature-bound. |
| `domains` | 29 | Archive, frames, ESS, Blue-Brain bridge, consolidation, Geist, neuromod, index, policy gateway. | `partial` to `functional-prototype`; advisory surfaces present | Archive/frames/ESS are more concrete than Blue-Brain and neuromod claims. |
| `runtime` | 12 | Gateway/client/ops/runtime/replay/compute/platform/bench. | `stub`/`toy` to `production-leaning` | Ops and gateway are gate-visible; compute has explicit stub/toy/backend lanes. |
| `crates` | 81 | DBM, biophys, microcircuit, replay executor/evidence, assets, scorecards, PVGS support. | `stub`, `partial`, `experimental` | Many region/microcircuit crates are research/deepening support, not Minimal Spine. |
| `protocol` | 1 | Protocol v1 Rust types and code-near specs. | `production-leaning` | Small, tested, and central for canonical boundary messages. |
| `ucf-sdk` | 1 | SDK record builders/types for external consumers. | `functional-prototype` | Useful boundary layer, but narrower than protocol/types authority. |
| `adapters` | 2 | Terminal and rig adapter surfaces. | `partial` | Peripheral adapters; not required for Minimal Spine. |
| `app` | 1 | Top-level app binary wiring DBM/profile/protocol pieces. | `partial-prototype` | Not spine-core; keep out of first canonical E2E unless explicitly scoped. |
| `chip4` | 1 | Local chip/PVGS-facing library. | `partial` | Related to chip/PVGS lane, not first spine. |
| `vendor` | 2 | Vendored Firewood/RPP reference crates. | `vendor-only` | Reference material even when workspace-visible. |
| `other` | 7 | AI runtime, engine, HPA, profiles, RSV, PVGS client, wire. | `partial` to `functional-prototype` | Supporting or integration-facing, but not first-spine default. |

## 4. Core Registry

| Module / Crate | Path | Role | Current status | Implementation depth | Tests | CI/Gate visibility | Feature-gated? | Minimal Spine candidate? | Main evidence | Main gap | Roadmap action |
|---|---|---|---|---|---|---|---|---:|---|---|---|
| `ucf-types` | `core/crates/ucf-types` | `spine-core` | `current-core` | `production-leaning` | Unit tests in crate | Workspace tests/checks | Optional serde derives | yes | Large shared type surface with deterministic identifiers, digests, policy, spike, and record structs. | Must decide whether it or `ucf-protocol` is schema authority for Minimal Spine. | Include; freeze only the subset needed by Spine v1. |
| `ucf-protocol` | `protocol/crates/ucf-protocol` | `code-near-spec` | `code-near-spec` | `production-leaning` | Unit and integration tests | Workspace tests, docs/spec lint | No notable feature lane | yes | Protocol v1 message/spec code plus boundary/canonical tests and textual specs, including minimal CandidateSetRecord/OutputRecord commitments for Spine v1. | Keep capability issuance and broader output semantics out until explicit policy/gateway specs require them; Minimal Spine v1.5 keeps `CapabilityIssuanceRecord` deferred and does not add an inert schema. | Include as canonical wire/boundary schema authority. |
| `ucf-sdk` | `ucf-sdk` | `integration-surface` | `current-supporting` | `functional-prototype` | Integration determinism tests | Workspace tests | `serde` feature | optional | Public SDK records wrap UCF type IDs and deterministic serialization checks. | Narrow surface; not the whole protocol authority. | Include only if Prompt 4 needs external consumer examples. |
| `ucf-bus` | `core/crates/ucf-bus` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | No | yes | In-memory publish/subscribe bus with envelope type and tests. | In-memory only; no durability/backpressure/multi-process semantics. | Include as deterministic in-process bus; label non-production transport. |
| `ucf-evidence` | `core/crates/ucf-evidence` | `spine-core` | `current-core` | `functional-prototype` | Unit tests, file-store tests | Workspace tests | No | yes | Evidence envelopes, in-memory store, append-log abstraction, and file-store module. | Store trait default `get` is non-load-bearing unless implemented; production retention/compaction absent. | Include with file/in-memory constraints explicit. |
| `ucf-fold` | `core/crates/ucf-fold` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | No | optional | Deterministic fold proof/state helper used by archive/evidence. | Dummy folder semantics are not a final proof system. | Optional for Spine v1 if evidence fold proof is in scope. |
| `ucf-archive` | `domains/archive/crates/ucf-archive` | `spine-core` | `current-core` | `functional-prototype` | Unit tests | Workspace tests | `firewood` backend conditional | yes | File archive, manifest/hash checks, fold snapshot support, in-memory/file store exports. | Firewood backend is feature-dependent; production store selection unresolved. | Include as canonical append/archive candidate with file backend. |
| `ucf-archive-store` | `domains/archive/crates/ucf-archive-store` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | No | yes | Store adapter for archive/evidence records used by router/runtime surfaces. | Scope overlaps with archive; authority must be simplified for Spine. | Include only one persistence boundary in Prompt 4, with this as adapter if needed. |
| `ucf-router` | `core/crates/ucf-router` | `spine-core` | `current-core` | `functional-prototype` | Unit/integration target present | Workspace tests | `mock-spike-producers` | yes | Router crate integrates AI, archive, policy, sandbox, Blue-Brain ports, and route tests. | Feature/mock lanes can overstate real producer maturity; boundary is broad. | Include a minimal deterministic route, not all feature lanes. |
| `ucf-runtime` | `runtime/ucf-runtime` | `integration-surface` | `partial-prototype` | `partial` | Unit/integration tests | Workspace tests/checks | Many compute/backend/GPU/LFM/LLM features | optional | Runtime binary/lib wires compute, policy, replay, frames, ESS, Blue-Brain bridge, DBM. | Too broad for first spine; features imply optional integrations, not maturity. | Use only if Minimal Spine host is chosen here; otherwise keep as later integration. |
| `ucf-frames` | `domains/ucf-frames` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit/integration target present | Workspace tests | No | optional | Frame model used by ESS, policy/runtime/gateway. | Needs explicit subset for spine input/output frames. | Include if Spine v1 includes frame-level records. |
| `ucf-ess` | `domains/ucf-ess` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit/integration target present, including minimal-spine read-model tests | Workspace tests | No | optional | Experience/state store concepts, record use, tests, and a crate-local Minimal Spine derived read model keyed by EvidenceId, OutputRecord digest, and archive output key. | ESS is not a production authoritative store and must remain subordinate to evidence/archive/protocol authority. | Keep as optional derived read model; do not promote to canonical append authority without a later spec. |
| `ucf-policy-ecology` | `core/crates/ucf-policy-ecology` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | `commit`/`ucf-commit` features | yes | Bounded policy ecology surfaces: PolicyFieldV1 read-only snapshot, PolicyConstraintV1 typing, PolicyEvaluationCandidateV1 candidate-only evaluation, and PolicyVerifyAuditV1 verify-only audit with deterministic digest. | This crate does not provide runtime enforcement, action approval, gateway authority, policy mutation, identity authority, or Evidence/Archive append authority. | Keep policy claims bounded to read-only/evaluation/audit semantics only. |
| `runtime/ucf-policy` | `runtime/ucf-policy` | `spine-supporting` | `current-supporting` | `functional-prototype` | Integration tests | Workspace tests | No | optional | Runtime policy layer includes broader capability/governor decision helpers for non-spine runtime flows. | Runtime policy capability decisions are not Minimal Spine authority; no Gateway write, self-grant, policy mutation, or compute-triggered capability grant is in scope for v1.x. | Keep optional; future active capability issuance needs a dedicated safety prompt and negative tests. |
| `policies/packs/*` | `policies/` | `operational` | `current-operational` | `functional-prototype` | Validated by ops commands | Readiness/policy validation gates | Profile overlays | yes | Base/test/dev/prod overlays are gate inputs. | Policy immutability claims depend on validation and manifests, not docs alone. | Include base+test policy pack as data dependency. |
| `ucf-consolidation` | `domains/consolidation/crates/ucf-consolidation` | `spine-supporting` for the bounded Minimal Spine consolidation path; `research-experimental` for broader kernel/runtime paths | `partial-prototype` | `partial` | Unit tests plus Minimal Spine micro hook, micro append, meso aggregation/append, macro candidate, local consolidation finalization, and E2E determinism tests | Workspace tests | No | optional | Consolidation now exposes a deterministic micro builder and explicit micro append/readback, deterministic meso aggregation and explicit meso append/readback, a macro candidate, and a local consolidation-level finalization boundary; broader kernel paths still touch archive, bus, commit, consistency, IIT, influence, macro-finalized publication, sleep, and replay. | The bounded path does not prove full memory readiness, Replay Scheduler readiness, sleep-cycle readiness, Geist/ISM integration, identity finalization/anchor, Gateway-visible consolidation, capability relevance, real compute activation, production consolidation, or a second event log. | Keep the bounded path as a deterministic derived consolidation line; use `docs/roadmap/full_consolidation_roadmap_boundary_audit.md` for the overclaim guard and defer Replay/Sleep/Geist/ISM/production claims to later prompts. |
| `ucf-replay` | `runtime/ucf-replay` | `operational` | `current-supporting` | `functional-prototype` | Golden replay tests, Prompt 37 record-authority alignment docs, and Prompt 38-42 bounded Token → Schedule → Audit → AppliedBoundary tests | Workspace tests | Backend argument selects stub/candle/burn | optional | Deterministic replay fixture/report path and audit-only `ReplayPlan`/`ReplayReport`/`ReplayResult` surfaces; current bounded replay adds `ReplayToken` intent/reference, `ReplaySchedule` planned ordering, verify-only audit, local replay-subsystem applied boundary, and deterministic E2E proof. | Golden fixture uses stub backend; bounded Replay now includes Prompt 65 Evidence/Archive append/readback as audit/provenance persistence only. It is not proof of real compute, runtime Replay Scheduler readiness, runtime replay apply/execution, scheduler queue/worker, Sleep/Geist/ISM runtime integration, memory mutation, Gateway-visible replay, production replay, identity finalization/anchor, or a second event log. | Optional validation lane for Minimal Spine; use `docs/roadmap/replay_record_authority_schema_alignment.md` Sections 0.1/0.2 and the `runtime/ucf-replay/tests/minimal_spine_replay_*` tests before making replay claims. |
| `ucf-sleep-coordinator` | `core/crates/ucf-sleep-coordinator` | `spine-supporting` for bounded local Sleep planning; `research-experimental` for existing trigger/report runner wiring | `partial-prototype` | `functional-prototype` | In-source unit tests plus bounded SleepPlan candidate, verify-only audit, local applied-boundary, and E2E determinism tests | Workspace tests | No | later | Existing WAL-style sleep trigger state, replay summary counters, and RSA sleep phase runner adapter are inventory/prototype surfaces; the current bounded line adds `MinimalSpineSleepPlanCandidate`, verify-only `MinimalSpineSleepPlanAudit`, local-only `MinimalSpineSleepAppliedBoundary`, and deterministic Replay-derived Sleep E2E tests. | Existing runnable trigger/report paths can append sleep reports through RSA/archive prototypes and must not be read as post-Replay Sleep v1 authority; bounded Sleep now includes Prompt 66 Evidence/Archive append/readback as audit/provenance persistence only. Bounded Sleep surfaces still do not prove Sleep runtime, Sleep Cycle Coordinator activation, coordinator trigger/report/WAL/journal, SleepCompleted, Geist/ISM runtime integration, identity stabilization/finalization/anchor, memory stabilization, Gateway visibility, production readiness, runtime scheduler/queue/worker readiness, or a second event log. | Use `docs/roadmap/sleep_integration_roadmap_boundary_audit.md` and `docs/roadmap/sleep_record_authority_schema_alignment.md` before making Sleep claims; current allowed claims are candidate-only builder, verify-only audit, local-only boundary, and bounded E2E determinism from bounded Replay metadata. |
| `replay_executor` | `crates/replay_executor` | `research-experimental` | `experimental` | `partial` | Unit tests not evident in `tests/` | Workspace compile/tests if member | Region/microcircuit features | later | Coordinates microcircuit stub/L4 crates. | Depends on stub microcircuits and feature-heavy deepening lanes. | Defer; use only as research replay lane. |
| `replay_evidence` | `crates/replay_evidence` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | No | optional | Evidence helpers for replay reports. | Secondary to core evidence/archive. | Optional; include only if replay is in Spine v1. |
| `ucf-geist` | `domains/geist/crates/ucf-geist` | `research-experimental` for broad kernel/store surfaces; `spine-supporting` for the bounded local Geist/ISM candidate chain | `partial-prototype` | `functional-prototype` for bounded candidate/audit/boundary helpers | Unit tests plus Prompt 57-60 bounded candidate-only projection, verify-only audit, local ISM candidate boundary, and E2E determinism tests | Workspace tests | No | later | Geist SelfState, loop-state, ISM-store, policy-gate, sleep-coupling, and archive-append prototype surfaces exist; the current bounded line adds `MinimalSpineGeistProjectionCandidate` candidate-only, `MinimalSpineGeistProjectionAudit` verify-only, `MinimalSpineIsmCandidateBoundary` local read-model/candidate-only, and deterministic Sleep-derived Geist/ISM E2E coverage. | Existing kernel/upsert/archive/sleep mutation surfaces are too broad for bounded post-Sleep Geist/ISM; bounded Geist/ISM now includes Prompt 67 Evidence/Archive append/readback as audit/provenance persistence only and Prompt 68 cross-layer readback E2E. Current bounded tests do not prove runtime Geist, `GeistApplied`, ISM write/upsert, `IsmStore::upsert_anchor`, IdentityAnchor, IdentityFinalization, memory stabilization, persistent self authority, Policy mutation, Gateway/action authority, production readiness, unbounded recursion, or a second event log. | Use `docs/roadmap/geist_ism_roadmap_boundary_audit.md`, `docs/roadmap/geist_ism_record_authority_schema_alignment.md`, and `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`; allowed claims are only candidate-only projection, verify-only audit, local read-model/candidate boundary, bounded Geist/ISM E2E determinism, and audit/provenance append/readback until a later explicit authority prompt. |
| `ucf-recursion-controller` | `core/crates/ucf-recursion-controller` | `spine-supporting` | `current-supporting` | `functional-prototype` | Unit tests | Workspace tests | No | optional | Bounded recursion control types used by Geist/output router. | Needs clear spine trigger semantics. | Optional guard if recursive loop is in scope. |
| `ucf-sle` | `core/crates/ucf-sle` | `research-experimental` | `experimental` | `partial` | Unit tests if member | Workspace tests | No | later | Sleep/learning/evaluation naming surface. | Not verified as Minimal Spine dependency. | Defer. |
| `ucf-neuromod` | `domains/ucf-neuromod` | `spine-supporting` for the minimal envelope hook; `research-experimental` for broader scheduler/state APIs | `partial-prototype` | `partial` | Unit tests plus Minimal Spine envelope hook integration test and targeted `HormoneStateV1` + `hormone_update_v1` tests | Workspace tests | No | optional | Neuromod now exposes a deterministic `MinimalSpineNeuromodEnvelope`, bounded fixed-point `HormoneStateV1`/`NormalizedHormoneLevelV1`, and deterministic bounded `update_hormone_state_v1(prev,input,config)` with advisory-only `HormoneModulationOutputV1`; existing v0 field/scheduler APIs remain broader experimental surfaces. | Replay/sleep/geist mapping and runtime integration are still deferred; no runtime authority was added in M3. | Keep as bounded deterministic contract + update lane; M4 completed deterministic advisory mapping hardening with targeted modulation tests, M5 added deterministic bounded replay/sleep candidate-only mapping (`domains/ucf-neuromod/src/replay_sleep_candidate_v1.rs`) with targeted boundary tests (`domains/ucf-neuromod/tests/replay_sleep_candidate_v1.rs`), and M6 added verify-only metabolic audit (`domains/ucf-neuromod/src/metabolic_audit_v1.rs`) with deterministic digest and stable PASS/FAIL failure reporting. M8 now provides closure baseline documentation in `docs/roadmap/metabolic_hormone_control_layer_closure.md`; scheduler/replay/sleep/geist/identity/archive authorities remain deferred. |
| `hpa` / `config/hpa.yaml` | `hpa`, `config/` | `research-experimental` | `experimental` | `partial` | Unit tests not central | Workspace tests for crate | No | no | HPA config/module support exists. | Not central to deterministic Minimal Spine. | Keep as later physiology/regulator lane. |
| `dbm_*` crates | `crates/dbm_*`, `domains/ucf-dbm` | `research-experimental` | `experimental` | `partial` | Mixed unit tests | Workspace tests/checks | No | later | Many anatomical/DBM modules compile in workspace. | Region completeness and biological claims are not production evidence. | Keep out of first spine; classify per region when needed. |
| `ucf-compute` | `runtime/ucf-compute` | `research-experimental` | `partial-prototype` | `stub fixture` / `toy golden` / `optional-real compile-only` by backend lane | Unit tests and CLI | Workspace tests; ops feature checks | `backend-stub`, `backend-toy`, `backend-burn`, `backend-candle` | optional | Explicit stub fixture, toy golden, optional-real compile-only, and remote/external compile-only feature lanes plus diagnostics. | No mandatory real ML backend; optional-real runtime is deferred until local artifact-backed fixture evidence exists; production claim forbidden for current lanes. | Exclude from required Spine v1; add optional deterministic stub fixture lane only if needed. |
| `ucf-ai-port` | `core/crates/ucf-ai-port` | `integration-surface` | `partial-prototype` | `partial` | Unit tests | Workspace tests | Many backend/port features | optional | AI port trait/functions and feature-gated backend bridges. | Ports are broader than implemented production backends. | Keep optional; define narrow trait if Prompt 4 needs compute hook. |
| `domains/ai-backends` | `domains/ai-backends` | `research-experimental` | `experimental` | `partial` | Mixed | Workspace compile if member | Backend-specific | later | Backend crates exist for AI lanes. | Not canonical, not mandatory, not production-proven. | Defer real-backend integration. |
| Burn/Candle/LFM/LNN/LLM/JEPA/SAE/SSM/NSR lanes | `runtime/ucf-compute`, `core/crates/*` | `research-experimental` | `experimental` | `partial` | Mixed unit tests | Workspace tests/features where enabled | Yes | later | Feature flags and crates exist for model families and symbolic/neural lanes. | Feature presence does not prove real production inference/training. | Keep as optional feature lanes after Minimal Spine. |
| `ucf-bluebrain-bridge` | `domains/ucf-bluebrain-bridge` | `advisory-boundary` | `advisory-only` | `functional-prototype` for boundary diagnostics | Unit tests | Workspace tests/docs checks | No | no | Bridge maps brainbus/frames/ESS/core but docs mark Blue-Brain authority caveats. | Blue-Brain completion/closure docs can overclaim production maturity. | Exclude from Spine v1; keep as advisory bridge. |
| `ucf-biophys` / `biophys_*` | `domains/ucf-biophys`, `crates/biophys_*` | `research-experimental` | `experimental` | `partial` | Mixed | Workspace tests/checks | Asset/features vary | later | Biophysical helper crates and assets compile. | Biological fidelity and real compute integration not proven. | Defer until hardening tests define claims. |
| `microcircuit_*_stub` | `crates/microcircuit_*_stub` | `research-experimental` | `experimental` | `stub` | Mixed | Workspace compile/tests | No | no | Stub microcircuits are explicitly named and used by replay executor. | Stub names can be mistaken for real microcircuits. | Keep out of spine; document as simulation/stub only. |
| `microcircuit_*_l4` | `crates/microcircuit_*_l4` | `research-experimental` | `experimental` | `partial` | Mixed | Workspace compile/tests | Biophys asset features | later | L4 microcircuit lanes exist behind features. | Not Minimal Spine and not production validated. | Later research hardening. |
| `ucf-gateway` | `runtime/ucf-gateway` | `integration-surface` | `current-supporting` | `production-leaning` | Integration tests | Workspace tests; ops can gate docs/reports | No | optional | Local-only transport, schemas, auth/rate/error handling, tests; Minimal Spine v1.1 internal read-only evidence/archive surface. | API inclusion may enlarge first spine; write/read scope must stay fixed; v1.1 read service is not production Gateway hardening. | Keep Minimal Spine surface read-only unless a later spec explicitly promotes transport/security scope. |
| `ucf-client` | `runtime/ucf-client` | `peripheral-adapter` | `current-supporting` | `functional-prototype` | Integration/smoke tests | Workspace tests | `gateway-smoke` | optional | Local client endpoint parsing and request handling. | Depends on gateway choice; smoke feature is optional. | Include only with gateway. |
| `ucf-console` | `runtime/ucf-console` | `peripheral-adapter` | `current-supporting` | `functional-prototype` | Tests | Workspace tests | No | no | Console uses client/ops. | UI/console not needed for canonical E2E. | Defer. |
| `runtime/ucf-ops` | `runtime/ucf-ops` | `operational` | `current-operational` | `production-leaning` | Extensive integration tests | Canonical docs/readiness/adversarial/golden/drift gates | Backend/formal features | yes | Implements docs lint, readiness gate, policy validation, reports, drift, goldens, adversarial flows. | Large operational surface; report freshness must be interpreted per HEAD. | Include as validation authority, not runtime cognitive module. |
| `scripts/` and `out/` reports | `scripts`, `out` | `operational` | `current-operational` | `functional-prototype` | Exercised by ops/gates | Generated artifacts/gates | N/A | optional | Artifact convention and report generators support audit trails. | Root reports can go stale when HEAD changes. | Do not commit fresh `out/*.json` unless policy requires. |
| `chip4` | `chip4` | `integration-surface` | `partial-prototype` | `partial` | Unit tests if member | Workspace tests | No | later | Chip4 library connects to protocol/PVGS lane. | Not required for minimal deterministic UCF E2E. | Defer to chip/PVGS prompt. |
| `chip-3` | `chip-3` | `historical-context` | `historical` | `historical` | Not primary workspace evidence | Historical/reference only | N/A | no | Legacy chip directory exists outside current first-spine surface. | Could be mistaken for active implementation. | Keep historical unless explicitly revived. |
| `pvgs`, `pvgs_client`, scorecard/query crates | `crates/pvgs*`, `pvgs_client`, `crates/*score*`, `crates/*query*` | `integration-surface` | `partial-prototype` | `partial` | Mixed | Workspace tests/checks | Chip/local features | later | PVGS/query/scorecard support exists. | Integration semantics not needed for first spine. | Later lane after spine authority is fixed. |
| `vendor/ucf-chip-4-main/vendor/*` | `vendor/` | `vendor-reference` | `vendor-only` | `vendor-only` | Vendor tests not UCF authority | Workspace-visible vendor crates only | Vendor features | no | Firewood/RPP vendored references are present. | External reference can be misread as UCF production store. | Keep vendor-only; do not use as maturity evidence. |
| `ucf-terminal` | `adapters/terminal/crates/ucf-terminal` | `peripheral-adapter` | `current-supporting` | `partial` | Unit tests if member | Workspace tests | No | no | Terminal adapter connects bus/events/protocol/sandbox/types. | Peripheral to spine. | Defer. |
| `ucf-rig` | `adapters/rig/crates/ucf-rig` | `peripheral-adapter` | `partial-prototype` | `partial` | Unit tests if member | Workspace tests | `digitalbrain` | no | Rig adapter connects bus and digitalbrain port. | Hardware/rig assumptions outside Minimal Spine. | Defer. |
| `app` | `app` | `peripheral-adapter` | `partial-prototype` | `partial` | App compile path | Workspace tests/checks | No | no | App binary wires engine/profile/protocol/wire crates. | Not canonical spine host without spec decision. | Defer. |
| Historical/deferred docs | `docs/*historical*`, `docs/*deferred*`, closure/readiness/sweep docs | `historical-context` | `historical` / `deferred` | `historical` / `deferred` | Some docs are checked by ops tests | Docs lint/readiness docs checks | N/A | no | Current-state index separates historical/deferred/advisory docs from current truth. | Titles like final/closure/readiness can overclaim. | Keep for audit trail; do not cite as implementation maturity. |

## 5. Stub / Mock / Toy / Deferred Taxonomy

| Group | Paths | Marker evidence | Intended meaning | Risk if misread | Registry label |
|---|---|---|---|---|---|
| Compute stub/toy lanes | `runtime/ucf-compute` | Cargo features `backend-stub`, `backend-toy`; replay CLI accepts `--backend stub`; fixtures use `stub`. | Deterministic/offline backend lanes for tests and demos. | Could be presented as real ML inference. | `stub`, `toy`, `partial-prototype` |
| Mock spike producers | `core/crates/ucf-router` | Feature `mock-spike-producers`. | Test/mock producer lane for routing. | Could overstate real sensor/spike integration. | `mock` |
| Microcircuit stubs | `crates/microcircuit_*_stub`, `crates/replay_executor` | Stub crates imported by replay executor. | Region-level stub behavior for replay/deepening. | Could be mistaken for biologically complete microcircuits. | `stub`, `experimental` |
| Blue-Brain advisory-only docs/code | `domains/ucf-bluebrain-bridge`, `runtime/ucf-compute/src/blue_brain_*`, `docs/blue_brain_*` | Advisory-only, diagnostic-only, ignored/unavailable/non-executing markers. | Boundary diagnostics and authority mapping, not full Blue-Brain production execution. | Completion/closure docs could be read as production completion. | `advisory-only`, `functional-prototype` for boundary diagnostics |
| HH deferred/simulation pilot | `docs/*hh*`, `runtime/ucf-compute/src/blue_brain_dynamics.rs` | Simulation-only/diagnostic-only and ignored feedback markers. | HH stays diagnostic/simulation-only unless explicitly promoted. | Could imply active HH execution in runtime selection. | `deferred`, `advisory-only` |
| Third deepening deferred/closed docs | `docs/*deepening*`, closure/final/readiness docs | Deferred/closed/final titles with current-state caveats. | Historical planning trail and closed decisions. | Could be used as current implementation proof. | `historical`, `deferred` |
| Vendor chip repos | `vendor/ucf-chip-4-main/vendor/*` | Vendored Firewood/RPP Cargo manifests. | External/reference implementation material. | Could be counted as UCF primary production path. | `vendor-only` |
| Historical Blue-Brain audit baselines | `out/blue_brain_audit_baseline_*` | Historical output naming and freshness requirements. | Audit snapshots for specific HEADs/runs. | Stale reports can be mistaken for current proof. | `historical` unless refreshed for HEAD |
| In-memory stores | `core/crates/ucf-bus`, `core/crates/ucf-evidence`, `domains/archive/crates/ucf-archive` | `InMemoryBus`, `InMemoryEvidenceStore`, `InMemoryArchiveStore`. | Deterministic local test/prototype storage and transport. | Could be treated as durable production persistence. | `functional-prototype`; non-production transport/store |
| Readiness-gate test-profile skips | `runtime/ucf-ops`, `docs/readiness_gate.md` | Readiness profile and test-profile behavior. | Keep gates runnable offline and bounded. | A skipped/relaxed test profile can be misread as prod readiness. | `current-operational` with profile caveat |
| Ignored tests | `runtime/ucf-ops/tests/release_build_rc.rs` | `#[ignore = "manual release flow smoke; runs cargo build --release"]`. | Manual release build smoke is excluded from default test execution because it runs a release build. | A passing default test run does not prove the manual release-build smoke path. | `current-operational` with manual-test caveat |

## 6. Minimal Spine Eligibility Matrix

| Module | Eligible now? | Required for Spine? | Why / Why not | Required hardening before inclusion |
|---|---:|---:|---|---|
| `ucf-types` | yes | yes | Deterministic core IDs/records; no external service requirement. | Decide schema-authority boundary with `ucf-protocol`. |
| `ucf-protocol` | yes | yes | Canonical v1 boundary types/specs and tests. | Decide whether it owns Minimal Spine schema over `ucf-types`. |
| `ucf-sdk` | optional | no | Useful external API, but not necessary for internal spine. | Keep SDK subset aligned with chosen protocol/types authority. |
| `ucf-bus` | yes | yes | In-process deterministic transport suitable for canonical E2E. | Explicitly label in-memory, no production durability/backpressure. |
| `ucf-evidence` | yes | yes | Evidence append/get path and file-store support. | Fix exact store implementation and evidence envelope subset. |
| `ucf-archive` | yes | yes | Append archive/manifest/hash path can persist evidence offline. | Select file backend and document Firewood as optional. |
| `ucf-archive-store` | yes | optional | Useful adapter for archive records. | Avoid duplicate persistence authority with `ucf-archive`/ESS. |
| `ucf-router` | yes | yes | Can host a bounded deterministic route. | Disable/mock-label feature producers; narrow route contract. |
| `ucf-runtime` | optional | no | Broad integration host, but too feature-heavy for first spine by default. | Define a minimal runtime binary path or keep router-hosted E2E. |
| `ucf-frames` | optional | optional | Deterministic frame types can structure I/O. | Define exact frame subset. |
| `ucf-ess` | optional | optional | Crate-local Minimal Spine projection/read model can index canonical Evidence/Archive/OutputRecord links deterministically. | Keep optional and read-only; no canonical append or Gateway write authority. |
| `ucf-policy-ecology` | yes | yes | Deterministic policy decision layer, small enough for E2E. | Tie decisions to evidence IDs and policy pack input. |
| `ucf-consolidation` | no | no | Cognitive consolidation is broader than minimal deterministic spine. | Add hardening tests and a bounded no-op/single-step contract before inclusion. |
| `ucf-geist` | no | no | Self/recursion semantics not necessary for first E2E and risk overclaim. | Define minimal deterministic state transition or defer to Spine v2. |
| `ucf-neuromod` | no | no | Experimental physiological/neuromod lane; not required for evidence-policy-route spine. | Add boundary tests and clarify production meaning. |
| `ucf-compute` | optional | no | Stub fixture/toy golden lanes are deterministic; optional-real compile-only is not mandatory or spine-safe runtime evidence. | Exclude compute from Minimal Spine; include only an explicit stub fixture hook if needed. |
| `ucf-ai-port` | optional | no | Narrow port can be an integration seam, but backends are partial. | Define a no-op/stub contract and avoid backend maturity claims. |
| `ucf-gateway` | yes for v1.1 readback | no for v1 route execution | Minimal internal read service can expose evidence/archive/output commitments without transport expansion. | Keep v1.1 surface read-only; defer HTTP/security hardening and mutation APIs. |
| `ucf-client` | optional | no | Useful only when gateway is in scope. | Keep behind gateway-smoke/local-only assumptions. |
| `ucf-ops` | yes | yes for validation, no for runtime loop | Gate/report authority is production-leaning and offline-testable. | Keep reports fresh; do not confuse validation tooling with cognitive runtime. |

## 7. Production-Leaning Candidates

- `ucf-types`: broad shared type surface, workspace-visible, deterministic, and unit-tested.
- `ucf-protocol`: code-near protocol implementation with integration/canonical tests and textual specs.
- `runtime/ucf-ops`: extensive operational gates, docs lint, readiness, drift, adversarial, and report tooling.
- `runtime/ucf-gateway`: tested local-only gateway with bounded schema/error/auth behavior; v1.1 adds only a minimal internal read-only Minimal Spine evidence/archive surface.
- `ucf-policy-ecology`: deterministic core policy structures suitable for a small Minimal Spine policy step.
- `ucf-archive` and `ucf-evidence`: credible functional prototypes for evidence/archive, but production-leaning only if the selected file-backed path and retention semantics are made explicit.

## 8. Highest-Risk Overclaim Areas

- **Real compute:** feature lanes and backend names exist, but stub/toy/optional backends are not proof of production real ML compute.
- **Blue-Brain completion/closure docs:** advisory and diagnostic boundaries can look like final completion if titles are read without the current-state index.
- **HH/deepening docs:** HH is simulation-/diagnostic-only/deferred unless a future prompt promotes it with tests and gates.
- **Microcircuit stubs:** `*_stub` crates compile and support replay, but are not real biological microcircuits.
- **Root report currentness:** reports are current only when freshness metadata matches the evaluated HEAD.
- **Vendor chip dirs:** vendored Firewood/RPP/chip material is reference material, not primary UCF maturity evidence.
- **Policy immutability claims:** validation and manifests matter; docs alone do not prove runtime immutability.
- **Full E2E cognitive loop:** router/runtime/consolidation/geist/neuromod/compute names imply a loop, but the first spine should prove only a small deterministic evidence-policy-route-archive path.

## 9. Roadmap Implications

- The Minimal Spine v1.x freeze matrix is `docs/minimal_spine_v1_freeze.md`; use it as the maintenance authority for v1.x component status, claims, authority boundaries, freeze criteria, and invalidators.
- Prompt 4 Minimal Spine should include: `ucf-types`, `ucf-protocol`, `ucf-bus`, `ucf-evidence`, `ucf-archive` or `ucf-archive-store`, `ucf-router`, `ucf-policy-ecology`, policy pack input, and `ucf-ops` validation.
- Prompt 4 may include as optional: `ucf-sdk`, `ucf-frames`, `ucf-ess`, `ucf-replay`, `ucf-gateway`, and `ucf-client`.
- Later integrations: `ucf-runtime` as broad host, real `ucf-compute` backends, `ucf-ai-port` backend families, `ucf-consolidation`, `ucf-geist`, `ucf-neuromod`, Blue-Brain bridge, microcircuits, DBM/HPA, chip/PVGS, adapters, and app.
- Hardening tests are needed first for: archive-vs-ESS authority, router minimal route contract, policy pack decision binding, evidence append/archive round trip, gateway inclusion, and any compute hook.
- Claims to avoid: production real compute, completed Blue-Brain execution, biological microcircuit fidelity, HH runtime integration, immutable policy enforcement without gate evidence, and full cognitive autonomy.

- Policy Ecology authority/schema alignment for the next bounded step is documented in `docs/roadmap/policy_record_authority_schema_alignment.md` (P2); treat protocol policy records as primitives/prototypes; P3 read-only contract landed locally in `core/crates/ucf-policy-ecology/src/policy_field_v1.rs` with targeted tests; P4 candidate-only evaluation contract landed in `core/crates/ucf-policy-ecology/src/policy_evaluation_candidate_v1.rs` with targeted tests.; P5 verify-only audit contract landed in `core/crates/ucf-policy-ecology/src/policy_verify_audit_v1.rs` with targeted tests.

## 10. Maintenance Rules

- Update this registry whenever a crate is added/removed, a module changes public API or feature lanes, a stub/mock/toy path is promoted, a docs-only/deferred/historical area becomes code-backed, or Minimal Spine scope changes.
- Reclassification is triggered by new integration tests, gate inclusion/removal, feature default changes, new persistence/compute backends, report freshness changes, or docs that change current/deferred/advisory authority.
- New crates must be assigned a workspace group, role category, current status, implementation-depth category, test/gate visibility, Minimal Spine eligibility, main gap, and roadmap action.
- Removed, deferred, and historical modules should remain traceable in historical docs or release notes; do not delete old docs merely to simplify the registry.
- Vendor/reference modules must stay `vendor-only` unless they become first-party UCF implementation with explicit workspace, tests, gates, and source authority.
- Generated `out/*.json` reports should normally remain uncommitted unless a repo policy or release workflow explicitly requires them.

| Policy Ecology (P1 roadmap boundary audit) | `docs/roadmap/policy_ecology_roadmap_boundary_audit.md`; `docs/roadmap/policy_ecology_closure.md`; `core/crates/ucf-policy-ecology`; `domains/policy-gateway/crates/ucf-policy-gateway`; `protocol/crates/ucf-protocol` | docs-first boundary audit + bounded closure baseline complete; existing crate/types remain bounded and non-authoritative for runtime action enforcement | docs-only (roadmap/closure) + partial (existing bounded crate/types) | no policy mutation; no runtime enforcement engine; no Gateway/action authority; no identity anchor/finalization; no ISM write/upsert; no Evidence/Archive append changes; no Minimal Spine v1.x change; no prod-readiness claim |
