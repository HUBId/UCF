# UCF Current-State Architecture Index

## 0. Purpose

This document defines which architecture information is authoritative for the current repository state. It separates current truth, code-near specs, operational docs, historical audit trail, deferred or experimental work, and advisory-only boundaries. It is not a full whitepaper and does not replace protocol, schema, policy, or generated spec snapshots.

Use this index as the reading order for roadmap prompts. The Module implementation depth registry is now available at `docs/module_implementation_depth_registry.md` and is required reading before Minimal Spine planning. Code and currently passing gates remain the final source of truth. Minimal UCF Spine v1 is now specified at `docs/minimal_ucf_spine_v1.md`; Prompt 5 must follow that spec.

## 1. Repository Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `29de5845c851ced20bef10aceb46cdcff8b72ee0` |
| HEAD short | `29de5845` |
| Dirty state at index creation | clean |
| Workspace package count | 192 |
| Docs count inspected | 727 files under `docs/` at max depth 2 |
| Markdown files inspected | 612 files at repository max depth 3 |
| Report freshness rule | Root reports and generated reports are current evidence only when embedded `git_head_full` matches the evaluated HEAD. |

Baseline commands used for this index: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -10`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, `find docs -maxdepth 2 -type f | sort`, `find . -maxdepth 3 -type f \( -name "README.md" -o -name "*.md" \) | sort`, `find out -maxdepth 2 -type f | sort | head -200`, and `find core domains runtime crates protocol ucf-sdk chip4 chip-3 -maxdepth 3 -type f \( -name "Cargo.toml" -o -name "README.md" -o -name "*.md" \) 2>/dev/null | sort`.

## 2. Current Truth Rules

1. Code and current tests/gates outrank docs.
2. Current authoritative docs outrank historical sweep, closure, final, and readiness docs.
3. Root reports are current only when embedded `git_head_full` matches the evaluated HEAD.
4. Historical Blue-Brain baselines are audit trail unless explicitly refreshed for the current HEAD.
5. `completion`, `final`, `closure`, `readiness`, or `sweep` in a title does not mean production readiness.
6. Deferred docs must not be used as implementation evidence.
7. Advisory-only docs define boundaries and diagnostics, not capability claims.
8. Feature flags must be interpreted with tests, gates, and enabled lanes, not by existence alone.
9. Vendor directories are reference material unless included in workspace, CI, and current docs.
10. Minimal Spine docs will outrank broad conceptual docs once created.
11. Generated snapshots and report artifacts describe a specific input set and HEAD; do not generalize them to newer HEADs without regeneration.
12. When documentation conflicts, prefer: current code and tests, then code-near specs, then this index, then current operational docs, then historical context.

## 3. Canonical Document Sets

### 3.1 Current Authoritative Architecture Docs

| Path | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `docs/current_state_architecture_index.md` | Current authoritative architecture docs | yes | Canonical reading order and drift-control index for current architecture truth. | whole repository | canonical |
| `docs/SPINE.md` | Current authoritative architecture docs | yes | Defines the top-level repository spine and dependency direction for core, domains, runtime, assets, vendor, and docs. | `core/`, `domains/`, `runtime/`, `assets/`, `vendor/`, `docs/` | canonical |
| `docs/architecture/COHERENCE_LOOP.md` | Current authoritative architecture docs | yes | Describes the coherence-loop concept; use only where backed by code and gates. | core cognitive loop crates, runtime orchestration | canonical |
| `docs/architecture/interfaces.md` | Current authoritative architecture docs | yes | Architecture-facing interface reference; defer to code for exact APIs. | `core/`, `domains/`, `runtime/` | canonical |
| `docs/module_map.md` | Current authoritative architecture docs | yes | Module-level map for repository navigation. | workspace crates | canonical |
| `docs/determinism_lock.md` | Current authoritative architecture docs | yes | Determinism constraints are safety invariants for externally visible outputs. | policy logic, reports, replay, goldens | canonical |
| `docs/minimal_spine_v1_freeze.md` | Current authoritative architecture docs | yes | Freezes the Minimal Spine v1.x integration matrix, authority boundaries, claims, tests, and invalidators. | Minimal Spine v1.x code/tests/docs | canonical |
| `docs/roadmap/post_freeze_roadmap_selection.md` | Current authoritative architecture docs | yes | Selects the post-freeze primary roadmap line and prompt series without changing Minimal Spine v1.x authority. | roadmap planning; optional compute lane | canonical-planning |
| `docs/roadmap/full_consolidation_roadmap_boundary_audit.md` | Current authoritative architecture docs | yes | Current planning and boundary audit for the consolidation line; defines the overclaim guard for the bounded Micro→Meso→Macro E2E test, Micro/Meso explicit append/readback, Macro candidate, local consolidation-level finalization, and deferred Replay/Sleep/Geist/ISM/identity/Gateway surfaces. | `domains/consolidation`, `protocol`, `core/crates/ucf-evidence`, `domains/archive`, `runtime/ucf-replay`, `domains/geist` | canonical-planning |
| `docs/roadmap/consolidation_record_authority_schema_alignment.md` | Current authoritative architecture docs | yes | Current planning authority for Micro/Meso/Macro record authority, candidate-vs-emitted semantics, Evidence/Archive boundaries, and the bounded Prompt 27-33 implementation sequence. | `domains/consolidation`, `protocol`, `core/crates/ucf-types`, `core/crates/ucf-evidence`, `domains/archive`, `runtime/ucf-replay`, `domains/geist` | canonical-planning |
| `docs/roadmap/readiness_gate_timeout_stability_audit.md` | Current authoritative architecture docs | yes | Operational/readiness risk audit for Prompt 35A; isolates local readiness-gate timeout to the internal workspace-test phase, records stale-report handling, and preserves readiness-spine drift as follow-up without weakening gates. | `runtime/ucf-ops`, `.github/workflows`, `out/` reports | canonical-planning-risk |
| `docs/roadmap/real_compute_lane_inventory.md` | Current authoritative architecture docs | yes | Inventories the optional Real Compute lane, feature/backend matrix, tests, docs drift, and guardrails without implementing or activating compute. | `runtime/ucf-compute`, `domains/ai-backends`, `core/crates/ucf-ai-port`, CI/docs | canonical-planning |
| `docs/roadmap/compute_backend_naming_boundary_plan.md` | Current authoritative architecture docs | yes | Defines the canonical compute backend naming taxonomy and boundary cleanup plan for stub, toy, mock, optional-real, remote/external, experimental, deferred, and forbidden wording without changing backend behavior. | `runtime/ucf-compute`, `domains/ai-backends`, `core/crates/ucf-ai-port`, CI/docs | canonical-planning |
| `docs/roadmap/compute_feature_ci_matrix.md` | Current authoritative architecture docs | yes | Defines the docs-only compute feature CI/check matrix for default no-real, stub, toy, optional-real compile-only, remote/external compile-only, link/audit, and docs/gates lanes without runtime or production claims. | `runtime/ucf-compute`, `domains/ai-backends`, `core/crates/ucf-ai-port`, `.github/workflows/*` | canonical-planning |
| `docs/roadmap/real_compute_optional_lane_closure.md` | Current authoritative architecture docs | yes | Closes the current optional compute lane baseline for Prompts 16-24 under compile-only/non-production claims and records readiness-gate timeout-risk monitoring. | `runtime/ucf-compute`, `domains/ai-backends`, `core/crates/ucf-ai-port`, `runtime/ucf-ops` | canonical-planning-closure |
| `docs/feature_matrix.md` | Current authoritative architecture docs | yes | Defines supported feature lanes and must be read with tests. | `runtime/ucf-compute`, backend crates, CI lanes | canonical |
| `docs/blue_brain_authority_chain_status_map.md` | Current authoritative architecture docs | qualified | Current only for Blue-Brain document authority boundaries, not for global UCF capability claims. | Blue-Brain docs and bridge crates | advisory-boundary |

### 3.2 Current Operational Docs

| Path | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `README.md` | Current operational docs | yes | Root entry point for feature lanes and canonical commands. | whole workspace | operational |
| `docs/README.md` | Current operational docs | yes | Operational documentation index; now points to this current-state index before historical Blue-Brain chains. | docs navigation | operational |
| `docs/artifact_convention_v0.md` | Current operational docs | yes | Defines artifact locations and report handling expectations. | `out/`, `runtime/ucf-ops` | operational |
| `docs/readiness_gate.md` | Current operational docs | yes | Readiness gate behavior and interpretation. | `runtime/ucf-ops`, policy packs | operational |
| `docs/adversarial_harness.md` | Current operational docs | yes | Operational adversarial suite reference. | `runtime/ucf-ops`, test harnesses | operational |
| `docs/replay-harness.md` | Current operational docs | yes | Replay harness operation and evidence use. | `runtime/ucf-replay`, replay crates | operational |
| `docs/replay_audit.md` | Current operational docs | yes | Replay audit interpretation. | `runtime/ucf-replay`, `crates/replay_*` | operational |
| `docs/golden_update.md` | Current operational docs | yes | Golden update workflow; use with current tests. | test fixtures, goldens | operational |
| `docs/airgap_workflows.md` | Current operational docs | yes | Offline-first workflow guidance. | ops, release, artifacts | operational |
| `docs/policy_packs.md` | Current operational docs | yes | Policy pack operation and validation. | `policies/`, `runtime/ucf-policy`, `runtime/ucf-ops` | operational |
| `docs/runbooks.md` | Current operational docs | yes | Operator runbook entry point. | runtime and ops crates | operational |
| `docs/perf_playbook.md` | Current operational docs | yes | Performance checks and playbook. | benches, runtime | operational |
| `docs/release_spine.md`, `docs/release_rc_pack.md` | Current operational docs | yes | Release/readiness operational packaging references. | release workflows, ops reports | operational |

### 3.3 Code-Near Specs

| Path | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `protocol/crates/ucf-protocol/spec/v1.md` | Code-near specs | yes | Protocol v1 spec colocated with the protocol crate. | `protocol/crates/ucf-protocol` | code-near-spec |
| `protocol/crates/ucf-protocol/spec/messages_v1.md` | Code-near specs | yes | Message definitions colocated with protocol implementation. | `protocol/crates/ucf-protocol` | code-near-spec |
| `protocol/crates/ucf-protocol/spec/README.md` | Code-near specs | yes | Protocol spec index. | `protocol/crates/ucf-protocol` | code-near-spec |
| `docs/spec_snapshot.md` | Code-near specs | yes when regenerated for current policy/code inputs | Generated snapshot from code registries and policy manifests. | frame records, stage contracts, model slots, policy packs | code-near-spec |
| `docs/artifact_schema_snapshots.md` and `docs/artifact_schema_snapshots/*.json` | Code-near specs | yes for schema validation | Schema snapshots used by artifact checks. | `runtime/ucf-ops`, artifact reports | code-near-spec |
| `docs/sdk_versioning.md` | Code-near specs | yes | SDK compatibility/versioning surface. | `ucf-sdk` | code-near-spec |
| `docs/policy_key_registry.md` | Code-near specs | yes | Registry for policy keys. | `policies/`, policy crates | code-near-spec |
| `docs/active_enablement_rule_v2.md`, `docs/active_evidence_v3.md` | Code-near specs | yes with current tests | Active enablement/evidence semantics. | model/backend eligibility and ops | code-near-spec |
| `docs/backend_evidence_snapshot_v4.md`, `docs/models_eligibility_v3.md`, `docs/model_slots.md` | Code-near specs | yes with current gates | Backend/model evidence and slot declarations. | `models/`, `runtime/ucf-compute`, backend crates | code-near-spec |

### 3.4 Advisory / Boundary Docs

| Path | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `docs/sandboxing-v1.md` | Advisory / boundary docs | yes | Sandbox boundary expectations. | `core/crates/ucf-sandbox`, runtime isolation | advisory-boundary |
| `docs/security_v1.md` | Advisory / boundary docs | yes | Security posture and constraints. | runtime, gateway, ops | advisory-boundary |
| `docs/threat_model_v1.md` | Advisory / boundary docs | yes | Threat model boundary reference. | whole workspace | advisory-boundary |
| `docs/no_hidden_network.md` | Advisory / boundary docs | yes | Network boundary and hidden dependency rule. | runtime, ops, adapters | advisory-boundary |
| `docs/process_isolation.md` | Advisory / boundary docs | yes | Process isolation expectations. | runtime, compute, sandbox | advisory-boundary |
| `docs/zero_trust_local.md` | Advisory / boundary docs | yes | Local zero-trust guidance. | runtime, client, gateway | advisory-boundary |
| `docs/strict_mode.md`, `docs/strict_mode_v3.md` | Advisory / boundary docs | yes | Strict mode semantics and guard behavior. | ops, policy, runtime | advisory-boundary |
| `docs/blue_brain_*guard*`, `docs/blue_brain_*boundary*`, `docs/blue_brain_*authority*` | Advisory / boundary docs | qualified | Useful for no-direct-authority and diagnostic boundaries; not capability proof. | Blue-Brain bridge, biophys, microcircuit docs/code | advisory-boundary |

### 3.5 Historical Audit Trail

| Path or group | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `docs/blue_brain_*readiness_sweep*` | Historical audit trail | no | Readiness snapshots for prior scopes/HEADs; title is not current readiness proof. | Blue-Brain and related crates | historical-only |
| `docs/blue_brain_*final*`, `docs/blue_brain_*closure*`, `docs/blue_brain_*completion*` | Historical audit trail | no unless specifically referenced by this index as boundary | Closure/completion records are audit trail and may be stale. | Blue-Brain docs/code | historical-only |
| `docs/*sweep_v*.md`, including governance/readiness/bundle/primary-semantics sweeps | Historical audit trail | no | Sweep outputs document historical checks, not live implementation state. | ops reports, artifacts | historical-only |
| `docs/roadmap_anchor_v*.md` | Historical audit trail | no | Prior roadmap anchors; consult only as lineage. | roadmap docs | historical-only |
| `docs/v*_signoff.md` | Historical audit trail | no | Version signoff snapshots. | release history | historical-only |
| `docs/blue_brain_final_evidence_baseline_refresh_2026_05_08.md`, `docs/blue_brain_final_evidence_baseline_refresh_2026_05_09.md`, `docs/blue_brain_maintenance_consolidation_refresh_2026_05_10.md`, `docs/blue_brain_finale_maintenance_convergence_evidence_sync_2026_05_12.md` | Historical audit trail | no unless HEAD metadata matches evaluated HEAD | Evidence refresh lineage; use freshness metadata before citing. | reports, ops | historical-only |
| `out/**` reports | Historical audit trail | no unless generated for evaluated HEAD | Generated reports are run artifacts and usually should not be committed. | ops reports | historical-only |

### 3.6 Deferred / Experimental Docs

| Path or group | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `docs/blue_brain_hh_*` | Deferred / experimental docs | no | HH pilot/candidate material is explicitly deferred or prerequisite-bound. | biophys, microcircuits, Blue-Brain docs | deferred-only |
| `docs/remote_compute_deferred.md` | Deferred / experimental docs | no | Remote compute is deferred. | `runtime/ucf-compute`, remote adapters | deferred-only |
| `docs/real_compute_exit_*`, `docs/real_compute_readiness_sweep_v*`, `docs/real_compute_reference_surface_v1.md` | Deferred / experimental docs | no production claim | Real-compute lane must be proven by enabled tests/gates before becoming current core. | `runtime/ucf-compute`, backend crates | deferred-only |
| `docs/backend_candle_*`, `docs/backend_burn_*`, `docs/sae_*`, `docs/ssm*`, `docs/lfm*`, `docs/world_*` | Deferred / experimental docs | partial | Backend/research lanes may be current only for feature-gated tests, not default spine capability. | AI/backend crates | deferred-only |
| `docs/blue_brain_md*`, `docs/blue_brain_*model_deepening*`, `docs/blue_brain_*third*` | Deferred / experimental docs | no for new capability claims | Model-deepening and third-region docs are bounded historical/experimental lineage unless refreshed. | Blue-Brain and microcircuit crates | deferred-only |
| `docs/architecture/DELTA_ONN_SNN.md`, `docs/architecture/microcircuit_path.md` | Deferred / experimental docs | partial | Research path docs; use with code registry before claims. | SNN, ONN, microcircuit crates | deferred-only |

### 3.7 Ambiguous Docs Requiring Review

| Path or group | Category | Current authority? | Reason | Related code areas | Roadmap use |
|---|---|---:|---|---|---|
| `docs/architecture.md` | Ambiguous / needs decision | no | Describes an older Chip 2 scaffold with placeholder APIs and does not represent the current 192-package workspace. | legacy scaffold crates | ambiguous-review-needed |
| `docs/biophys_runtime.md`, `docs/biophys_runtime_v1.md`, `docs/biophys_neuro.md` | Ambiguous / needs decision | partial | Real crates exist, but production depth must be measured by the future registry. | `domains/ucf-biophys`, `crates/biophys_*` | ambiguous-review-needed |
| `docs/readiness_spine_v8.md` and `docs/readiness_lock_sweep_v21.md` | Ambiguous / needs decision | partial operational lineage | Names imply authority; current use requires HEAD freshness and gate context. | readiness gates | ambiguous-review-needed |
| `docs/reviewability_truth_v7.md` | Ambiguous / needs decision | partial | May be useful operationally, but needs alignment with this index and current gates. | ops/review workflows | ambiguous-review-needed |
| `chip-3/README.md`, `chip4/` docs/code | Ambiguous / needs decision | no global authority | Chip material exists, but workspace/runtime role must be classified by implementation depth. | `chip-3/`, `chip4/`, `crates/pvgs` | ambiguous-review-needed |
| `vendor/**` Markdown or reference material | Ambiguous / needs decision | no | Reference material unless wired into workspace, CI, and current docs. | vendor integrations | ambiguous-review-needed |

## 4. Domain Current-State Map

| Domain | Current status | Canonical docs | Canonical code paths | Historical docs | Deferred/experimental docs | Main drift risk |
|---|---|---|---|---|---|---|
| Repository / Workspace / CI | current-operational | `README.md`, `docs/SPINE.md`, `docs/feature_matrix.md`, `docs/determinism_lock.md` | `Cargo.toml`, `.github/`, `core/`, `domains/`, `runtime/` | `docs/v*_signoff.md`, `docs/*sweep_v*.md` | none primary | Old signoff/readiness docs read as current CI state. |
| UCF Types / Protocol / SDK | code-near-spec | `protocol/crates/ucf-protocol/spec/*.md`, `docs/sdk_versioning.md`, `docs/spec_snapshot.md` | `core/crates/ucf-types`, `protocol/crates/ucf-protocol`, `ucf-sdk` | old schema/signoff docs | none primary | Schema authority split between generated snapshot, protocol spec, and types crate. |
| Evidence / Archive / Event Log | current-core | `docs/active_evidence_v3.md`, `docs/artifact_convention_v0.md`, `docs/proof_carrying_logs.md` | `core/crates/ucf-evidence`, `core/crates/ucf-events`, `crates/replay_evidence`, archive/artifact crates | evidence refresh reports | deferred remote/archive concepts | Confusing generated reports with current evidence semantics. |
| Bus / Runtime / Router | current-core | `docs/SPINE.md`, `docs/architecture/interfaces.md`, runtime READMEs | `core/crates/ucf-bus`, `core/crates/ucf-router`, `runtime/ucf-runtime`, `core/crates/ucf-output-router` | old architecture/readiness docs | runtime-selection Blue-Brain lineage | Router/runtime capability inferred from historical docs instead of code/tests. |
| Core Cognitive Loop | partial-prototype | `docs/architecture/COHERENCE_LOOP.md`, `docs/ops_explain_tick.md`, `docs/spec_snapshot.md` | `core/ucf-core`, `core/crates/ucf-predictive-coding`, `core/crates/ucf-attn-controller`, `core/crates/ucf-coupling`, `core/crates/ucf-fold` | older conceptual docs | research model lanes | Concept docs overstating integrated production loop. |
| ESS / Frames / Experience Records | current-supporting | `docs/spec_snapshot.md`, `docs/artifact_schema_snapshots.md` | `domains/ucf-ess`, `domains/ucf-frames` | readiness/snapshot lineage | none primary | ESS vs archive/evidence boundary unclear. |
| Metabolic / Neuromod / HPA / DBM | partial-prototype | `docs/biophys_neuro.md`, Blue-Brain boundary docs | `domains/ucf-neuromod`, `domains/ucf-dbm`, `crates/dbm_*`, `hpa` | Blue-Brain BR/BB/MD sweeps | HH pilot docs | Biological naming mistaken for production physiological fidelity. |
| Consolidation Kernel | current-supporting | `docs/roadmap/full_consolidation_roadmap_boundary_audit.md`, `docs/roadmap/consolidation_record_authority_schema_alignment.md`, `docs/roadmap/full_consolidation_closure.md`, `docs/architecture/COHERENCE_LOOP.md` | `domains/consolidation/crates/ucf-consolidation`, `core/crates/ucf-commit`, `core/crates/ucf-sleep-coordinator` | Blue-Brain consolidation refreshes | bounded retrieval/consolidation candidate docs | Historical consolidation closure interpreted as current implementation depth; current claims stop at Micro/Meso explicit append/readback, Macro candidate, and local consolidation-level finalization with no Replay/Sleep/Geist/ISM/identity readiness; Prompt 35 closure is gate-stability-pending because readiness-gate timed out under 300 second guards. |
| Geist / Self-State / ISM / Recursion | partial-prototype | `docs/architecture/interfaces.md`, `docs/module_map.md` | `domains/geist/crates/ucf-geist`, `core/crates/ucf-ism`, `core/crates/ucf-recursion-controller` | self-state historical docs if present | research recursion docs | Conceptual claims outrun code integration. |
| Policy Ecology / Normative Field | current-core | `docs/policy_packs.md`, `docs/policy_key_registry.md`, `docs/spec_snapshot.md` | `core/crates/ucf-policy-ecology`, `runtime/ucf-policy`, `policies/` | governance sweeps | experimental overlays | Governance sweep titles confused with live policy authority. |
| Gateway / Client / API | current-supporting | `docs/ucf_client.md`, `docs/operator_console.md`, `docs/runbooks.md` | `runtime/ucf-gateway`, `runtime/ucf-client`, `runtime/ucf-console` | operator signoff/history | adapters | API availability inferred without current endpoint/test evidence. |
| Replay / Goldens / Drift | current-operational | `docs/replay-harness.md`, `docs/replay_audit.md`, `docs/golden_update.md`, `docs/adversarial_harness.md` | `runtime/ucf-replay`, `crates/replay_executor`, `crates/replay_evidence`, test/golden fixtures | drift/nightly reports | Replay Scheduler line remains deferred for consolidation | Stale drift reports or consolidation E2E tests used as proof of replay readiness. |
| Real Compute / AI Backends / Burn / Candle / Toy / Stub | experimental | `docs/feature_matrix.md`, `docs/backends.md`, `docs/models_eligibility_v3.md`, `docs/backend_evidence_snapshot_v4.md`, `docs/roadmap/compute_feature_ci_matrix.md`, `docs/roadmap/real_compute_lane_inventory.md`, `docs/roadmap/real_compute_optional_lane_closure.md` | `runtime/ucf-compute`, `domains/ai*`, `runtime/ucf-ebm-train`, backend crates | real-compute readiness sweeps | `docs/real_compute_*`, `docs/backend_burn_*`, `docs/backend_candle_*` | Feature existence mistaken for enabled real-compute production lane; use stub fixture/toy golden/optional-real compile-only taxonomy and Prompt 24 closure scope. |
| Blue-Brain / Biophys / Microcircuits | advisory-only | `docs/blue_brain_authority_chain_status_map.md`, `docs/blue_brain_canonical_model_boundary_map_v1.md`, `docs/blue_brain_canonical_matrices_final_freeze_v1.md` | `domains/ucf-bluebrain-bridge`, `domains/ucf-biophys`, `crates/biophys_*`, `crates/microcircuit_*` | numerous `docs/blue_brain_*` sweeps/closures | HH/model-deepening/third-region docs | Historical Blue-Brain reports treated as current implementation proof. |
| Chip-3 / Chip-4 / PVGS / Multi-chip | ambiguous | `chip-3/README.md`, `docs/architecture/chip2_overview.md` | `chip4`, `crates/pvgs`, `pvgs_client`, legacy scaffold crates | chip signoffs if any | multi-chip concepts | Chip docs/code role unclear in current minimal spine. |
| Safety / Sandbox / Governor / Authority Boundaries | current-supporting | `docs/sandboxing-v1.md`, `docs/security_v1.md`, `docs/threat_model_v1.md`, `docs/no_hidden_network.md`, `docs/zero_trust_local.md`, `docs/minimal_ucf_spine_v1.md` v1.5 | `core/crates/ucf-sandbox`, `core/crates/ucf-risk-gate`, `runtime/ucf-policy`, governance crates | authority sweeps | research authority maps | Boundary docs read as active capability authority; Minimal Spine v1.5 keeps capability issuance deferred. |
| Ops / Reports / Artifact Conventions | current-operational | `docs/artifact_convention_v0.md`, `docs/readiness_gate.md`, `docs/runbooks.md`, `docs/attested_runs.md` | `runtime/ucf-ops`, `out/` generated artifacts | committed/generated historical reports | none primary | Root report self-reference and stale HEAD metadata. |
| Vendor / External reference material | advisory-only | current docs only when explicitly referenced | `vendor/`, external snapshots | imported reference notes | vendor experiments | Reference material mistaken for maintained UCF implementation. |

## 5. Minimal Spine Candidate

This is a candidate only. It does not finalize the Minimal UCF Spine Specification.

| Candidate module | Role in spine | Current maturity | Include now? | Reason |
|---|---|---|---:|---|
| `core/crates/ucf-types` | Canonical shared IDs/types and serialization surfaces | current-core | yes | Needed before protocol, SDK, policy, and evidence can be stable. |
| `protocol/crates/ucf-protocol` | Wire/protocol schema and message contract | code-near-spec | yes | Colocated specs make it the natural protocol authority. |
| `ucf-sdk` | Consumer-facing SDK surface | current-supporting | yes, if public API/read API is in scope | Required for external integration, but should follow protocol/types authority. |
| `core/crates/ucf-bus` | Internal event/message bus | current-core | yes | Runtime/router spine needs a bus contract. |
| `core/crates/ucf-evidence` | Evidence records and digestable proof surfaces | current-core | yes | Evidence is required for gates, replay, and roadmap truth claims. |
| Archive/artifact crates (`crates/assets`, `crates/asset_*`, archive-equivalent surfaces) | Persisted artifact/archive support | current-supporting | yes, after registry confirms exact canonical archive crate | The spine needs artifact storage boundaries; naming must be clarified. |
| `core/crates/ucf-router` | Routing contract and dispatch path | current-core | yes | Minimal spine needs deterministic routing. |
| `runtime/ucf-runtime` | Runtime orchestration | current-core | yes | Provides executable wiring for core surfaces. |
| `domains/ucf-frames` | Frame/record domain | current-supporting | yes, if frame records are in the minimal spec | Spec snapshot includes frame/record schemas. |
| `domains/ucf-ess` | ESS governance/state records | current-supporting | yes, if ESS is canonical state surface | Needs explicit boundary against archive/evidence. |
| `core/crates/ucf-policy-ecology` | Normative/policy ecology | current-core | yes | Policy gates and readiness depend on policy surfaces. |
| `domains/consolidation/crates/ucf-consolidation` | Consolidation kernel | current-supporting | yes, if consolidation is part of minimal loop | Include only after implementation-depth registry verifies depth. |
| `domains/geist/crates/ucf-geist` | Self-state/geist surface | partial-prototype | maybe | Integrate only if registry shows stable code and tests. |
| `core/crates/ucf-ism` and `core/crates/ucf-recursion-controller` | ISM/recursion support | partial-prototype | maybe | Candidate supporting modules; avoid capability claims until measured. |
| `runtime/ucf-gateway`, `runtime/ucf-client` | Read/API integration | current-supporting | maybe | Include if Minimal Spine requires external read/API surface. |
| `runtime/ucf-compute` | Compute/backends lane | experimental | no by default | Keep optional; stub fixture and toy golden are not real inference, optional-real compile-only is not runtime proof, and production claims remain forbidden for current lanes. |

## 6. Historical / Deferred Warning List

Do not use these groups as current implementation proof without current HEAD refresh and code/test corroboration:

- `docs/blue_brain_*readiness_sweep*`
- `docs/blue_brain_*completion*`, `docs/blue_brain_*closure*`, and `docs/blue_brain_*final*`
- `docs/blue_brain_hh_*`
- `docs/blue_brain_md*` and third-deepening/third-region material
- `docs/*sweep_v*.md`, `docs/*ultimate*`, `docs/*terminal*`, `docs/*absolute*`
- `docs/roadmap_anchor_v*.md`
- `docs/v*_signoff.md`
- `docs/real_compute_*` unless the target lane is tested and enabled for the evaluated HEAD
- `out/**` reports unless generated for the evaluated HEAD and not self-invalidated by root report references
- `vendor/**` reference material unless explicitly wired into workspace/CI/current docs

## 7. Roadmap Implications

- Before new features, create the Module Implementation-Depth Registry and verify crate-by-crate status as real, partial, stub, toy, deferred, advisory, or unknown.
- The Minimal UCF Spine Specification should cover types/protocol, SDK if public API is in scope, bus, evidence, archive/artifacts, router/runtime, frames/ESS, policy ecology, consolidation, optional geist/ISM/recursion, and gateway/client only if read API is in scope.
- Avoid claims that Blue-Brain, HH, real compute, Candle/Burn, biophysical simulation, multi-chip, or recursive self-state are production-ready unless current code, tests, and gates prove the exact lane.
- Resolve ESS vs archive/evidence before using either as the canonical state/evidence surface.
- Resolve `ucf-types` vs `ucf-protocol` schema authority before changing schemas.
- Treat Blue-Brain advisory lines as bounded diagnostics unless a future spec explicitly promotes them.
- Treat `chip-3`, `chip4`, PVGS, and vendor content as ambiguous until the registry classifies their workspace and CI role.

## 8. Maintenance Rules

Update this document when any of the following changes occur:

- Workspace package membership, top-level spine layout, or canonical crate ownership changes.
- Protocol, record, artifact, policy, or generated spec snapshot schemas change.
- A historical/deferred/advisory document is promoted to current authority.
- A real-compute, Blue-Brain, HH, chip, vendor, or research lane is promoted into CI-enforced current behavior.
- Readiness/report freshness rules or artifact conventions change.
- The Module Implementation-Depth Registry or Minimal UCF Spine Specification is created or updated.

Freshness check procedure:

1. Capture `git rev-parse HEAD` and `git status --short`.
2. Regenerate required reports for the evaluated HEAD.
3. Confirm each root or generated report embeds matching `git_head_full` when the schema supports it.
4. Treat reports without matching HEAD metadata as historical context only.
5. Only explicit code-reviewed documentation updates may mark historical docs as current; names such as `final`, `closure`, `completion`, `readiness`, or `sweep` never promote themselves.
