# Minimal UCF Spine v1.x Freeze

## 0. Purpose

This document freezes the Minimal Spine v1.x basis.

It is not a Full-UCF-readiness declaration. It defines the v1.x integration matrix, authority boundaries, test baseline, allowed and forbidden claims, freeze criteria, invalidators, maintenance rules, and next roadmap blocks.

The freeze is deliberately narrow: it protects the deterministic candidate/output/evidence/archive spine and the derived read surfaces added through v1.5 without adding new features, runtime paths, real compute, Gateway HTTP, Replay Scheduler, Geist/ISM, DBM/HPA, Blue-Brain, HH, production microcircuit, vendor-chip, capability issuance, or full Micro -> Meso -> Macro integration.

Post-freeze Evidence/Archive append/readback additions for Replay, Sleep, and Geist/ISM are documented in `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md` as bounded audit/provenance persistence only. They do not alter Minimal Spine v1.x, add runtime/identity/Gateway semantics, or create a second event log.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `db04585b6e5e272aa797b183b420ec9cbb720921` |
| HEAD short | `db04585b` |
| Dirty state at freeze-authoring start | clean |
| Workspace package count | 192 |
| Minimal spine test present | yes |
| Gateway read test present | yes |
| ESS read model test present | yes |
| Consolidation hook test present | yes |
| Neuromod hook test present | yes |
| Minimal spine spec present | yes |
| Module registry present | yes |
| Current-state index present | yes |

Canonical companion documents:

- `docs/roadmap/post_freeze_roadmap_selection.md` selects the next post-freeze roadmap line and prompt series while preserving this v1.x freeze.
- `docs/minimal_ucf_spine_v1.md` defines the Minimal UCF Spine v1 technical specification and v1.1-v1.5 additions.
- `docs/module_implementation_depth_registry.md` classifies module maturity, overclaim risks, and maintenance triggers.
- `docs/current_state_architecture_index.md` defines current architecture truth ordering and report freshness rules.

Baseline commands used for this freeze: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -15`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and presence checks for the v1.x test/spec/index files.

## 2. v1.x Scope Summary

| Version | Component | Status |
|---|---|---|
| v1.0 | Minimal Spine E2E | frozen |
| v1.1 | Gateway Read API | frozen |
| v1.2 | ESS Read Model | frozen |
| v1.3 | Consolidation Micro Hook | frozen |
| v1.4 | Neuromod Envelope | frozen |
| v1.5 | Capability Boundary | frozen/deferred |

The v1.x scope is a minimal, deterministic, offline-testable integration basis. It proves canonical record creation, evidence/archive append and readback, read-only audit projection, and bounded derived metadata/candidate/read-model hooks. It does not prove a full cognitive runtime.

## 3. v1.x Integration Matrix

| v1.x item | Path | Role | Authority level | Test coverage | Status |
|---|---|---|---|---|---|
| `ucf-protocol` `CandidateSetRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/messages_v1.md` | Canonical candidate-set protocol record and deterministic bytes/digest source. | schema-authority | `cargo test -p ucf-protocol --all-targets`; `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` | frozen-v1 |
| `ucf-protocol` `OutputRecord` | `protocol/crates/ucf-protocol/src/lib.rs`; `protocol/crates/ucf-protocol/spec/messages_v1.md` | Canonical output protocol record and deterministic bytes/digest source. | schema-authority | `cargo test -p ucf-protocol --all-targets`; `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture`; `cargo test -p ucf-gateway --test minimal_spine_read_api -- --nocapture` | frozen-v1 |
| `ucf-types` primitives/digests/IDs | `core/crates/ucf-types/src/lib.rs` | Shared deterministic primitive, digest, ID, fixed-point, time, and re-export support layer. | primitive-authority | `cargo test --workspace`; downstream v1.x integration tests | supporting |
| `ucf-evidence` `EvidenceEnvelope` / `EvidenceStore` use | `core/crates/ucf-evidence/src/lib.rs` | Evidence envelope and append/get proof surface used by the spine. | append-authority | `cargo test --workspace`; `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` | frozen-v1 |
| `ucf-archive` `ExperienceRecord` / proof append use | `domains/archive/crates/ucf-archive/src/lib.rs` | Experience/proof append helper surface for evidence/archive integration. | append-authority | `cargo test --workspace`; `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` | frozen-v1 |
| `ucf-archive-store` output event / readback / root use | `domains/archive/crates/ucf-archive-store/src/lib.rs` | Deterministic local archive append, output-event readback, and root commitment surface. | append-authority | `cargo test --workspace`; `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture`; `cargo test -p ucf-gateway --test minimal_spine_read_api -- --nocapture` | frozen-v1 |
| `ucf-router` `minimal_spine_e2e` | `core/crates/ucf-router/tests/minimal_spine_e2e.rs` | Canonical allow, deny/no-mutation, and deterministic replay E2E host. | validation-only | `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` | frozen-v1 |
| `ucf-gateway` `SpineReadService` | `runtime/ucf-gateway/src/spine_read.rs`; `runtime/ucf-gateway/tests/minimal_spine_read_api.rs` | Internal read-only audit service for health, evidence links, output events, and archive root. | read-only-audit | `cargo test -p ucf-gateway --test minimal_spine_read_api -- --nocapture` | frozen-v1.1 |
| `ucf-ess` `MinimalSpineEssProjection` / `MinimalSpineEssReadModel` | `domains/ucf-ess/src/v1/minimal_spine.rs`; `domains/ucf-ess/tests/minimal_spine_read_model.rs` | Derived ESS read-model summaries over canonical evidence/archive/output links. | derived-read-model | `cargo test -p ucf-ess --all-targets` | frozen-v1.2 |
| `ucf-consolidation` `MinimalSpineMicroMilestoneCandidate` | `domains/consolidation/crates/ucf-consolidation/src/lib.rs`; `domains/consolidation/crates/ucf-consolidation/tests/minimal_spine_micro_hook.rs` | Candidate-only micro milestone hook derived from canonical spine links. | derived-candidate | `cargo test -p ucf-consolidation --all-targets` | frozen-v1.3 |
| `ucf-neuromod` `MinimalSpineNeuromodEnvelope` | `domains/ucf-neuromod/src/minimal_spine.rs`; `domains/ucf-neuromod/tests/minimal_spine_envelope.rs` | Derived neuromod metadata envelope with conservative bounded hints and no override authority. | derived-metadata | `cargo test -p ucf-neuromod --all-targets` | frozen-v1.4 |
| `CapabilityIssuanceRecord` v1.5 deferred boundary | `docs/minimal_ucf_spine_v1.md`; this document | Safety boundary stating capability issuance remains deferred and unimplemented for Minimal Spine v1.x. | deferred-boundary | Docs lint; readiness gate interpretation; absence of active Minimal Spine issuance implementation | frozen-v1.5 |

## 4. Authority Boundary Matrix

| Area | Authority | Explicitly not allowed | Enforced by |
|---|---|---|---|
| Protocol schema | `ucf-protocol` owns v1 canonical protocol records and deterministic encoding for `CandidateSetRecord` and `OutputRecord`. | Duplicate protocol authority in ESS, Gateway, Consolidation, Neuromod, SDK helpers, or docs-only schemas. | Protocol specs, protocol tests, and Minimal Spine E2E digest assertions. |
| Shared primitives | `ucf-types` owns shared deterministic primitive, digest, ID, time, and helper types. | Promotion of primitive helpers into new protocol record authority without an explicit protocol version/spec change. | Workspace tests and source ownership boundaries. |
| Evidence/Archive append | `ucf-evidence`, `ucf-archive`, and `ucf-archive-store` provide the canonical append/proof surface used by v1.x. | ESS/Gateway/Neuromod/Consolidation append authority over canonical spine events. | Router E2E append/readback assertions and read-only/derived module tests. |
| Archive readback | `ucf-archive-store` provides deterministic readback/root/output-event audit data. | Treating readback projections as write authority or production database semantics. | Router E2E and Gateway read-only tests. |
| Router E2E host | `ucf-router` hosts the minimal deterministic allow, deny/no-mutation, and replay validation path. | Full runtime orchestration, real compute, Gateway HTTP, production cognitive loop, or replay scheduler claims. | `minimal_spine_e2e` integration tests. |
| Gateway read service | `SpineReadService` is an internal read-only audit surface over supplied spine links, output events, and archive root. | Gateway writes, HTTP API claims, capability grants, policy mutation, output mutation, or archive append from Gateway. | `minimal_spine_read_api` tests and absence of write endpoints in the v1.1 surface. |
| ESS | ESS is a derived read model only. | Event-log authority, canonical append, Gateway write authority, policy mutation, or output override. | `minimal_spine_read_model` tests and no append-authority in the ESS v1.2 module. |
| Consolidation hook | Consolidation emits a derived micro milestone candidate only. | Macro consolidation, replay trigger, scheduler wiring, event-log authority, or output/policy override. | `minimal_spine_micro_hook` tests and candidate-only struct semantics. |
| Neuromod envelope | Neuromod emits derived bounded metadata only. | Policy override, output override, scheduler control, capability grant, or canonical event authority. | `allows_decision_override=false`, bounded hint validation, and `minimal_spine_envelope` tests. |
| Capability issuance | Capability issuance is deferred for Minimal Spine v1.x. | Active issuance, grant, revoke, refresh, credential/token/scope creation, self-grant, policy mutation, Gateway write bypass, or compute-triggered grants. | v1.5 docs boundary, no `CapabilityIssuanceRecord` Minimal Spine implementation, docs lint, and readiness gate interpretation. |
| Policy | Existing policy decision authority remains in policy components/packs selected by the Minimal Spine route. | Policy-pack rewriting, self-mutating policy, capability-triggered policy change, or derived-module policy authority. | Minimal Spine E2E allow/deny assertions and policy validation/gate commands. |
| Compute | No real compute is required by v1.x. | Burn/Candle/LLM/LFM/JEPA/SAE/SSM/NSR or other real compute as a required v1.x dependency. | v1.x docs, router-hosted deterministic test path, and absence of required compute integration. |
| Gateway write | No Gateway write authority exists in v1.x. | HTTP write, local write, archive append, evidence append, policy mutation, output materialization, or capability issuance through Gateway. | Gateway read-only tests and documented v1.1 boundary. |
| Replay | v1.x validates deterministic replay in the router test only. | Replay Scheduler, macro replay trigger, Geist write, or scheduler-mediated output changes. | Router deterministic replay assertions and out-of-scope freeze text. |
| Geist/ISM | Geist/ISM runtime/store authority is out of Minimal Spine v1.x scope; the separate post-Sleep line currently proves only candidate-only projection, verify-only audit, local ISM candidate boundary, bounded E2E determinism, and post-freeze audit/provenance append/readback outside Minimal Spine v1.x. | Self-state authority, runtime Geist, `GeistApplied`, ISM write/upsert, `IdentityAnchor`, IdentityFinalization, memory stabilization, persistent self authority, cognitive recursion claims, Policy mutation, Gateway/action authority, production readiness, second event-log authority, or Geist write path. | Current-state index overclaim rules, Geist/ISM roadmap guard, and absence from v1.x integration tests. |
| Blue-Brain/HH/Microcircuits/DBM | These areas are out of v1.x scope. | Production Blue-Brain, HH, biological microcircuit, DBM/HPA, vendor-chip, or full metabolic-layer claims. | Current-state index, module registry overclaim rules, and absence from v1.x tests/gates. |

## 5. Test / Gate Baseline

| Test / Command | Proves | Required for Freeze? | Last known status |
|---|---|---:|---|
| `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` | v1.0 allow path, deny/no-mutation path, canonical records, evidence/archive append/readback, and deterministic replay. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test -p ucf-gateway --test minimal_spine_read_api -- --nocapture` | v1.1 internal read-only health/evidence/output-event/archive-root audit surface. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test -p ucf-ess --all-targets` | v1.2 ESS derived projection/read-model behavior and read-only boundary. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test -p ucf-consolidation --all-targets` | v1.3 candidate-only micro hook behavior. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test -p ucf-neuromod --all-targets` | v1.4 derived neuromod envelope, bounded hints, deterministic digest, and no override authority. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test -p ucf-protocol --all-targets` | Protocol record and canonical encoding stability for v1.x records. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo test --workspace` | Workspace-wide regression coverage for the frozen basis and supporting crates. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo clippy --workspace --all-targets -- -D warnings` | Workspace lint cleanliness with warnings denied. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo fmt --check` | Formatting stability. | yes | PASS in 2026-05-13 freeze validation. |
| `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json` | Strict docs lint and generated docs-lint report freshness for the current run. | yes | PASS in 2026-05-13 freeze validation; report remains uncommitted. |
| `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | Test-profile readiness gate for current HEAD/run. | yes | PASS in 2026-05-13 freeze validation; report remains uncommitted. |

## 6. Allowed Claims

After the v1.x freeze, the following claims are allowed when the listed tests/gates are current for the evaluated HEAD:

- Minimal Spine v1.x has a deterministic canonical record path for candidate sets, output records, evidence, archive append/readback, and replay assertions.
- `CandidateSetRecord` and `OutputRecord` are protocol authority records for the Minimal Spine v1.x path.
- Evidence and Archive components form the canonical append/readback proof surface used by Minimal Spine v1.x.
- The router E2E test proves the v1.0 allow path, deny/no-mutation path, and deterministic replay path.
- Gateway exposes an internal read-only audit service for Minimal Spine evidence/archive/output commitments.
- ESS projects derived read-model summaries and does not become event-log or append authority.
- Consolidation exposes a derived micro milestone candidate hook and does not run macro consolidation or replay scheduling.
- Neuromod exposes a derived metadata envelope with bounded hints and no policy/output override authority.
- Capability issuance is explicitly deferred for Minimal Spine v1.x.
- Root `out/*.json` reports are evidence only for the HEAD/run whose metadata they describe, and should normally not be committed as self-referential truth.

## 7. Forbidden Claims

The v1.x freeze explicitly forbids these claims unless a future explicit prompt, code implementation, negative tests, and gates promote them:

- No full UCF cognitive loop claim.
- No real compute claim.
- No full metabolic layer claim.
- No DBM/HPA integration claim.
- No full consolidation claim.
- Post-freeze consolidation work is tracked separately in `docs/roadmap/full_consolidation_roadmap_boundary_audit.md`; it currently proves only bounded Micro/Meso explicit append/readback, Macro candidate, and local consolidation-level finalization, with no Replay/Sleep/Geist/ISM/identity/Gateway/capability production claim.
- No Macro/Replay claim.
- No Geist/ISM runtime, store, identity, memory-stabilization, Gateway/action, Policy mutation, second event-log authority, or production readiness claim; post-freeze Geist/ISM docs may claim only the bounded candidate-only projection, verify-only audit, local ISM candidate boundary, E2E determinism, and audit/provenance append/readback proven outside Minimal Spine v1.x.
- No Gateway production/security claim.
- No HTTP API claim unless implemented and tested in a future scope.
- No capability issuance claim.
- No Blue-Brain/HH/microcircuit production claim.
- No external-service or distributed-runtime claim.
- No claim that derived ESS, Consolidation, or Neuromod views are canonical event authorities.
- No claim that v1.x readiness equals production readiness.

## 8. Freeze Criteria

Minimal Spine v1.x is frozen only when all of the following hold for the evaluated HEAD:

- All v1.x tests pass.
- `docs/minimal_ucf_spine_v1.md` is current for the v1.x scope.
- `docs/module_implementation_depth_registry.md` contains the v1.x status and overclaim boundaries.
- `docs/current_state_architecture_index.md` points to the v1.x freeze document and truth-order context.
- The readiness gate test profile passes.
- Strict docs lint passes.
- `cargo fmt --check` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Root `out/*.json` reports are not committed as self-referential truth unless a release workflow explicitly requires them and freshness metadata matches the evaluated HEAD.
- Capability issuance remains deferred unless a future explicit prompt changes it.
- All derived modules remain non-authoritative.

## 9. Freeze Invalidators

Any of the following invalidates this v1.x freeze and requires Full-Spine revalidation or a new versioned prompt:

- Changing `CandidateSetRecord` or `OutputRecord` canonical encoding.
- Changing Evidence/Archive authority or append/readback semantics used by the spine.
- Adding a Gateway write path into the spine.
- Granting ESS append or event-log authority.
- Wiring Consolidation macro or replay triggers into v1.x.
- Allowing Neuromod to override policy or output decisions.
- Adding active capability issuance.
- Adding real compute as a required v1.x dependency.
- Using stale reports as current truth.
- Removing, ignoring, or weakening v1.x tests.
- Promoting Gateway HTTP/security, Geist/ISM, Blue-Brain, HH, microcircuit, DBM/HPA, vendor-chip, or distributed runtime claims into v1.x without an explicit future prompt and tests.

## 10. v1.x Maintenance Rules

- Update this document when any v1.x component path, authority boundary, required test, freeze criterion, invalidator, or allowed/forbidden claim changes.
- A new prompt is required for behavior-changing changes to protocol schemas, canonical encodings, evidence/archive authority, Gateway writes, capability issuance, real compute dependency, Replay Scheduler, Geist/ISM, DBM/HPA, Blue-Brain/HH/microcircuit integration, or full consolidation.
- Full-Spine revalidation is required when a freeze invalidator is touched, when a derived module is promoted toward authority, when required tests/gates change, or when report freshness is used as release evidence.
- Generated reports must be treated as fresh only when generated for the evaluated HEAD and command/profile. Root `out/*.json` reports should normally remain uncommitted.
- v1.6 work may add another bounded, documented, non-overclaiming slice only if it preserves v1.x authority boundaries or explicitly updates this freeze. v2 work may change larger architecture assumptions, but must not retroactively rewrite v1.x claims.

## 11. Next Roadmap Blocks

Prioritized post-freeze roadmap blocks:

1. **Real Compute Optional Lane** - Add a first real/stub-separated backend lane as optional evidence only, with no required v1.x dependency and no production compute claim.
2. **Full Micro -> Meso -> Macro Consolidation** - Promote the v1.3 micro candidate toward a real milestone pipeline, but keep replay out until explicitly introduced.
3. **Replay Scheduler v1** - Define deterministic replay tokens and scheduler boundaries, with no Geist write until a later prompt.
4. **Geist/ISM Minimal Hook** - Start with self-state projection only and avoid identity finalization or recursion authority.
5. **Metabolic Scheduler / DBM-HPA** - Move from derived neuromod metadata toward a bounded scheduler, while preserving no policy override.
6. **Gateway HTTP/Security Hardening** - Introduce transport, auth, rate-limit, and security hardening read-only first; write later only behind a dedicated boundary.
7. **Capability Issuance Subsystem** - Implement only after an explicit subject/scope/resource/action model, revocation/expiration semantics, audit evidence, and negative tests.
8. **Prod-profile Readiness** - Move from the test profile to a stricter profile after the preceding authority boundaries are validated.

Recommended next prompt: `UCF Prompt 13 - Post-Freeze Roadmap Selection and Prompt Series Plan`.
