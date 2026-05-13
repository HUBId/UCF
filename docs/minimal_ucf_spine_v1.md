# Minimal UCF Spine v1 Specification

## 0. Purpose

Minimal UCF Spine v1 defines the smallest deterministic, auditable, CI-capable UCF through-path:

`ControlFrame / frame input -> deterministic route -> policy gate -> decision/output candidate -> evidence record -> archive append -> deterministic readback/query -> canonical E2E test`.

This document is the canonical technical basis for Prompt 5. It is intentionally not a complete UCF whitepaper, production-readiness claim, real-compute claim, Blue-Brain claim, or full cognitive-loop specification.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `9d57136085cbf514a95cf525abbf6d6a8fe9a171` |
| HEAD short | `9d571360` |
| Dirty state at spec creation | clean |
| Workspace package count | 192 |
| Current-State Index | `docs/current_state_architecture_index.md` |
| Module Implementation-Depth Registry | `docs/module_implementation_depth_registry.md` |

Baseline commands used for this spec: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -10`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, presence checks for the current-state index and module registry, and a targeted `find` over candidate module files.

## 2. Scope

### 2.1 In Scope

Spine v1 requires only the modules needed to prove a deterministic evidence-policy-route-archive slice without external services:

- `protocol/crates/ucf-protocol`: protocol-facing ControlFrame, PolicyDecision, ExperienceRecord, canonical encoding, and protocol boundary authority.
- `core/crates/ucf-types`: shared deterministic primitives, digest/ID/value helper types, and re-exported protocol v1 types where already used.
- `core/crates/ucf-policy-ecology`: deterministic allow/deny policy gate.
- `core/crates/ucf-router`: E2E test host and route/decision coordinator, using a deliberately narrow non-real-compute path.
- `core/crates/ucf-evidence`: evidence envelope/store surface.
- `domains/archive/crates/ucf-archive`: evidence append surface and compact ExperienceRecord helpers.
- `domains/archive/crates/ucf-archive-store`: deterministic local archive record append/readback/root-commit surface.
- `runtime/ucf-ops`: validation tooling only; not part of the runtime loop.

The required flow is:

1. deterministic ControlFrame or frame-like input,
2. normalization into a canonical spine input envelope,
3. explicit policy gate,
4. deterministic route/decision candidate,
5. evidence envelope with stable IDs/digests,
6. archive append,
7. deterministic readback/query,
8. canonical allow and deny/suppress E2E assertions.

### 2.2 Optional in v1

- `core/crates/ucf-bus`: optional in-memory message hop if it remains simple and deterministic.
- `ucf-sdk`: optional boundary helper if the test needs SDK-level deterministic encoding without creating schema ambiguity.
- `domains/ucf-frames`: optional adapter/source of frame vocabulary if it can map clearly to protocol authority.
- `domains/ucf-ess`: optional read model only if it is kept subordinate to archive/evidence authority.
- `runtime/ucf-runtime`: optional later host if Prompt 5 decides router-hosted E2E is insufficient.

### 2.3 Explicitly Out of Scope

Spine v1 does not require or validate:

- real compute, Burn, Candle, LLM, LFM, JEPA, SAE, SSM, NSR, or production AI/ML inference;
- Blue-Brain, HH, microcircuit, biophysical, DBM, or vendor/chip runtimes;
- full Geist recursion or full ISM authority;
- full Micro -> Meso -> Macro consolidation;
- Gateway HTTP/local transport API;
- external services, network dependencies, distributed production bus, or production database;
- durable database semantics beyond deterministic local archive/file or in-memory test stores.

## 3. Spine v1 Module Decisions

| Module | Role candidate | Include in Spine v1? | Reason | Required contract | Exclusions / caveats |
|---|---|---:|---|---|---|
| `core/crates/ucf-types` | Shared deterministic primitive, digest/ID/value-type layer | required | Provides fixed-point helpers, canonical digest/ID types, logical/wall-time wrappers, and compatibility re-exports used by spine candidates. | Use for `Digest32`, `EvidenceId`, `LogicalTime`, `WallTime`, fixed-point values, and stable helper/value types only. | Must not become protocol record authority for new protocol-facing records unless explicitly versioned there; avoid expanding broad conceptual surfaces. |
| `protocol/crates/ucf-protocol` | Protocol-facing schema and canonical encoding authority | required | Defines `v1::spec::ControlFrame`, `PolicyDecision`, `ExperienceRecord`, canonical encode/decode helpers, and boundary `ControlFrameV1`/`Envelope` digests. | Authoritative source for ControlFrame-like protocol records, PolicyDecision shape, ExperienceRecord shape, and canonical bytes. | Existing overlapping SDK/boundary types are adapters/helpers; Prompt 5 must avoid duplicate new protocol schemas. |
| `core/crates/ucf-bus` | In-memory deterministic message hop | optional | Small dependency footprint and simple in-memory publisher/subscriber can carry a frame if a bus hop is useful. | If used, one local in-memory hop only with fixed logical/wall time and no ordering ambiguity beyond single message. | Not a distributed production bus; not required for canonical E2E. |
| `core/crates/ucf-evidence` | Evidence envelope and evidence store surface | required | Provides `EvidenceEnvelope`, `EvidenceStore`, in-memory store, file store, evidence IDs, logical/wall time, and append/get semantics. | Evidence append must return/carry a deterministic `EvidenceId`; readback by evidence ID must be asserted. | Current trait `append` does not report failure for in-memory append; file-backed failures must not be silently ignored if selected. |
| `domains/archive/crates/ucf-archive` | ExperienceRecord-to-evidence append adapter and fold-capable archive | required | Provides `ExperienceAppender`, `build_compact_record`, `InMemoryArchive`, `FileArchive`, and append/fold support over evidence records. | Build/append the spine evidence record or an existing compact `ExperienceRecord`; prove readback/list or selected append result. | Fold/snapshot is not required for v1 unless Prompt 5 elects a trivial non-ambiguous fold assertion. |
| `domains/archive/crates/ucf-archive-store` | Deterministic archive record append/readback/root commit | required | Provides `ArchiveAppender`, `ArchiveStore`, `ArchiveRecord`, `RecordKind`, `RecordMeta`, in-memory append/get/iter/root-commit tests. | Append a deterministic record keyed by digest and assert `get(key)` plus stable root/digest behavior. | It stores record metadata/payload commits, not full production database semantics. |
| `core/crates/ucf-router` | E2E test host and deterministic route/decision coordinator | required | Existing router tests already integrate policy, archive, archive-store, and deterministic mock/stub ports. It is closer to the route decision than `runtime/ucf-runtime`. | Host `tests/minimal_spine_e2e.rs`; route path must be narrow, deterministic, and explicitly no-real-compute. | Router has a broad dependency footprint and many optional/stub integrations; Prompt 5 must avoid overclaiming full runtime capability. |
| `core/crates/ucf-policy-ecology` | Deterministic policy gate | required | Provides `PolicyEcology`, explicit rules, `ReplayGate`, `RiskDecision`, and policy tests for deny behavior. | Produce an explicit allow/deny or suppress decision with reason/code and evidence metadata. Deny path must be testable. | Existing gates are minimal; policy pack immutability or production enforcement is outside v1. |
| `runtime/ucf-ops` | Validation/docs/readiness tooling | required for validation | Provides docs lint, readiness gates, and operational checks; not in the runtime flow. | Prompt 5 must run docs lint and feasible Rust checks. | Must not be treated as evidence-policy-route runtime authority. |
| `ucf-sdk` | Stable external helper/API surface | optional | Provides deterministic `ControlFrameV1`, `DecisionEventV1`, ESS query/response helpers with narrow boundary API. | Use only as an adapter if it reduces boilerplate and does not conflict with `ucf-protocol` authority. | SDK types are not primary protocol schema authority for new spine records. |
| `domains/ucf-frames` | Domain frame vocabulary | optional | Contains frame modules including control/decision/neuromod/archive vocabulary. | If used, document exact adapter from `ucf-frames` to `ucf-protocol` ControlFrame/ExperienceRecord. | Broad domain vocabulary includes biophys/brain/microcircuit material; not all frame types are Spine v1. |
| `domains/ucf-ess` | Domain-specific experience/state store/read model | later | Provides rich `ExperienceRecord` domain state, retrieval, governance, output, neuromod, and summary surfaces. | Later read-model integration may consume archive/evidence output. | Not canonical event append authority for v1; avoid archive-vs-ESS ambiguity. |
| `runtime/ucf-runtime` | Broad runtime/orchestration host | later | Useful future integration host but depends on many compute, biophys, ESS, and cognitive-loop crates. | May host Spine v1.1/v2 once minimal route is already proven. | Too feature-heavy for first canonical E2E; no real compute required. |
| `runtime/ucf-gateway` | Local API surface | later | Tested local gateway can expose reads later. | Spine v1.1 may add read API over archived evidence. | HTTP/local transport/security audit not required for v1. |
| `runtime/ucf-client` | Gateway client | later | Useful only when gateway is in scope. | Can support gateway smoke tests after v1. | Not required without gateway. |
| `ucf-compute` and AI backend families | Real/stub compute lane | excluded | Compute is explicitly non-required and risks false AI/ML inference claims. | None for v1. | No Burn/Candle/LLM/LFM/JEPA/SAE/SSM/NSR backend required. |
| Blue-Brain/HH/microcircuit/DBM/vendor chip dirs | Advisory/deferred domains and vendor/reference material | excluded | Not needed to prove minimal evidence-policy-route-archive determinism. | None for v1. | No authority in Spine v1 decisions or acceptance tests. |

## 4. Schema Authority

| Concern | Decision | Reason | Affected modules |
|---|---|---|---|
| ControlFrame authority | `ucf-protocol` is the code-near schema/message authority for protocol-facing ControlFrame records. | `v1::spec::ControlFrame` has canonical encode/decode tests, while boundary/SDK frame types are narrower helper surfaces. | `protocol/crates/ucf-protocol`, `core/crates/ucf-types`, `ucf-sdk`, `domains/ucf-frames` |
| Experience/Evidence record authority | `ucf-protocol::v1::spec::ExperienceRecord` is the protocol-facing record schema; `ucf-evidence::EvidenceEnvelope` is the append envelope carrying evidence ID, proof, fold proof, and test time. | Archive and evidence code already build/store `ExperienceRecord` payloads inside `EvidenceEnvelope`. | `protocol/crates/ucf-protocol`, `core/crates/ucf-evidence`, `domains/archive/crates/ucf-archive` |
| Digest/ID authority | `ucf-types` provides shared `Digest32`, `EvidenceId`, logical/wall time, algorithm/domain digest helpers; protocol schema may carry serialized digest fields when record-facing. | Shared primitives avoid duplicate digest/ID wrappers, while protocol keeps message layout authority. | `core/crates/ucf-types`, `protocol/crates/ucf-protocol`, archive/evidence crates |
| Output/Decision record authority | Existing `ucf-protocol::v1::spec::PolicyDecision` is the policy/decision schema seed for Spine v1; any `SpineOutputCandidate` introduced in Prompt 5 must be a small test-local envelope or a clearly versioned protocol record. | Existing `PolicyDecision` already has kind/action/rationale/confidence/constraints; no new broad output schema is justified in the spec phase. | `protocol/crates/ucf-protocol`, `core/crates/ucf-policy-ecology`, `core/crates/ucf-router` |
| Archive manifest/hash authority | `ucf-archive-store` is the deterministic archive record/key/root-commit authority; `ucf-archive`/`ucf-evidence` are the evidence payload append/readback surface. | `ArchiveAppender` and `ArchiveStore` define deterministic keys/root commits; evidence stores define evidence IDs/envelopes. | `domains/archive/crates/ucf-archive-store`, `domains/archive/crates/ucf-archive`, `core/crates/ucf-evidence` |

Rule: Spine v1 may use both `ucf-types` and `ucf-protocol`, but any new record authority must be unambiguous. Protocol-facing records belong in `ucf-protocol`; shared primitives belong in `ucf-types`; test-only glue belongs in the E2E test unless promoted by a later schema change.

## 5. Persistence / Evidence Authority

| Concern | Decision | Reason | Affected modules |
|---|---|---|---|
| Canonical event append | Spine v1 canonical event append is the combination of `ucf-evidence`/`ucf-archive` for evidence envelopes plus `ucf-archive-store` for deterministic archive record append/readback. | This proves both full evidence envelope storage and digest-keyed archive metadata without a production DB. | `core/crates/ucf-evidence`, `domains/archive/crates/ucf-archive`, `domains/archive/crates/ucf-archive-store` |
| Evidence ID generation or carrying | The minimal test must carry deterministic `EvidenceId` derived from the record ID or an explicitly documented digest/id rule; append result must match the expected ID. | `ucf-archive` currently maps `ExperienceRecord.record_id` to `EvidenceId`; `ucf-evidence` stores by `EvidenceId`. | `ucf-types`, `ucf-evidence`, `ucf-archive` |
| Archive readback | Required. Prompt 5 must assert evidence readback/list from the selected evidence/archive surface and archive-store `get(record.key)`. | Minimal persistence proof requires readback by ID or digest/key, not just append success. | `ucf-evidence`, `ucf-archive`, `ucf-archive-store` |
| Fold/snapshot | Later by default; optional in v1 only if using existing `append_and_fold` without adding scope or ambiguity. | Fold state exists but is not necessary to prove the minimal spine. | `ucf-archive`, `ucf-fold` |
| ESS status | Later/optional read model, not canonical v1 event log. | ESS is broad domain state/retrieval/governance and overlaps with archive/evidence if promoted too early. | `domains/ucf-ess` |
| File vs in-memory store status | In-memory store is sufficient for canonical CI E2E; file store is optional if Prompt 5 wants local-disk proof with tempdir and no committed artifacts. | In-memory is deterministic and service-free; file store adds IO failure handling that can be tested separately. | `ucf-evidence`, `ucf-archive`, `ucf-archive-store` |
| No silent failure | Required. The test must assert append IDs, store length/readback, archive key lookup, and stable digest/root behavior. File-backed paths must propagate `StoreResult` failures if selected. | A spine without explicit persistence assertions can hide evidence failures. | all persistence candidates |

## 6. Spine v1 Flow

| Step | Required module(s) | Input | Output | Determinism assertion | Failure behavior |
|---|---|---|---|---|---|
| 1. Input | `ucf-protocol`, `ucf-types` | Fixed test ControlFrame or boundary `ControlFrameV1` with fixed IDs, times, nonce, policy ID, and digest. | Canonical input bytes/digest and `SpineInputEnvelope` metadata. | Encoding/digest of the same input is identical across two runs. | Invalid or missing required fields must fail the test before route/archive mutation. |
| 2. Normalize | `ucf-protocol`; optional `ucf-frames` adapter | Raw ControlFrame/frame-like input. | Canonical spine input envelope with input ID/digest and provenance. | Normalized forms sort unordered evidence/constraint lists where existing normalized types support that. | Ambiguous adapter mapping is rejected; no archive append before policy. |
| 3. Policy Gate | `ucf-policy-ecology` | Canonical input envelope or compact `ExperienceRecord` view. | `SpinePolicyDecision` allow/deny/suppress with reason/code. | Same input and same policy rules produce same decision and reason/code. | Deny/suppress path must be explicit; unauthorized output must not be produced or archived as allow. |
| 4. Route / Decide | `ucf-router`, `ucf-policy-ecology` | Allowed input plus policy decision. | Deterministic `SpineOutputCandidate` or existing `PolicyDecision`/output metadata. | Output candidate ID/digest is stable across two runs. | Policy denial returns explicit denied/suppressed result; no real compute fallback. |
| 5. Evidence | `ucf-protocol`, `ucf-types`, `ucf-evidence`, `ucf-archive` | Input reference, policy result, output candidate reference/status. | `SpineEvidenceEnvelope` or existing `ExperienceRecord` plus `EvidenceEnvelope`. | Evidence ID/digest and canonical payload bytes are stable. | Evidence build failure aborts; no silent success without evidence metadata. |
| 6. Archive | `ucf-archive`, `ucf-archive-store`, `ucf-evidence` | Evidence record/envelope and payload commit. | `ArchiveAppendResult` or existing append ID/key/root commit. | Archive key/readback/root commit is stable for the same append sequence. | Append/readback mismatch fails; file IO errors are propagated if using file path. |
| 7. Validation | `ucf-router`, `ucf-ops` for docs/checks | Completed allow path and stored evidence/archive record. | Assertions for output, evidence, archive readback, policy path, digest. | Run the same path twice and assert stable IDs/digests/readbacks. | Any unstable digest, missing evidence, or missing archive record fails. |
| 8. Negative path | `ucf-policy-ecology`, `ucf-router`, archive/evidence surface if denied evidence is intended | Denied/suppressed input. | Explicit denied/suppressed result, optionally denied evidence if policy design requires. | Deny result/reason/digest is stable across two runs. | No unauthorized output/archive mutation; alternatively exactly one explicit denied evidence record is asserted. |

Optional v1 flow extensions: a single `ucf-bus` in-memory hop, `ucf-sdk` deterministic encoding helper, or an ESS read model if trivial and subordinate. Explicitly later: gateway read API, real compute, neuromod modulation, consolidation, Geist recursion, and Blue-Brain advisory diagnostics.

## 7. Record / Envelope Contracts

| Record | Existing type? | Source module | Required fields | Digest/ID behavior | Notes |
|---|---:|---|---|---|---|
| `SpineInputEnvelope` | partial | Existing `ucf-protocol::v1::spec::ControlFrame`, `ucf_protocol::boundary::v1::ControlFrameV1`, or test-local wrapper | `version`, `input id/frame_id/control_id`, `input digest`, `policy id/class`, deterministic test timestamp/cycle/nonce, `provenance/source module` | Digest must come from canonical bytes or `Digest32` helper and be stable across replay. | Prefer existing protocol ControlFrame. A new reusable type is not required unless Prompt 5 proves a gap. |
| `SpinePolicyDecision` | partial | `ucf-protocol::v1::spec::PolicyDecision`; `ucf-policy-ecology` decision/rules | `version`, `policy id/version`, `decision kind/status`, `action`, `reason/code/rationale`, `confidence/bounds`, constraints/rules hit | Decision digest, if added, must hash canonical decision fields and fixed policy identity. | Allow and deny/suppress must both be represented explicitly. |
| `SpineOutputCandidate` | no exact required type | Test-local wrapper or existing router output metadata; possible future protocol record | `version`, `route/output id`, `status`, `input digest`, `policy decision ref`, `payload/output commit`, `provenance`, no-real-compute marker | Output ID/digest must be deterministic and must not claim ML inference. | Keep test-local for Prompt 5 unless code already has a clean equivalent in the route path. |
| `SpineEvidenceEnvelope` | partial | `ucf-protocol::v1::spec::ExperienceRecord` plus `ucf-evidence::EvidenceEnvelope` | `version`, `evidence id/record_id`, deterministic observed/test time, `subject_id`, canonical payload, input ref, policy ref, output ref/status, optional proof/fold refs | Evidence ID should be deterministic; digest should be canonical payload or record digest. Append/readback must preserve it. | Existing `EvidenceEnvelope` lacks a version field; version may be encoded in payload/record subject/schema string for the first test. |
| `ArchiveAppendResult` | partial | `ucf-archive::ExperienceAppender`, `ucf-archive-store::ArchiveStore::append` | appended `EvidenceId` and/or archive `key`, `payload_commit`, `root/record commit`, `status`, store length/readback signal | Archive key/root commit must be deterministic for fixed append sequence. | No production DB contract; use in-memory or tempdir local file path. |
| `SpineReadbackResult` | partial | `ucf-evidence::EvidenceStore::get`, `ucf-archive` list/readback, `ucf-archive-store::get` | `status`, requested `evidence id` or `archive key`, returned record/envelope, digest/key/root comparison | Readback digest/key must equal the append result and original expected value. | Missing readback is a failure, not a warning. |

If Prompt 5 needs new types, it must keep them bounded to the E2E test or add a small, clearly versioned record in the correct authority module. It must not introduce a broad new schema family.

## 8. Canonical E2E Test Plan

### 8.1 Test placement

The first canonical E2E test should be placed at:

`core/crates/ucf-router/tests/minimal_spine_e2e.rs`

Rationale: the router is the required route/decision coordinator and already has integration tests wiring policy, archive, archive-store, deterministic mocks/stubs, and protocol ControlFrames. `runtime/ucf-runtime` is broader and pulls in compute/biophys/cognitive-loop dependencies that are not required for v1. Router-hosted E2E therefore minimizes new behavior while proving the required route seam.

### 8.2 Required test cases

| Test case | Expected behavior | Required modules |
|---|---|---|
| Allow path appends evidence and archive readback works | Fixed allowed ControlFrame produces an explicit allow decision/output candidate, deterministic evidence ID/digest, archive append, evidence readback, and archive-store key readback. | `ucf-protocol`, `ucf-types`, `ucf-policy-ecology`, `ucf-router`, `ucf-evidence`, `ucf-archive`, `ucf-archive-store` |
| Deny/suppress path behaves safely | Fixed denied input produces explicit deny/suppress reason; no unauthorized allow output or archive mutation occurs, or exactly one explicit denied evidence record is appended if that policy is chosen. | `ucf-policy-ecology`, `ucf-router`, persistence surface if denied evidence is recorded |
| Deterministic replay | Running the same allow path twice produces the same canonical input bytes, policy decision, output candidate digest, evidence ID/digest, archive key/readback, and root/record commit for equivalent fresh stores. | same required modules |
| No real compute backend | Test passes with default workspace features and without Burn/Candle/LLM/LFM/JEPA/SAE/SSM/NSR backend calls. | `ucf-router` with narrow deterministic path |
| No external service | Test uses in-memory stores or tempdir local files only. | archive/evidence candidates |
| Explicit boundary exclusions | Test does not require Blue-Brain, HH, microcircuit, gateway HTTP, production DB, vendor chip dirs, full consolidation, full Geist, or neuromod modulation. | test review/assertions |


### 8.3 Prompt 5 implemented test path

Prompt 5 implements the first canonical Minimal UCF Spine v1 E2E test at:

`core/crates/ucf-router/tests/minimal_spine_e2e.rs`

Implemented semantics and APIs:

- protocol-facing input uses `ucf::v1::spec::ControlFrame`, `PolicyDecision`, `DecisionKind`, and canonical `ControlFrame::decode_canonical` / `canonical_bytes`;
- the explicit policy gate uses `ucf-policy-ecology::PolicyEcology` with `PolicyRule::DenyReplayIfDecisionClass` through the existing `ReplayGate` trait;
- the minimal route output candidate is a local deterministic test helper derived from canonical input bytes and policy decision; this is intentionally not a new production `OutputRecord` architecture;
- evidence append uses `ucf-archive::InMemoryArchive::append_with_proof`, which builds/stores the existing `ucf-evidence::EvidenceEnvelope` surface for an `ExperienceRecord`;
- archive-store readback/root proof uses `ucf-archive-store::InMemoryArchiveStore`, `ArchiveAppender`, `ArchiveStore::get`, and `ArchiveStore::root_commit`;
- deny semantics are Option 1: denied input is stopped before route output materialization, evidence append, and archive-store append. The test asserts zero evidence entries, zero archive output records, no archive key/commit/root, and no output candidate.

Boundary notes: the test does not call real compute, Gateway HTTP, external services, Blue-Brain, HH, microcircuit, vendor-chip, full consolidation, full Geist recursion, or neuromodulation paths.

### 8.4 Required commands for Prompt 5

Prompt 5 must run, at minimum:

```bash
cargo fmt --check
cargo test -p ucf-router --test minimal_spine_e2e
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
```

If feasible, Prompt 5 should also run:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ucf-ops --all-targets
```

`out/docs_lint_report.json` should normally remain uncommitted unless a release workflow explicitly requires report artifacts.

### 8.5 Non-goals for Prompt 5

- no real ML inference;
- no production database;
- no gateway security/API audit;
- no full consolidation;
- no Geist recursion;
- no neuromod decision modulation;
- no Blue-Brain/HH/microcircuit runtime proof.

## 9. Boundary Guarantees

Spine v1 guarantees only the bounded through-path described here:

- no real compute required;
- no advisory module authority;
- no external service;
- no production distributed bus;
- no production DB requirement;
- policy gate is explicit;
- archive append/readback is explicit;
- evidence metadata is explicit;
- deny/suppress path is explicit;
- deterministic repeatability is required;
- no silent evidence failure is acceptable;
- optional modules remain optional unless a later v1.1/v2 spec promotes them.

## 10. Non-Goals

Spine v1 is not:

- the complete UCF system;
- a production-readiness certificate;
- a full cognitive loop;
- a real AI/ML compute integration;
- a Blue-Brain/HH/microcircuit/biophys runtime;
- a gateway/API deliverable;
- a consolidation, Geist, or neuromodulation deliverable;
- a durable production database or distributed systems proof.

## 11. Prompt 5 Implementation Plan

Likely files to change:

- `core/crates/ucf-router/tests/minimal_spine_e2e.rs` — new canonical E2E test.
- Maybe `core/crates/ucf-router/Cargo.toml` — only if a missing dev-dependency is required.
- Maybe small compile/doc-link fixes if the new test exposes an existing missing import or doc-link issue.

Expected tests/checks:

- `cargo fmt --check`
- `cargo test -p ucf-router --test minimal_spine_e2e`
- `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
- If feasible: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p ucf-ops --all-targets`.

Risks:

- Router dependency footprint may tempt Prompt 5 to use broad mock/stub runtime paths; mitigation: keep a narrow deterministic route path and cite no-real-compute boundary.
- Archive/evidence overlap may cause authority ambiguity; mitigation: use `ExperienceRecord` + `EvidenceEnvelope` for evidence and `ArchiveStore` for key/readback/root proof.
- Deny path semantics may vary; mitigation: choose one explicit behavior and assert it strictly.
- Existing traits may not expose every desired readback from `InMemoryArchive`; mitigation: use the existing `list`, `EvidenceStore::get`, and `ArchiveStore::get` surfaces without production refactors.

## 12. Maintenance Rules

Update this spec whenever:

- the canonical E2E test placement changes;
- required modules are added, removed, renamed, or their public contracts change;
- schema authority between `ucf-types`, `ucf-protocol`, SDK, frames, ESS, archive, or evidence changes;
- persistence authority changes from in-memory/local file to another store;
- policy deny/suppress semantics change;
- gateway, compute, consolidation, Geist, neuromod, or advisory domains are promoted into a required spine lane.

Spine v1 becomes invalid if:

- the allow path no longer appends and reads back evidence/archive records deterministically;
- the deny/suppress path becomes implicit or silently mutates authorized-output/archive state;
- default tests require external services, real compute, gateway transport, Blue-Brain/HH/microcircuit runtime, or production DB;
- canonical encoding/digest/readback stability is lost.

Versioning rule:

- v1.1 additions must remain backward-compatible and may add optional gateway read API, bus hop, SDK helper, or ESS read model sections.
- v2 additions may promote consolidation, Geist/ISM, neuromod envelopes, real compute lanes, authority boundary tests, or runtime-hosted E2E only with explicit tests, gates, and updated module registry entries.
- Historical docs must not be deleted to simplify scope; they remain audit trail unless reclassified by the current-state index and registry.
