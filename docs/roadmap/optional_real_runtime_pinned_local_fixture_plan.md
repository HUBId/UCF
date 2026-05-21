# UCF OptionalRealRuntime Pinned Local Runtime Artifact Fixture Plan

## 0. Purpose
- Plan only.
- No runtime activation.
- No prod readiness claim.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD | `b0382da8473a463d1499106efbb82b100c09a778` |
| Dirty state | clean |
| Workspace package count | 192 |

Links:
- [`docs/roadmap/optional_real_runtime_artifact_fixture_inventory.md`](optional_real_runtime_artifact_fixture_inventory.md)
- [`docs/roadmap/optional_real_runtime_artifact_format_decision.md`](optional_real_runtime_artifact_format_decision.md)
- [`docs/roadmap/optional_real_runtime_prod_compute_semantics_roadmap.md`](optional_real_runtime_prod_compute_semantics_roadmap.md)
- OptionalRealRuntime contract tests:
  - [`runtime/ucf-compute/tests/optional_real_runtime_contract.rs`](../../runtime/ucf-compute/tests/optional_real_runtime_contract.rs)
  - [`runtime/ucf-compute/tests/backend_identity_contract.rs`](../../runtime/ucf-compute/tests/backend_identity_contract.rs)
  - [`runtime/ucf-compute/tests/optional_real_compile_gate.rs`](../../runtime/ucf-compute/tests/optional_real_compile_gate.rs)

## 2. Artifact / Fixture Surface Inventory

| Concern | Path | Current behavior | Fixture relevance | Gap |
|---|---|---|---|---|
| Metadata-only OptionalRealRuntime contract fields exist (`artifact_id`, `artifact_kind`, `artifact_digest`, `artifact_size_bytes`, `source_note`, `license_note`, `local_only`, `network_required`, `fixture_id`, `input_digest`, `expected_output_digest`, bounds). | `runtime/ucf-compute/src/runtime_contract.rs`, `runtime/ucf-compute/tests/optional_real_runtime_contract.rs` | Contract validation is strict; no active backend promoted. | Provides schema for pinned fixture manifest and acceptance checks. | No concrete pinned fixture files yet. |
| Golden-test convention uses deterministic digest assertions for compute outputs. | `runtime/ucf-compute/tests/toy_compute_golden.rs`, `runtime/ucf-compute/tests/compute_output_link.rs`, `runtime/ucf-compute/tests/compute_audit_records.rs` | Deterministic toy/stub digest checks exist. | Same pattern can verify expected runtime output digest deterministically. | OptionalRealRuntime-specific golden is absent. |
| Fixture directories already exist for compute tests. | `runtime/ucf-compute/fixtures/`, `fixtures/` | Local fixture assets and scenario fixtures are in-repo and offline. | Suitable precedent for local-only OptionalRealRuntime fixture location. | No dedicated `optional_real_runtime` fixture directory yet. |
| Hash convention in compute/ops is predominantly SHA-256 digests and hex prefixes. | `runtime/ucf-compute/src/contracts.rs`, `runtime/ucf-compute/src/ipc.rs`, `runtime/ucf-compute/src/stage_v1_candle.rs`, `runtime/ucf-ops/src/lib.rs` | Digest helpers and assertions rely on SHA-256 bytes/hex. | Reusing SHA-256 avoids introducing an extra digest algorithm in first fixture candidate. | No explicit runtime-artifact hash file format decision recorded. |
| Burn/Candle seams are feature-gated and currently compile-oriented. | `runtime/ucf-compute/Cargo.toml`, `runtime/ucf-compute/tests/optional_real_compile_gate.rs` | `backend-burn`/`backend-candle` compile lanes exist; runtime claim remains forbidden. | Future fixture backend invocation can stay opt-in and feature-gated. | No OptionalRealRuntime execution lane is wired. |
| CI matrix already has optional backend feature jobs and can host opt-in fixture lane. | `.github/workflows/ci.yml`, `docs/roadmap/compute_feature_ci_matrix.md`, `.github/workflows/nightly_verify.yml` | Compile/check jobs include candle/burn permutations; goldens lane exists. | A dedicated optional fixture job can be added without changing default gates. | No runtime fixture job currently exists. |
| Source/license documentation pattern exists in docs and manifest-driven workflows. | `README.md`, `docs/roadmap/*.md`, artifact/report docs | Policy/docs emphasize traceability and explicit boundaries. | Fixture manifest/README can require source and license notes before test enablement. | No runtime fixture-specific provenance template yet. |
| Repo-size discipline implied by tiny fixture approach; no large model artifacts in current plan. | roadmap docs and existing small test fixtures | Existing tests rely on compact fixture files. | Supports strict max-size cap for first candidate. | Explicit byte cap needs to be formalized. |

Answers (Phase 2):
- Geeignete testdata/fixture-Verzeichnisse: **ja** (`runtime/ucf-compute/fixtures/` and test-local fixture patterns).
- Bestehende Artefakt-/Model-Hash-Konvention: **ja**, SHA-256 is dominant in compute/ops.
- Bestehende Golden-Test-Konvention: **ja**, deterministic digest-based assertions.
- Feature-Gates für Burn/Candle-Seams: **ja**, via cargo features and compile-gate tests.
- CI-Matrix-Platz für opt-in runtime fixture: **ja**, optional feature jobs and golden lanes exist.
- Lizenz-/Source-Doku-Muster: **teilweise ja**, but runtime fixture-specific template missing.
- Größenlimits/repo-size constraints: **implizit**, explicit cap needed in this plan.

## 3. Candidate Fixture Strategy Evaluation

| Candidate | Pros | Cons | Artifact feasibility | Determinism | Repo-size risk | Overclaim risk | Recommendation |
|---|---|---|---|---|---|---|---|
| A. Tiny local Candle artifact fixture | Reuses safetensors-oriented code paths and candle helpers. | Could be misread as runtime activation if not tightly gated. | Medium | Medium-high once pinned | Medium | Medium | Secondary option after format decision. |
| B. Tiny local Burn artifact fixture | Aligns with existing compile lane focus (`backend-burn`). | Same overclaim risk; runtime semantics still absent today. | Medium | Medium-high once pinned | Medium | Medium | Secondary option after format decision. |
| C. Synthetic minimal local artifact wrapper, backend-neutral | Lowest immediate risk; cleanly separated from toy promotion and remote services; easiest offline pinning. | Needs careful wording to avoid becoming “toy v2”. | High | High (manifest + digest deterministic) | Low | Low-medium | **Choose first as planning baseline.** |
| D. Promote Toy golden to runtime fixture | Already deterministic. | Violates boundary: toy must not be promoted to real runtime. | High technically | High | Low | **High** | **Reject.** |
| E. Remote/external service fixture | Could mimic remote runtime topology. | Violates no-network/no-external-service rule for first candidate. | Low in scope | Low | N/A | **High** | **Reject.** |

Decision: choose **C** for Prompt 79H planning baseline; defer A/B implementation details to 79I or 79H2 after explicit artifact-format decision.

## 4. Artifact Policy Decision

| Policy item | Decision | Reason |
|---|---|---|
| artifact location | `runtime/ucf-compute/tests/fixtures/optional_real_runtime/` | Scoped to compute tests, local-only, deterministic fixture ownership. |
| max artifact size | `<= 256 KiB` hard cap for first candidate | Keeps repo small, avoids large model commits, supports fast deterministic CI. |
| hash algorithm | SHA-256 hex (lowercase), full 64-char digest | Matches dominant compute/ops digest patterns. |
| source/license note | Required in manifest and README (`source_note`, `license_note`) | Provenance and legal traceability required before enabling fixture tests. |
| network policy | no network | Offline-first repo rule and prompt constraints. |
| external service policy | no external service | First candidate must be local and deterministic. |
| deterministic input | required | Stable fixture input is prerequisite for reproducible digest checks. |
| deterministic output digest | required | Golden output check must be digest-pinned and repeatable. |
| timeout/cost bounds | required (`max_runtime_ms`, `max_memory_bytes`) | Prevents flaky/expensive optional lane behavior. |
| CI opt-in | required | Keep runtime fixture lane non-default and explicit. |
| production claim | forbidden | Fixture plan is not production/runtime-readiness proof. |

## 5. Fixture Manifest Plan

Planned path (not implemented now):

```text
runtime/ucf-compute/tests/fixtures/optional_real_runtime/
  README.md
  fixture_manifest.json
  input.fixture.json
  expected_output.digest
  artifact.bin (or selected tiny format)
  artifact.sha256
```

| Fixture file / field | Purpose | Required? | Notes |
|---|---|---:|---|
| `artifact_id` | Stable artifact identity | yes | Immutable once pinned for a fixture version. |
| `artifact_kind` | Declares artifact format/kind | yes | Set by 79H2/79I decision (backend-neutral first). |
| `artifact_digest` | Pinned artifact hash | yes | SHA-256 full hex digest. |
| `artifact_size_bytes` | Enforce small size cap | yes | Must be `<= 262144`. |
| `source_note` | Provenance statement | yes | Origin, generation method, and local-only statement. |
| `license_note` | License declaration | yes | SPDX-style identifier or explicit internal fixture note. |
| `local_only` | Local artifact guarantee | yes | Must be `true`. |
| `network_required` | Network requirement flag | yes | Must be `false`. |
| `fixture_id` | Fixture identity/version | yes | Separate from artifact identity. |
| `input_digest` | Digest of canonical input file | yes | SHA-256 of `input.fixture.json`. |
| `expected_output_digest` | Golden output digest | yes | SHA-256 of canonical serialized output bytes. |
| `max_runtime_ms` | Runtime budget bound | yes | Determinism and CI stability guardrail. |
| `max_memory_bytes` | Memory budget bound | yes | Protect CI/resource determinism. |
| `deterministic` | Determinism marker | yes | Must be `true`. |
| `offline` | Offline capability marker | yes | Must be `true`. |
| `external_service_required` | External dependency marker | yes | Must be `false`. |

## 6. Prompt 79I Acceptance Criteria

| Acceptance criterion | Required for 79I? | Notes |
|---|---:|---|
| Artifact format explicitly selected (backend-neutral or specific). | yes | If unresolved, run 79H2 first. |
| Artifact is tiny, local, and within size cap. | yes | Hard fail if cap exceeded. |
| Artifact hash is pinned and verified in tests. | yes | Must fail on mismatch. |
| Fixture manifest exists and validates strict contract fields. | yes | Reuse OptionalRealRuntime contract validator. |
| Deterministic fixture input is committed and digest-pinned. | yes | Canonical serialization required. |
| Expected output digest is deterministic and pinned. | yes | Golden digest check required. |
| Runtime backend invocation is feature-gated and opt-in only. | yes | No default-lane activation. |
| Test asserts no network and no external service requirement. | yes | Explicit checks on manifest/contract flags. |
| Test asserts no production claim. | yes | `production_claim=false` remains mandatory. |
| Test proves candidate satisfies OptionalRealRuntime contract metadata. | yes | No claim of prod runtime readiness. |
| CI integration is opt-in lane only. | yes | Non-default workflow job or matrix entry. |
| No Toy promotion, no Remote candidate substitution. | yes | Explicit negative tests or docs assertions. |
| Prod readiness still explicitly marked no. | yes | Must remain true after implementation. |

## 7. Current Prod Status
- Prod ready: **no**.
- OptionalRealRuntime: **absent** (active runtime lane not implemented).
- `backend-burn` compile lane pass: **insufficient** for runtime/prod semantics.
- This fixture plan: **not runtime activation**.

## 8. Open Questions
- Candle vs Burn vs backend-neutral tiny fixture for first implementation.
- Exact artifact format (`artifact.bin` wrapper vs tiny safetensors/other).
- Final hash algorithm lock (current recommendation: SHA-256).
- Exact license/source provenance wording requirements.
- Whether max size cap should stay 256 KiB or be tighter.
- Exact CI opt-in lane shape (dedicated job vs matrix include).
- Whether contract satisfaction test lands before any runtime invocation hook or in same prompt.

## 9. Recommended Next Prompt
- Preferred next: **UCF Prompt 79I-A — OptionalRealRuntime Fixture Files and Manifest Validation**.
- Then: **UCF Prompt 79I — Deterministic OptionalRealRuntime Golden Test Implementation**.

## Prompt 79I-A — Implementation Linkage

Implemented fixture artifacts (local/offline only):
- `runtime/ucf-compute/tests/fixtures/optional_real_runtime/artifact.fixture.bin`
- `runtime/ucf-compute/tests/fixtures/optional_real_runtime/input.fixture.json`
- `runtime/ucf-compute/tests/fixtures/optional_real_runtime/expected_output.fixture.bytes`
- `runtime/ucf-compute/tests/fixtures/optional_real_runtime/fixture_manifest.json`
- `runtime/ucf-compute/tests/fixtures/optional_real_runtime/README.md`

Implemented validator:
- `runtime/ucf-compute/tests/optional_real_runtime_fixture_manifest.rs`

Recommended next prompt:
- **UCF Prompt 79I-B — Deterministic OptionalRealRuntime Planned Golden Contract**.

## Prompt 79I-B-lite — Planned Golden Contract (Implemented)

- Added explicit planned deterministic golden-contract tests for `expected_output.fixture.bytes` digest semantics.
- Contract semantics now explicitly assert: digest exists, digest is deterministic, and digest validation is metadata/static only (not runtime inference proof).
- No backend invocation, no OptionalRealRuntime activation, no backend promotion, and no prod-readiness claim were introduced.
- Recommended next prompt: **UCF Prompt 79J-lite — OptionalRealRuntime ComputeOutputLink / Audit Metadata Integration** (alternative: **UCF Prompt 79I-C-lite — Runtime Invocation Boundary Planning**).

## Prompt 79J-lite — ComputeOutputLink / Audit Metadata Integration (Implemented)

- Added targeted fixture-link/audit test coverage for metadata-only OptionalRealRuntime planned-golden linkage.
- `ComputeOutputLink` now demonstrated (via test) to carry planned expected output digest as `compute_result_digest` with a deterministic synthetic non-authoritative output reference digest.
- `ComputeAuditRecord` is demonstrated to capture this linkage with non-runtime/non-production metadata-only status and deterministic audit digest.
- Boundaries preserved: no runtime invocation, no OptionalRealRuntime activation, no backend promotion, no prod-readiness claim.
- Recommended next prompt: **UCF Prompt 79I-C-lite — Runtime Invocation Boundary Planning**.
