# UCF OptionalRealRuntime Artifact Format Decision

## 0. Purpose
- Format decision only.
- No runtime activation.
- No prod readiness claim.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD | `04e1d377f19bb56b22b31f2bf553939c58554294` |
| Dirty state | clean |
| Workspace package count | 192 |
| Pinned fixture plan present | yes |
| Artifact fixture inventory present | yes |
| OptionalRealRuntime contract test present | yes |
| `ucf-compute` present | yes |

Links:
- [`docs/roadmap/optional_real_runtime_pinned_local_fixture_plan.md`](optional_real_runtime_pinned_local_fixture_plan.md)
- [`docs/roadmap/optional_real_runtime_artifact_fixture_inventory.md`](optional_real_runtime_artifact_fixture_inventory.md)
- [`runtime/ucf-compute/tests/optional_real_runtime_contract.rs`](../../runtime/ucf-compute/tests/optional_real_runtime_contract.rs)

## 2. Existing Format / Fixture Inventory

| Concern | Path | Existing pattern | Relevance | Gap |
|---|---|---|---|---|
| OptionalRealRuntime metadata contract fields (`artifact_id`, `artifact_kind`, `artifact_digest`, `artifact_size_bytes`, `source_note`, `license_note`, `fixture_id`, `input_digest`, `expected_output_digest`) | `runtime/ucf-compute/src/runtime_contract.rs`, `runtime/ucf-compute/tests/optional_real_runtime_contract.rs` | Strict metadata contract, no active runtime lane | Direct basis for manifest required fields | No pinned fixture files yet |
| Deterministic digest-based fixtures/goldens | `runtime/ucf-compute/tests/toy_compute_golden.rs`, `runtime/ucf-compute/tests/stub_compute_fixture.rs` | SHA-256 style deterministic digest assertions | Reusable semantics for input/output digest pinning | OptionalRealRuntime-specific fixture absent |
| Derived metadata boundaries for compute outputs/audit | `runtime/ucf-compute/tests/compute_output_link.rs`, `runtime/ucf-compute/tests/compute_audit_records.rs` | Link/audit records stay metadata-only and non-authoritative | Guardrail against overclaim during fixture onboarding | Optional runtime fixture coverage missing |
| Optional-real compile gate exists and forbids runtime/prod claim | `runtime/ucf-compute/tests/optional_real_compile_gate.rs` | Compile-only remains non-runtime inference | Confirms no-runtime-activation constraint | Runtime class remains unactivated |
| Artifact schema snapshot tooling exists in docs/ops | `docs/artifact_schema_snapshots/*.json`, `.github/workflows/ci.yml` (`spec artifact-schemas-check`) | JSON schema snapshot convention for artifacts | Supports JSON-first manifest documentation fit | No dedicated OptionalRealRuntime fixture manifest snapshot yet |
| Fixture directory convention exists | `runtime/ucf-compute/fixtures/`, `fixtures/` | In-repo offline fixture files | Fits local-only fixture requirement | Dedicated `tests/fixtures/optional_real_runtime/` not yet present |
| CI optional lanes and matrix documentation exist | `docs/roadmap/compute_feature_ci_matrix.md`, `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml` | Optional compile-lanes and non-default checks documented | Supports opt-in-only runtime-fixture lane | No explicit optional runtime fixture lane |
| JSON/TOML usage in repo | `docs/artifact_schema_snapshots/*.json`, `models/manifest.toml`, `policies/manifest.toml` | JSON for artifact schemas, TOML for package/model manifests | Suggests JSON for test fixture manifest is natural for artifact schema workflows | Final format had not been explicitly decided |

Antworten (Phase 2):
- JSON/TOML manifest patterns exist: **yes** (JSON artifact-schema snapshots; TOML model/policy manifests).
- `ucf-compute` digest usage: **SHA-256 is already used in tests/helpers**; no need to introduce BLAKE3 for this fixture decision.
- Stub/Toy pinned digest style: **deterministic digest assertions in tests**.
- Existing fixture directories: **yes**.
- Artifact schema snapshot tooling reusable: **yes** (JSON schema snapshot conventions and ops check surface).
- `serde_json` dependency need: **for 79H2 docs-only decision, manifest remains documentation-first; 79I may parse JSON in tests if needed**.

## 3. Format Options

| Option | Pros | Cons | Determinism | Tooling fit | Recommendation |
|---|---|---|---|---|---|
| A. JSON manifest + binary artifact bytes | Aligns with artifact-schema JSON conventions and deterministic digest pinning | Requires explicit canonical hashing rules | High | High | **Choose** |
| B. TOML manifest + binary artifact bytes | Matches model/policy TOML usage | Less aligned with artifact-schema JSON snapshots; more custom parsing in tests | High | Medium | No |
| C. Rust const fixture only, no files | Avoids file IO | Harder provenance/source/license separation; less reusable for CI artifact checks | Medium | Medium-low | No |
| D. Single self-describing `.fixture` file | One file to ship | Adds custom format surface and parser risk early | Medium | Low | No |
| E. Schema-only docs for now | Lowest implementation risk | Blocks deterministic fixture realization in 79I | High (docs only) | Medium | Not selected as final format decision |

## 4. Artifact Format Decision

| Concern | Decision | Reason |
|---|---|---|
| manifest format | JSON file: `fixture_manifest.json` | Best fit with existing artifact schema JSON patterns and explicit field validation |
| artifact bytes format | Opaque tiny binary: `artifact.fixture.bin` | Backend-neutral synthetic wrapper with strict size and digest pinning |
| input fixture format | JSON file: `input.fixture.json` | Human-reviewable deterministic test input with pinned digest |
| output digest format | SHA-256 hex in manifest field `expected_output.sha256` | Mirrors existing digest-pin golden conventions |
| hash algorithm | SHA-256 | Required by Prompt 79H |
| max size | `<= 256 KiB` | Required by Prompt 79H |
| schema enforcement | Phase 79I test validation against strict required fields + value rules | Keeps this prompt docs-only while enabling deterministic implementation next |

Directory/file decision (79I target):

```text
runtime/ucf-compute/tests/fixtures/optional_real_runtime/
  fixture_manifest.json
  artifact.fixture.bin
  input.fixture.json
```

No runtime activation is introduced by this decision.

## 5. Manifest Schema Draft

Placeholders are explicit placeholders until 79I computes real digests.

```json
{
  "schema_version": 1,
  "fixture_id": "optional_real_runtime_synthetic_v1",
  "backend_candidate": "synthetic_backend_neutral_v1",
  "artifact": {
    "artifact_id": "synthetic_artifact_v1",
    "artifact_kind": "synthetic-local-runtime-fixture",
    "path": "artifact.fixture.bin",
    "sha256": "<PLACEHOLDER_64_HEX_REPLACED_BY_79I>",
    "size_bytes": 0,
    "source_note": "<PLACEHOLDER_SOURCE_NOTE_REPLACED_BY_79I>",
    "license_note": "<PLACEHOLDER_LICENSE_NOTE_REPLACED_BY_79I>",
    "local_only": true,
    "network_required": false,
    "external_service_required": false
  },
  "input": {
    "path": "input.fixture.json",
    "sha256": "<PLACEHOLDER_64_HEX_REPLACED_BY_79I>"
  },
  "expected_output": {
    "sha256": "<PLACEHOLDER_64_HEX_REPLACED_BY_79I>",
    "encoding": "deterministic-bytes-v1"
  },
  "bounds": {
    "max_runtime_ms": 0,
    "max_memory_bytes": 0
  },
  "claims": {
    "production_claim": false,
    "gateway_visible": false,
    "policy_mutating": false
  }
}
```

### Field validation table

| Field | Required? | Meaning | Validation rule |
|---|---:|---|---|
| `schema_version` | yes | Manifest schema revision | integer == 1 |
| `fixture_id` | yes | Stable fixture identity | non-empty lowercase slug; suffix `_vN` |
| `backend_candidate` | yes | Candidate label only | must include `synthetic` and must not imply prod/runtime ready |
| `artifact.artifact_id` | yes | Stable artifact identity | non-empty |
| `artifact.artifact_kind` | yes | Artifact type label | exactly `synthetic-local-runtime-fixture` for v1 |
| `artifact.path` | yes | Relative artifact path | exactly `artifact.fixture.bin` |
| `artifact.sha256` | yes | Artifact digest | 64 lowercase hex chars |
| `artifact.size_bytes` | yes | Artifact byte size | integer, `0 < size_bytes <= 262144`, must equal file byte count |
| `artifact.source_note` | yes | Provenance note | non-empty; must state synthetic+local origin |
| `artifact.license_note` | yes | License note | non-empty |
| `artifact.local_only` | yes | Local-only guarantee | true |
| `artifact.network_required` | yes | Network dependency flag | false |
| `artifact.external_service_required` | yes | External service dependency flag | false |
| `input.path` | yes | Input file path | exactly `input.fixture.json` |
| `input.sha256` | yes | Input digest | 64 lowercase hex chars |
| `expected_output.sha256` | yes | Expected output digest | 64 lowercase hex chars |
| `expected_output.encoding` | yes | Digest byte contract | exactly `deterministic-bytes-v1` |
| `bounds.max_runtime_ms` | yes | Runtime upper bound | integer >= 0 |
| `bounds.max_memory_bytes` | yes | Memory upper bound | integer >= 0 |
| `claims.production_claim` | yes | Production claim flag | false |
| `claims.gateway_visible` | yes | Gateway exposure flag | false |
| `claims.policy_mutating` | yes | Policy mutation claim | false |

### Hash semantics (deterministic)
- `artifact_digest` (`artifact.sha256`): SHA-256 over **raw bytes** of `artifact.fixture.bin` exactly as stored (no transcoding, no newline normalization).
- `input_digest` (`input.sha256`): SHA-256 over raw bytes of `input.fixture.json` exactly as committed.
- `expected_output_digest` (`expected_output.sha256`): SHA-256 over the deterministic output byte payload produced by the fixture contract (defined in 79I test as canonical bytes domain-separated for the synthetic fixture path).
- Hex format: lowercase, 64 chars, no `0x` prefix.

## 6. Prompt 79I Implementation Plan

| 79I task | Required? | Acceptance criteria | Guardrail |
|---|---:|---|---|
| Create fixture directory | yes | `runtime/ucf-compute/tests/fixtures/optional_real_runtime/` exists | No network/external assets |
| Add tiny synthetic artifact file | yes | `artifact.fixture.bin` committed, `size <= 256 KiB` | No real model/runtime weights; no toy promotion claim |
| Add deterministic input file | yes | `input.fixture.json` committed with stable formatting | Input is local fixture only |
| Add manifest file | yes | `fixture_manifest.json` contains all required fields and strict values | Keep `production_claim=false`, `network_required=false` |
| Compute SHA-256 values | yes | Manifest hashes match actual file bytes | SHA-256 only |
| Add manifest validation test | yes | Test fails on missing field/hash mismatch/size mismatch | No runtime activation side-effects |
| Add offline/no-network assertions | yes | Test asserts `local_only=true`, `network_required=false`, `external_service_required=false` | No external service calls |
| Add no-prod-claim assertions | yes | Test asserts `production_claim=false` and non-runtime-readiness language | No readiness overclaim |
| Add CI opt-in lane | yes | Optional/non-default job or matrix include for this fixture test only | Default CI semantics unchanged |
| Keep runtime inactive | yes | No backend activation path, no inference run, no scheduler/queue/gateway additions | Must remain metadata/fixture-only |

If ambiguity remains on runtime invocation semantics: implement 79I as **fixture files + manifest validation only**, and defer runtime-invocation golden to 79J.

## 7. Current Prod Status
- Prod ready: **no**.
- OptionalRealRuntime active runtime lane: **absent**.
- This decision document is **not runtime activation**.

## 8. Open Questions
- Are placeholder hashes acceptable before fixture files exist? (recommended: yes in docs-only 79H2; replace in 79I)
- Should 79I implement fixture files only, or include runtime-invocation golden wiring?
- Where should strict schema validation live (new test-only helper vs existing runtime-contract module)?
- Should a JSON schema snapshot be added later for `fixture_manifest.json`?
- How should 79I prove no Toy promotion while still enabling OptionalRealRuntime candidate metadata?

## 9. Recommended Next Prompt
- Preferred: **UCF Prompt 79I-A — OptionalRealRuntime Fixture Files and Manifest Validation**.
- Then: **UCF Prompt 79I — Deterministic OptionalRealRuntime Golden Test Implementation** (only after fixture manifest hashing contract is merged).

## Prompt 79H2-S — Remaining Validation Completion

The artifact format decision validation was completed after the previous long-running workspace-test phase. `cargo test --workspace` and workspace clippy completed successfully. This does not change runtime status: OptionalRealRuntime remains absent and roadmap-only, no fixture files are implemented yet, no backend is promoted, and no prod-readiness claim is made.

Recommended next prompt: UCF Prompt 79I-A — OptionalRealRuntime Fixture Files and Manifest Validation.

## Prompt 79I-A — Fixture Files + Manifest Validation (Implemented)

Status: implemented as static local fixture files plus deterministic manifest validation tests.

Implemented paths:
- Fixture directory: `runtime/ucf-compute/tests/fixtures/optional_real_runtime/`
- Manifest: `runtime/ucf-compute/tests/fixtures/optional_real_runtime/fixture_manifest.json`
- Validation tests: `runtime/ucf-compute/tests/optional_real_runtime_fixture_manifest.rs`

Boundaries preserved:
- No runtime inference execution.
- No OptionalRealRuntime activation.
- No production-readiness claim.
- No backend promotion from existing Stub/Toy/Candle/Burn compile-only mappings.

## Prompt 79I-B-lite — Planned Golden Contract (Implemented)

- Added explicit planned-golden contract coverage for `expected_output.fixture.bytes` as deterministic bytes digest metadata.
- The planned golden digest is validated as deterministic and present, but this remains **not runtime inference proof**.
- This change is **not OptionalRealRuntime activation**, does not promote any backend, and does not claim prod readiness.
- Recommended next prompt: **UCF Prompt 79J-lite — OptionalRealRuntime ComputeOutputLink / Audit Metadata Integration** (alternative: **UCF Prompt 79I-C-lite — Runtime Invocation Boundary Planning**).

## Prompt 79J-lite — ComputeOutputLink / Audit Metadata Integration (Implemented)

- Added metadata-only linkage coverage proving `ComputeOutputLink` + `ComputeAuditRecord` can reference OptionalRealRuntime fixture/planned-golden digests without runtime invocation.
- Integration records fixture-manifest digest context plus artifact/input/planned-expected-output digest references as static metadata in link source semantics.
- Audit recording remains metadata-only with no runtime inference claim, no production claim, and no evidence/archive authority.
- No backend is executed, no backend is promoted to OptionalRealRuntime, and this does not claim prod readiness.
- Recommended next prompt: **UCF Prompt 79I-C-lite — Runtime Invocation Boundary Planning** (alternative: **UCF Prompt 79K-lite — Prod Compute Runtime Gate Wiring Plan**).
