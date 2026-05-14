# Compute OutputRecord Linkage Boundary

## Purpose

This note documents the Prompt 20 boundary for optional compute-to-`OutputRecord` linkage. The linkage is derived metadata only. It does not make compute part of the Minimal Spine, does not give compute authority over protocol records, and does not enable any real compute runtime.

## Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `9ab255ef2ddaa65065b8e55944ea0d09c5aeaf4e` |
| HEAD short | `9ab255ef` |
| Dirty state | clean |
| Workspace package count | 192 |
| Backend identity contract present | yes |
| Stub fixture test present | yes |
| Toy golden test present | yes |
| Optional-real compile gate present | yes |
| Minimal Spine E2E present | yes |
| Freeze doc present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -15`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and required path presence checks.

## Existing inventory

| Concern | Existing API/type | Path | Current behavior | Gap |
|---|---|---|---|---|
| Canonical output record | `ucf_protocol::v1::spec::OutputRecord` | `protocol/crates/ucf-protocol/src/lib.rs` | Protocol-owned record with canonical field encoding and canonical bytes through `canonical_bytes`. | Compute had no optional metadata reference to protocol output digests. |
| OutputRecord canonical bytes | `CanonicalEncode` for `OutputRecord` plus `canonical_bytes` | `protocol/crates/ucf-protocol/src/lib.rs` | Canonical bytes exist; digest algorithm/domain remains caller-owned. | No compute-side helper should become protocol digest authority. |
| Stub result summary | `StubComputeFixtureOutput`, `stub_compute_fixture_digest` | `runtime/ucf-compute/src/lib.rs` | Deterministic stub output with portable digest, no real inference, no production claim. | No derived link to an OutputRecord digest. |
| Toy golden summary | `ToyComputeGoldenOutput`, `toy_compute_golden_digest` | `runtime/ucf-compute/src/lib.rs` | Deterministic toy output with pinned digest, no real inference, no production claim. | No derived link to an OutputRecord digest. |
| Backend classification | `BackendIdentity`, `BackendClass` | `runtime/ucf-compute/src/contracts.rs` | Machine-readable stub/toy/optional-real-compile/etc. classification. | Link needed to carry the same classification without reclassifying lanes. |
| Minimal Spine E2E | protocol `OutputRecord` fixture | `core/crates/ucf-router/tests/minimal_spine_e2e.rs` | Builds and archives protocol records without compute imports. | Must remain independent of compute. |
| Existing audit records | evidence/archive paths | `runtime/ucf-compute/src/evidence.rs`, archive crates | Evidence/archive remain separate authorities. | Link must not replace evidence/archive authority. |
| Compute/protocol dependency | none in normal `ucf-compute` dependencies | `runtime/ucf-compute/Cargo.toml` | Runtime compute does not depend on `ucf-protocol`. | Tests may use protocol fixtures; normal compute code should not. |

Answers:

- `OutputRecord` is canonically defined in `ucf-protocol` under `v1::spec::OutputRecord`.
- `OutputRecord` has canonical bytes via `CanonicalEncode`/`canonical_bytes`; there is no compute-owned canonical digest authority.
- Compute summaries exist as `ComputeSignalsSummary`, `StubComputeFixtureOutput`, and `ToyComputeGoldenOutput` with portable fixture/golden digests.
- There was no compute-to-output linkage before this prompt.
- A regular compute-to-protocol dependency would be risky because it could imply Compute is part of the Minimal Spine or owns protocol schema authority.
- Existing evidence/archive records do not fit because this prompt only needs optional metadata references, not a new authority path.
- The smallest safe location is `runtime/ucf-compute/src/output_link.rs`, exported by `ucf-compute`, because the link belongs to compute-derived metadata and stores only digests/IDs/classification.

## Authority decisions

| Concern | Decision | Reason |
|---|---|---|
| OutputRecord authority | `ucf-protocol` | Protocol owns schema and canonical encoding. |
| Compute linkage authority | derived metadata only | Compute stores references to output and compute digests; it cannot mutate, replace, or produce authority for `OutputRecord`. |
| Minimal Spine dependency | no | Minimal Spine tests and runtime paths remain free of compute imports. |
| Evidence/Archive authority | unchanged | The link is not evidence, archive truth, or proof authority. |
| Gateway exposure | no | No gateway/API exposure is added in this prompt. |
| Runtime inference claim | no | Stub, toy, and optional-real-compile links preserve no-real-runtime and no-production flags. |

## Implemented link shape

`ComputeOutputLink` carries:

- `output_record_digest` plus optional `output_record_id` and `output_record_bytes_digest` references.
- `compute_result_digest` for the stub fixture, toy golden, or compile probe summary/reference digest.
- Backend identity fields: `backend_class`, `backend_name`, determinism/offline/external-service flags.
- Safety flags: `no_real_runtime`, `runtime_inference_supported`, `production_claim`.
- Boundary flags: `metadata_only = true`, `output_record_authority = false`, `minimal_spine_required = false`.

The link digest is deterministic over link metadata only. It does not encode or mutate a protocol `OutputRecord`, and it is optional for compute callers.
