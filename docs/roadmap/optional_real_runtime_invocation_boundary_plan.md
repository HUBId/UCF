# UCF OptionalRealRuntime Runtime Invocation Boundary Plan

## 0. Purpose
- Boundary plan only.
- No runtime invocation.
- No prod readiness claim.

## 1. Baseline
- HEAD: `908c3da12f5d7f5078f20a1e56169cc58fc3eec1`
- Fixture/docs/tests present:
  - `runtime/ucf-compute/tests/optional_real_runtime_fixture_manifest.rs`
  - `runtime/ucf-compute/tests/optional_real_runtime_planned_golden_contract.rs`
  - `runtime/ucf-compute/tests/optional_real_runtime_fixture_link_audit.rs`
  - `runtime/ucf-compute/tests/optional_real_runtime_contract.rs`
  - `runtime/ucf-compute/tests/fixtures/optional_real_runtime/fixture_manifest.json`
  - `docs/roadmap/optional_real_runtime_artifact_format_decision.md`
  - `docs/roadmap/optional_real_runtime_pinned_local_fixture_plan.md`
  - `docs/roadmap/optional_real_runtime_artifact_fixture_inventory.md`
  - `docs/roadmap/optional_real_runtime_prod_compute_semantics_roadmap.md`

## 2. Runtime Boundary Inventory

| Concern | Current behavior | Gap |
|---|---|---|
| Runtime invocation seam | Metadata contracts and tests exist (`runtime_contract`, fixture-manifest tests, planned-golden tests, link/audit tests), but no execution path is wired for OptionalRealRuntime. | A dedicated invocation-owner seam is not yet codified as contract type/trait. |
| Artifact loader | Fixture files are loaded in tests from local `tests/fixtures/optional_real_runtime/`; digest validation is deterministic and local-only. | No dedicated runtime invocation loader layer with explicit boundary guards yet. |
| Timeout/memory bounds | `max_runtime_ms` and `max_memory_bytes` exist in manifest and contract as bounded metadata fields. | Enforcement is contract-level metadata only; no invocation-time bound enforcement path exists. |
| Natural fixture invocation location | `runtime/ucf-compute` is already the home for OptionalRealRuntime contract metadata and fixture validation tests. | A single owner function/module for optional runtime invocation is not yet reserved. |
| Forbidden paths | Current docs/tests consistently require offline/local-only behavior, forbid external service/network requirement, forbid production claim, and keep audit/link metadata-only semantics. | Future invocation path must preserve all forbiddens explicitly with deterministic reject errors. |

Required answers:
- Runtime invocation seam already present: **no**, only metadata/contract seam exists.
- Artifact loader already present: **test-fixture loader yes**, runtime invocation loader **no**.
- Timeout/memory bounds today: **metadata-only**.
- Natural location for future fixture invocation: **inside `runtime/ucf-compute` OptionalRealRuntime contract lane**, not in gateway/scheduler/queue/worker.
- Must remain forbidden: **network, remote/external service, gateway, policy mutation, evidence/archive append authority, backend promotion without explicit contract, production claim**.

## 3. Boundary Options

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A — Docs-only runtime boundary plan | yes | Safest in this phase; preserves current no-runtime-activation guarantees while making the invocation scope explicit. | No machine-checkable invocation type boundary yet. |
| B — Add no-op/runtime-boundary trait without implementation | no (deferred) | Useful next step for contract hardening, but still introduces new production-facing surface in this prompt. | Confusion risk if interpreted as partial activation surface. |
| C — Add test-only planned invocation harness | no | Could prepare implementation, but this phase prioritizes zero ambiguity that no invocation exists. | May be misread as runtime inference preparation/activation. |
| D — Implement real invocation now | no (forbidden) | Explicitly disallowed by prompt constraints. | Would violate scope and overclaim risk. |

## 4. Runtime Invocation Boundary

| Boundary item | Decision | Reason |
|---|---|---|
| invocation owner | Future single owner in `runtime/ucf-compute` OptionalRealRuntime contract lane (contract-type boundary first, then implementation). | Keeps ownership local to existing compute contract domain and avoids cross-cutting authority drift. |
| allowed input | Fixture manifest + pinned artifact file + pinned input fixture only. | Matches existing manifest/contract semantics and deterministic fixture workflow. |
| allowed artifact | Local pinned artifact only (manifest-pinned digest + size + local-only flag). | Preserves offline-first and deterministic reproducibility. |
| network | forbidden | Current contract/roadmap semantics require no network dependency for optional fixture candidate. |
| external service | forbidden | Existing contract and fixture manifest fields already model this as false-only for accepted candidate. |
| gateway | forbidden | Invocation boundary must remain compute-local and non-gateway-exposed. |
| policy mutation | forbidden | Runtime invocation candidate must remain non-authoritative and non-policy-mutating. |
| evidence/archive append | forbidden | Link/audit semantics remain metadata-only; no authority expansion. |
| runtime output | Deterministic bytes only under pinned fixture contract domain. | Required for reproducible digest assertions and no-overclaim posture. |
| timeout | Required from manifest (`max_runtime_ms > 0`) and enforced as hard reject/timeout violation in future invocation path. | Bound required by current contract metadata and deterministic failure-mode planning. |
| memory bound | Required from manifest (`max_memory_bytes > 0`) and enforced in future invocation path. | Same bounded-cost invariant as runtime bound. |
| failure modes | Deterministic, typed reject/violation outcomes (no ambiguous pass/timeout-as-pass). | Prevents stale/ambiguous readiness interpretation and keeps replay deterministic. |
| production claim | forbidden | OptionalRealRuntime boundary planning does not imply production readiness. |
| BackendClass claim gate | Candidate may claim `BackendClass::OptionalRealRuntime` only when invocation boundary contract tests pass and no forbidden path is enabled. | Prevents class overclaim from metadata-only or compile-only states. |

## 5. Future Test Plan

| Future test | Purpose | Required before OptionalRealRuntime? |
|---|---|---:|
| `optional_real_runtime_invocation_rejects_network_required_fixture` | Ensure invocation rejects fixtures that require network access. | yes |
| `optional_real_runtime_invocation_rejects_external_service_fixture` | Ensure invocation rejects fixtures requiring external services. | yes |
| `optional_real_runtime_invocation_respects_timeout_bound` | Ensure runtime bound is enforced deterministically. | yes |
| `optional_real_runtime_invocation_produces_expected_output_digest` | Ensure deterministic output bytes match pinned expected digest. | yes |
| `optional_real_runtime_invocation_is_deterministic` | Ensure repeated runs over same fixture are byte-stable. | yes |
| `optional_real_runtime_invocation_does_not_append_evidence_archive` | Ensure no evidence/archive append authority is added by invocation path. | yes |
| `optional_real_runtime_invocation_does_not_promote_backend_without_contract` | Ensure no backend class promotion occurs without explicit contract guard fulfillment. | yes |
| `optional_real_runtime_invocation_does_not_claim_prod` | Ensure invocation path cannot claim production readiness. | yes |

## 6. Current Status
- Fixture files exist.
- Manifest validation exists.
- Planned golden exists.
- ComputeOutputLink/Audit metadata exists.
- Runtime invocation absent.
- OptionalRealRuntime absent (not active in current backend mappings).
- Prod ready: no.

## 7. Recommended Next Prompt
- UCF Prompt 79K-lite — OptionalRealRuntime Invocation Boundary Contract Types
- or
- UCF Prompt 79I-D-lite — Deterministic Runtime Invocation Harness Planning
