# UCF Compute Feature CI Matrix

## 0. Purpose

- Defines the CI/check matrix for current compute feature lanes: default no-real-compute, backend identity, stub fixture, toy golden, optional-real compile-only, remote/external compile-only, compute link/audit, and docs/gates.
- This document is a planning and validation matrix only. It does not activate real compute, remote services, production compute, runtime model loading, runtime inference, policy override authority, evidence/archive authority, or output-record schema authority.
- Compile checks for optional real features prove compilation only; they do not prove runtime inference, artifact availability, service availability, or production readiness.
- Stub and toy lanes remain deterministic offline fixtures and are never real-compute or production claims.
- Minimal Spine v1.x remains independent of compute lanes and has no dependency on optional real compute.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `ae954286fee1a38af4d9f23546243f83666aa510` |
| HEAD short | `ae954286` |
| Dirty state | clean |
| Workspace package count | 192 |
| Backend identity test present | yes |
| Stub fixture test present | yes |
| Toy golden test present | yes |
| Optional-real compile test present | yes |
| Compute output link test present | yes |
| Compute audit test present | yes |
| Real compute inventory present | yes |
| Naming plan present | yes |

Baseline commands used: `pwd`, `git branch --show-current`, `git status --short`, `git rev-parse HEAD`, `git rev-parse --short HEAD`, `git log --oneline -15`, `cargo metadata --no-deps --format-version 1 | jq '.packages | length'`, and file-presence checks for the compute package tests and roadmap documents.

Required companion documents:

- [`docs/roadmap/real_compute_lane_inventory.md`](real_compute_lane_inventory.md)
- [`docs/roadmap/compute_backend_naming_boundary_plan.md`](compute_backend_naming_boundary_plan.md)
- [`docs/roadmap/post_freeze_roadmap_selection.md`](post_freeze_roadmap_selection.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)

## 2. Current CI Coverage Inventory

| CI/Script location | Current compute coverage | Missing lane | Risk |
|---|---|---|---|
| `.github/workflows/ci.yml` core job | Runs `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, workspace tests in test/proptest steps, docs lint, readiness gate, and model/golden checks. Workspace tests include default `ucf-compute` tests, so backend identity, stub fixture, toy golden, output link, and audit tests are covered when workspace tests execute. | No explicit per-test compute lane names in the core job; optional-real compile-only checks are not expressed as the exact compile-only command list. | Low for default no-real coverage; medium for traceability because optional-real feature intent is implicit or non-blocking elsewhere. |
| `.github/workflows/ci.yml` feature-matrix job | Has a blocking `default` lane running `cargo test --workspace --all-targets`; has non-blocking `candle-cpu` and `burn-cpu` lanes running workspace tests with compute Candle/Burn feature groups. | No remote/external lane; no explicit `ucf-ai-port` or `ucf-ai-backends` feature checks; no `lfm-lnn`; no per-feature compile-only split. | Medium: non-blocking workspace feature tests are broader and slower than compile-only probes and could be mistaken for runtime inference if not documented. |
| `.github/workflows/nightly_verify.yml` linux/windows nightly | Runs `cargo test --workspace --all-targets`, docs lint, goldens, readiness gate, adversarial suite on linux, and summary artifacts. | No optional-real compile-only feature matrix; no remote/external compile check. | Low for default/stub/toy regression; medium for optional-real drift visibility. |
| Roadmap docs and compute tests | `runtime/ucf-compute/tests/backend_identity_contract.rs`, `stub_compute_fixture.rs`, `toy_compute_golden.rs`, `optional_real_compile_gate.rs`, `compute_output_link.rs`, and `compute_audit_records.rs` define current compute-lane tests. | No artifact-backed optional-real runtime fixture; remote is compile-only/optional. | Low if claims remain bounded; high only if compile gates are described as runtime proof. |
| `runtime/ucf-ops` docs/readiness gates | Docs lint and readiness gate are already canonical validation commands and do not require compute activation. | No dedicated compute matrix gate. | Low; adding a new heavy gate is deferred to avoid changing authority or runtime behavior. |

Current answers:

- Compute tests are covered by workspace tests for default features because `ucf-compute` is a workspace package and its package tests are included by `cargo test --workspace --all-targets`.
- Existing CI has broad non-blocking feature-matrix lanes for Candle and Burn feature groups, but it does not encode the exact optional-real compile-only command list below.
- Optional-real compile checks are represented by repository tests and non-blocking feature-matrix behavior, not by a dedicated split compile-only CI step for every feature.
- Remote/external checks are not in default CI and should remain optional/non-default.
- The broad workspace feature-matrix lanes can be riskier than package-local compile probes because they increase runtime and may look stronger than compile-only evidence.
- No duplicate new workflow checks are introduced by this document.
- This document is the docs-only compute matrix for Prompt 22.

## 3. Target Compute CI Matrix

| Lane | Purpose | Required command(s) | CI treatment | Runtime claim? | Production claim? |
|---|---|---|---|---:|---:|
| default-no-real | Verify default offline compute behavior without optional real backends. | `cargo test -p ucf-compute --all-targets`<br>`cargo test --workspace` | Blocking local validation; already covered indirectly by blocking workspace CI. | no | no |
| backend-identity-contract | Verify backend class/identity claims remain explicit and bounded. | `cargo test -p ucf-compute --test backend_identity_contract` | Blocking local validation; safe candidate for future explicit CI step. | no | no |
| stub-fixture | Verify deterministic no-real stub fixture behavior. | `cargo test -p ucf-compute --test stub_compute_fixture` | Blocking local validation; safe candidate for future explicit CI step. | no | no |
| toy-golden | Verify deterministic portable toy golden behavior. | `cargo test -p ucf-compute --test toy_compute_golden` | Blocking local validation; safe candidate for future explicit CI step. | no | no |
| optional-real-compile | Verify optional real feature seams compile only. | `cargo check -p ucf-compute --features backend-burn`<br>`cargo check -p ucf-compute --features backend-candle`<br>`cargo check -p ucf-compute --features compute-burn`<br>`cargo check -p ucf-compute --features compute-candle`<br>`cargo check -p ucf-compute --features llm-burn`<br>`cargo check -p ucf-compute --features llm-candle`<br>`cargo check -p ucf-compute --features lfm-burn`<br>`cargo check -p ucf-compute --features lfm-candle`<br>`cargo check -p ucf-compute --features lfm-lnn`<br>`cargo check -p ucf-ai-port --features burn`<br>`cargo check -p ucf-ai-port --features candle`<br>`cargo check -p ucf-ai-port --features ai-runtime`<br>`cargo check -p ucf-ai-backends --features ai-burn --all-targets`<br>`cargo check -p ucf-ai-backends --features ai-candle --all-targets` | Optional/non-default compile-only lane. Prefer package-local split checks before making blocking CI broader. | no | no |
| remote-external-compile | Verify remote/external feature compiles without making it default. | `cargo check -p ucf-compute --features remote-compute` | Optional, non-default, no external service, not a blocking default lane. | no | no |
| compute-link-audit | Verify compute output linkage and audit records remain derived/audit-only metadata. | `cargo test -p ucf-compute --test compute_output_link`<br>`cargo test -p ucf-compute --test compute_audit_records` | Blocking local validation; safe candidate for future explicit CI step. | no | no |
| prod-backend-feature-gate | Validate prod-profile gate in explicit backend-burn compile feature lane. | `cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`<br>`cargo run -p ucf-ops --features backend-burn -- readiness-gate --profile prod --out ./out/gate_report_prod_split.json --workdir ./.ucf_gate_prod --workspace-test-report ./out/workspace_test_report.json` | Explicit documented lane (local/docs); CI promotion can be added later as a dedicated strict step. | no | no |
| docs/gates | Verify documentation and readiness gates without activating compute. | `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`<br>`timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | Blocking handoff validation; generated `out/*.json` files are not committed unless explicitly required. | no | no |

Commands that do not belong in default CI:

- Any command requiring external services, remote endpoints, non-checked-in model artifacts, GPU/CUDA devices, secrets, production credentials, or network downloads.
- Remote/external runtime checks for `remote-compute`.
- Optional-real runtime inference checks until an artifact-backed local fixture and explicit policy/spec authority exist.
- Claims-oriented production readiness checks for Candle, Burn, LFM, LLM, or remote compute.

## 4. Workflow Change Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A — Docs-only CI matrix | yes | Current CI already runs blocking workspace tests, has a non-blocking broad feature matrix for Candle/Burn groups, and nightly verifies workspace tests. A docs-only matrix avoids expanding runtime, avoids accidental remote/default activation, and gives future CI work an exact command list. | Low; optional-real split compile checks remain local/documented rather than CI-enforced. |
| B — Minimal workflow additions | no | Adding the full optional-real split matrix now could duplicate existing broad feature-matrix work, increase CI time, and blur compile-only versus runtime evidence. Remote/external checks must remain non-default and do not need a workflow step. | Deferred; future additions should be package-local, compile-only, no external services, and explicitly labeled non-runtime. |

## 5. Required Local Validation Commands

Default and deterministic fixture lanes:

```bash
cargo fmt --check
cargo test -p ucf-compute --test backend_identity_contract
cargo test -p ucf-compute --test stub_compute_fixture -- --nocapture
cargo test -p ucf-compute --test toy_compute_golden -- --nocapture
cargo test -p ucf-compute --test optional_real_compile_gate -- --nocapture
cargo test -p ucf-compute --test compute_output_link -- --nocapture
cargo test -p ucf-compute --test compute_audit_records -- --nocapture
```

Optional-real compile-only lanes:

```bash
cargo check -p ucf-compute --features backend-burn
cargo check -p ucf-compute --features backend-candle
cargo check -p ucf-compute --features compute-burn
cargo check -p ucf-compute --features compute-candle
cargo check -p ucf-compute --features llm-burn
cargo check -p ucf-compute --features llm-candle
cargo check -p ucf-compute --features lfm-burn
cargo check -p ucf-compute --features lfm-candle
cargo check -p ucf-compute --features lfm-lnn
cargo check -p ucf-compute --features remote-compute
cargo check -p ucf-ai-port --features burn
cargo check -p ucf-ai-port --features candle
cargo check -p ucf-ai-port --features ai-runtime
cargo check -p ucf-ai-backends --features ai-burn --all-targets
cargo check -p ucf-ai-backends --features ai-candle --all-targets
```

Package/workspace, lint, docs, and gate validation:

```bash
cargo test -p ucf-compute --all-targets
cargo test -p ucf-ai-port --all-targets
cargo test -p ucf-ai-backends --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
git diff --check
git status --short
```

## 6. Non-Default / Excluded Lanes

- `remote-compute` is excluded from default CI and default runtime behavior. It may be checked only as an optional compile-only lane without an external service.
- Optional-real runtime inference is deferred until a checked-in or otherwise governed artifact-backed fixture exists and the relevant policy/spec docs explicitly authorize it.
- Production real compute is deferred. No current lane in this matrix proves production readiness.
- Artifact/model runtime fixtures are deferred unless the repository adds an explicit local artifact, deterministic verification path, and bounded claims language.
- GPU, CUDA, external endpoint, secret-backed, model-download, and service-backed checks are not part of default CI.

## 7. Claims Boundary

- A compile pass is not runtime inference proof.
- Stub and toy lanes are deterministic no-real fixtures, not real backends.
- Optional-real-compile is not optional-real-runtime.
- Remote/external compile support is not remote service availability or runtime acceptance.
- No lane in this matrix makes a production compute claim.
- Compute output links are derived metadata only and do not change OutputRecord authority.
- Compute audit records are audit-only metadata and do not change Evidence, Archive, or policy authority.
- Minimal Spine v1.x has no dependency on compute lanes, optional-real features, remote compute, or production compute.

## 8. Future CI Hardening Steps

- Add a package-local compute feature matrix workflow only if CI maintainers want explicit split visibility for each optional-real compile-only command.
- Keep remote/external checks optional, non-default, compile-only, and service-free unless a later spec explicitly authorizes a governed runtime fixture.
- Add caching strategy only if the split feature checks become too slow.
- Add prod-profile readiness only after policy/spec docs define what prod-profile compute readiness means without overclaiming.
- Add artifact-backed optional-real runtime fixtures only when a governed local model artifact exists and tests can prove deterministic bounded behavior without network access.
- Add overclaim guardrails for docs so compile-only, toy, stub, and remote/external lanes are not described as production or runtime inference evidence.
