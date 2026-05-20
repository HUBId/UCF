# Readiness Gate v0 (Real Compute Ready)

`ucf-ops readiness-gate` executes an offline, deterministic readiness checklist and writes a bounded JSON report. It is a gate artifact, not by itself a production-readiness claim.

## PASS semantics

`status=PASS` means all hard checks in the report passed for the current repository state and fixture set:

- workspace tests succeeded in offline mode,
- deterministic bringup/replay checks matched,
- observability endpoints (`explain-tick`, `metrics summary`) produced non-empty data,
- backend feature-gate checks enforce fail-fast behavior for disabled packs,
- report data itself is bounded and CI-comparable via digest prefixes.

`FAIL` always includes a bounded `failure_reason` and `remediation_hint` per failing check.

## Local run

Embedded workspace-test mode remains the strict default:

```bash
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate
```

Split prerequisite mode is explicit and still mandatory:

```bash
cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate --workspace-test-report ./out/workspace_test_report.json
```

The split report is accepted only when it is `PASS`, records `cargo test --workspace --offline`, and matches the current HEAD and dirty state. Missing, stale, mismatched, wrong-command, or non-PASS evidence fails `build_workspace_tests`; it is not a silent skip and is never treated as `PASS`. The commands exit with code `0` on `PASS` and `2` on `FAIL`.




## Prod backend feature-lane invocation (compile-only gate context)

`--profile prod` currently selects a backend pack path that requires the compile-time feature `backend-burn`. This is a **feature-lane requirement**, not a runtime-inference readiness claim.

Use explicit split evidence plus a feature-enabled `ucf-ops` invocation:

```bash
cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json
cargo run -p ucf-ops --features backend-burn -- readiness-gate --profile prod --out ./out/gate_report_prod_split.json --workdir ./.ucf_gate_prod --workspace-test-report ./out/workspace_test_report.json
```

Boundary notes:
- Passing this lane proves only that prod-profile gate checks pass in the `backend-burn` compile feature context.
- It does **not** enable or prove OptionalRealRuntime, real runtime scheduling, queue/worker services, or production runtime inference.
- A fail (for example another prod check blocker) remains a fail and must be reported as such.

## Prod-profile overclaim guard

- A `PASS` from `--profile test` is **not** a `--profile prod` pass.
- `SKIP` is not `PASS`.
- `TIMEOUT` is not `PASS`.
- Missing `workspace_test_report` evidence is not `PASS`.
- Stale reports (HEAD/dirty mismatch) are not current truth.
- `cargo test --workspace` is useful validation, but not a substitute for fresh `workspace-test-check` split evidence when split mode is required.

## Workspace-test runtime diagnostics

`workspace-test-check` intentionally runs the same strict workspace command required by the gate:

```bash
cargo test --workspace --offline
```

The command is broad: it compiles and executes the full workspace test graph, including doc-tests. On a cold or partially invalidated target directory it can spend several minutes compiling before test output advances. This is normal Cargo behavior and is not, by itself, a deadlock. Operators should wrap local/CI evidence generation with an explicit outer timeout that is large enough for the 192-package workspace; a 300 second guard can be too small on cold or mixed-profile builds, while a 600 second guard is the current practical minimum observed for this environment.

`workspace-test-check` emits stderr progress for:

- preflight metadata,
- `cargo test --workspace --offline`,
- report assembly,
- report write.

The generated report also records `phase_timings[]` for the metadata, Cargo command, and report assembly phases, plus additive diagnostic fields (`last_phase`, `cargo_command_started_at_utc`, `cargo_command_duration_ms`, `diagnostic_notes`). If an external timeout kills the process, the last visible `[workspace-test-check] start:` line identifies the active phase. A timeout remains a failed/missing evidence condition and must not be treated as `PASS`.

## Report schema

`ReadinessGateReport` (JSON):

- `code_version_tag`
- `fixtures_digest_prefix`
- `backend_pack_digest_prefix`
- `timestamp` (optional)
- `status` (`PASS` / `FAIL`)
- `checks[]`
- `phase_timings[]`

Additional v1.1 section checks (also `CheckResult` objects):

- `weights_lifecycle`
- `world_vljepa_evidence`
- `sae_real`
- `ssm_opt`
- `gpu_lane` (optional section check)

Each check is a bounded `CheckResult`:

- `name`
- `status` (`PASS` / `FAIL` / `SKIP`)
- `evidence` (bounded key/value map)
- `failure_reason` (bounded string)
- `remediation_hint` (bounded string)

## Interpreting failures

1. Open `checks[]` entries with `status=FAIL`.
2. Use `failure_reason` for the direct invariant that failed.
3. Apply `remediation_hint` and rerun the command.
4. Compare digest prefixes in `evidence` with prior CI runs for deterministic drift diagnosis.

## Adding new checks

When new roadmap stages are added:

1. Add a dedicated check helper in `runtime/ucf-ops/src/lib.rs`.
2. Keep evidence bounded (prefixes/counters, no large blobs).
3. Append the check in `readiness_gate(...)`.
4. Add/extend tests in `runtime/ucf-ops/src/lib.rs` and `runtime/ucf-ops/tests/ops_flow.rs`.
5. Update this document with new check intent and remediation guidance.
