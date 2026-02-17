# Readiness Gate v0 (Real Compute Ready)

`ucf-ops readiness-gate` executes an offline, deterministic production-readiness checklist and writes a bounded JSON report.

## PASS semantics

`status=PASS` means all hard checks in the report passed for the current repository state and fixture set:

- workspace tests succeeded in offline mode,
- deterministic bringup/replay checks matched,
- observability endpoints (`explain-tick`, `metrics summary`) produced non-empty data,
- backend feature-gate checks enforce fail-fast behavior for disabled packs,
- report data itself is bounded and CI-comparable via digest prefixes.

`FAIL` always includes a bounded `failure_reason` and `remediation_hint` per failing check.

## Local run

```bash
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate
```

The command exits with code `0` on `PASS` and `2` on `FAIL`.

## Report schema

`ReadinessGateReport` (JSON):

- `code_version_tag`
- `fixtures_digest_prefix`
- `backend_pack_digest_prefix`
- `timestamp` (optional)
- `status` (`PASS` / `FAIL`)
- `checks[]`

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
