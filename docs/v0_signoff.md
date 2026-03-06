# v0 Signoff Gate

## Definition: "v0 complete"

`v0` is considered complete when `ucf-ops v0 gate` returns PASS (`exit code 0`) against the canonical v0 scenario.

A PASS means all required v0 checks pass in fixed order:

1. `policy_graph_lock`
2. `determinism_double_run`
3. `e2e_flow_a`
4. `record_boundedness`
5. `schema_versions_known`
6. `no_tool_execution`

The gate is conservative: if any check fails or cannot be executed, overall status is FAIL (`exit code 2`).

## Run

```bash
cargo run -p ucf-ops -- v0 gate \
  --scenario fixtures/e2e/v0_flow_a.json \
  --out ./out/v0_gate_report.json
```

## Report

Output schema: `V0GateReportV1`

- `schema_version`
- `overall_status` (`PASS`/`FAIL`)
- `checks[]` in deterministic fixed order with:
  - `name`
  - `status`
  - `evidence_digest_prefixes`
  - `remediation_hint_code`

## PASS guarantees

- Offline, hardware-neutral v0 signoff execution.
- Determinism check via double-run digest comparison.
- Policy graph digest lock check against canonical v0 snapshot digest prefix.
- Record boundedness enforced by max per-record serialized byte cap.
- Known, non-zero schema versions for required v0 records.
- No `ToolExecution` records in the v0 flow.

## Explicitly not included in v0

- Real model weights (slots remain stub/hash-locked scaffolding).
- Real external tool execution paths.
