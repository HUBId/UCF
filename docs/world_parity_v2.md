# World Parity v2

## Purpose

This stage introduces deterministic shadow parity for exactly one slot: the World predictor.
It compares the same `WorldPredictorV1` contract across:

- `stub_world_v1` (primary)
- `candle_world_v1` (shadow)
- `burn_world_v1` (shadow)

## Parity record

`WorldParityRecordV1` captures bounded, deterministic compare evidence:

- run/window identifiers (`run_id`, `window_id`, `t0`, `t1`)
- primary backend id
- compared backends (bounded to 2, sorted by backend id)
- scalar deltas (`prediction_error_delta_q_*`, `surprise_delta_q_*`)
- digest mismatch and invalid output counters
- digest prefix samples (bounded to 4)
- per-backend status (`ok|warn|severe`)
- parity digest and policy graph digest prefix

## Eligibility summary

`WorldBackendEligibilityV1` is deterministic and stage-aware:

- `eligible_for_shadow = probe_pass && !severe_drift_present`
- `eligible_for_active`
  - `stub_world_v1`: allowed by existing stub path
  - `candle_world_v1` / `burn_world_v1`: always false in v2 stage with reason code
    `ACTIVE_NOT_ENABLED_IN_V2_STAGE`

## Ops command

Generate report:

```bash
cargo run -p ucf-ops -- world parity-report --run <id> --out ./out/world_parity_report.json
```

The report contains:

- latest parity windows (<=10)
- eligibility summary for stub/candle/burn
- remediation hints (probe, drift inspection, keep shadow-only)

## Strict mode hook

When compare is explicitly configured and both candle+burn shadows are enabled, strict mode
requires parity evidence. Missing report fails with stable code:

- `PARITY_EVIDENCE_MISSING`

This does **not** auto-enable active mode for real backends.


See shared semantics: `docs/compare_window_semantics_v3.md`.
