# Drift Budget v1

`DriftBudgetV1` is the single source of truth for envelope/drift thresholds.

## Location

- Base pack: `policies/packs/base_v1/drift_budget.toml`
- Overlays: `policies/packs/overlays/*/drift_budget.toml`

## Schema

```toml
schema_version = 1

[[entries]]
stage_id = "world_vljepa"
window_size = 20
latency_p95_max_ms = 5
invalid_rate_max_q = 500
timeout_rate_max_q = 500
delta_scalar_max_q = 500
digest_mismatch_rate_max_q = 0 # optional by stage
action_on_breach = "disable_shadow" # disable_shadow | force_toy | recommend_rollback
```

All rates use Q values in `[0..10000]`.

## Drift evaluation

`ucf-ops drift report` evaluates bounded latest windows per stage and emits standardized `DriftAlarmRecordV1` entries:

- `stage_id`, `window_id`
- breached fields
- quantized observed values
- action recommendation (tightening-only)
- reason code
- evidence digest list

## Dashboard report

Generate:

```bash
cargo run -p ucf-ops -- drift report --run <id> --windows 20 --out ./out/drift_report.json
```

The report contains per-stage summaries, active alarms, and operator text summary.

Status semantics:

- `OK`: no breaches in reported windows
- `DEGRADED`: one or more breaches; apply tightening/rollback recommendations only

## Explain tick

`explain-tick` includes drift statuses for active stages and alarm/reason references when current tick maps to an alarmed window.
