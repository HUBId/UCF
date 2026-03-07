# Operational Alerts v1

`Operational Alerts` provide deterministic, bounded operator-facing alerts derived from policy rules and local artifacts.

## Rule configuration

Rules are configured in policy pack `alerts.toml` files:

- `policies/packs/base_v1/alerts.toml`
- `policies/packs/overlays/{test,dev,prod}/alerts.toml`

Schema:

- `schema_version`
- `[[rules]]`
  - `id`
  - `kind` (`DriftSevereRate`, `GatewayAuthFailRate`, `StrictFailurePresent`, `DegradedFallbackRate`)
  - `window_size`
  - `threshold`
  - `clear_after_windows` (deterministic hysteresis)
  - `severity` (`low|medium|high|critical`)
  - `action` (`recommend_rollback|recommend_disable_shadow|require_operator`)

## Deterministic windows and counts

Alerts are evaluated deterministically with bounded windows:

- Tick-window based for ESS-derived degraded fallback signals.
- Bounded latest-event windows for gateway abuse and drift JSONL artifacts.
- Drift alerts count only `DriftAlarmRecordV1` records where `severity == "SEVERE"`.
- Strict-mode failure uses local `./out/strict_failure.json` presence.
- Clear events require `clear_after_windows` consecutive below-threshold windows.

No network access is required.

## Records

Alert lifecycle events are persisted to:

- `.ucf/out/alerts_records.jsonl`

Event types:

- `trigger` with `AlertRecordV1`
- `clear` with `AlertClearRecordV1`

Evaluator state for hysteresis is persisted to:

- `.ucf/out/alerts_state.json`

## Report command

Run:

```bash
cargo run -p ucf-ops -- alerts report --run <id> --out ./out/alerts_report.json
```

The report includes:

- active alerts (bounded, <=16)
- last trigger history (<=20)
- deterministic remediation command suggestions
- short operator summary text

## Typical remediation flow

Depending on active alerts:

1. Recompute drift view:
   - `ucf-ops drift report --run <id> --windows 20 --out ./out/drift_report.json`
2. Re-test gateway abuse posture:
   - `ucf-ops gateway threat-test --out ./out/gateway_threat.json`
3. Re-run strict posture checks:
   - `ucf-ops strict check --strict --out ./out/strict_check.json`
4. Consider rollback recommendation (recommendation only):
   - `ucf-ops models rollback --slot world --to <hash>`
