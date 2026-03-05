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
  - `kind` (`DriftAlarmRate`, `GatewayAuthFailRate`, `StrictModeFailure`, `DegradedFallbackRate`, `EmergencyActiveRate`)
  - `window_size`
  - `threshold`
  - `severity` (`low|medium|high|critical`)
  - `action` (`recommend|tighten|disable_slot|require_operator`)

## Deterministic windows and counts

Alerts are evaluated deterministically with bounded windows:

- Tick-window based for ESS-derived signals (`DegradedFallbackRate`, `EmergencyActiveRate`).
- Bounded latest-event windows for gateway and drift JSONL artifacts.
- Strict-mode failure uses local strict failure artifact presence.

No network access is required.

## Records

Alert lifecycle events are persisted to:

- `.ucf/out/alerts_records.jsonl`

Event types:

- `trigger` with `AlertRecordV1`
- `clear` with `AlertClearRecordV1`

Records are digest-only evidence (no sensitive payload text), bounded evidence vectors, and deterministic remediation code mapping.

## Report command

Run:

```bash
cargo run -p ucf-ops -- alerts report --run <id> --out ./out/alerts_report.json
```

The report includes:

- active alerts (bounded)
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
4. Consider rollback recommendation:
   - `ucf-ops models recommend-rollback --slot world`
