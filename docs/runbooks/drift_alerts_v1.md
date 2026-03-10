# Drift + Alerts Runbook v1 (Offline)

This runbook covers deterministic operator handling for drift and alert signals in v1.

> Scope note: v1 provides bounded drift/alert scaffolding and operator recommendations. Auto-remediation is intentionally limited.

## 0) First-stop consolidated operator report

```bash
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
```

Use this report first for a bounded cross-section summary before diving into drift/alerts details.

## 1) Generate drift report

```bash
cargo run -p ucf-ops -- drift report --run <run_id> --windows 20 --out ./out/drift_report.json
```

Review:

- per-slot status (`OK|WARN|SEVERE`)
- breached fields and last alarms
- recommended actions (`disable_shadow|recommend_rollback|none`)

## 2) Generate alerts report

```bash
cargo run -p ucf-ops -- alerts report --run <run_id> --out ./out/alerts_report.json
```

Review:

- active alerts
- trigger history
- deterministic remediation command suggestions

## 3) Understand shadow auto-disable behavior

When drift is severe and the budget `action_on_severe` is `disable_shadow`, runtime can auto-disable shadow for that slot.

Operator implications:

- decision path remains primary; no shadow promotion occurs automatically
- auto-disable is tightening-only safety behavior
- rollback remains operator-invoked

## 4) Recommended remediation sequence

Run in order:

```bash
cargo run -p ucf-ops -- drift report --run <run_id> --windows 20 --out ./out/drift_report.json
cargo run -p ucf-ops -- alerts report --run <run_id> --out ./out/alerts_report.json
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
```

If rollback is recommended and approved by operator policy:

```bash
cargo run -p ucf-ops -- models rollback --slot <slot> --to <known_good_sha256> --out ./out/models_rollback_<slot>.json
```

## Artifacts you should see

- `./out/drift_report.json`
- `./out/alerts_report.json`
- `./out/strict_check.json`
- `./out/gate_report.json`
- `./out/models_rollback_<slot>.json` (only if rollback executed)

## Troubleshooting

- Drift report empty/unexpected:
  - validate `--run` identifier and ensure compare-window artifacts exist
- Alerts report empty while drift severe exists:
  - confirm alert rules in policy pack include drift-severe rule kind
- Repeated severe after rollback:
  - keep shadow disabled for impacted slot
  - re-stage candidate and rerun probe + readiness before any new promotion
