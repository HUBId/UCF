# Shadow Runbook v1 (Observational-Only, Offline)

This runbook describes how to enable and validate slot shadow mode in v1.

> Scope note: v1 shadow is scaffolding for comparison telemetry. Shadow outputs are observational-only and must not alter decision selection.

## Preconditions

- Work from repository root.
- Keep operation offline.
- Ensure compare-window wiring is configured (`UCF_SLOT_COMPARE_WINDOW > 0`).

## 1) Configure slot for shadow

Example environment for a local run:

```bash
export UCF_REAL_ENABLEMENT_MODE=shadow
export UCF_SLOT_COMPARE_WINDOW=128
```

## 2) Execute deterministic shadow smoke check

```bash
cargo run -p ucf-ops -- v1 smoke --shadow --out ./out/v1_shadow_smoke.json
```

## 3) Verify decision neutrality

- Confirm smoke result is pass.
- Confirm no command/report indicates shadow replaced primary decision output.

## 4) Inspect compare-window evidence

Generate drift view from compare windows:

```bash
cargo run -p ucf-ops -- drift report --run <run_id> --windows 20 --out ./out/drift_report.json
```

Interpretation checklist for compare windows:

- `OK`: within configured budget for sampled windows
- `WARN`: budget edge trend, continue shadow and monitor
- `SEVERE`: threshold breach; follow remediation flow below

## Artifacts you should see

- `./out/v1_shadow_smoke.json`
- `./out/drift_report.json`
- ESS drift alarm records and compare-window records in run artifacts

## Troubleshooting

- Shadow does not start:
  - ensure mode is `shadow` and compare window is greater than zero
  - check strict mode output for missing drift budget wiring
- Shadow appears to impact decisions:
  - stop rollout and return slot to non-shadow mode
  - rerun smoke check and strict checks before re-enabling
- Frequent severe windows:
  - run remediation commands from `drift report`
  - optionally disable shadow for the slot and investigate candidate bytes/config
