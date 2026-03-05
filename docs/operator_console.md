# Operator Console v1 (`ucf-console`)

`ucf-console` is a local-only, read-only terminal dashboard for operators.

## What it shows

- **Overview (tab 1)**: health status, strict mode, digest prefixes, drift status, tick age.
- **Alerts (tab 2)**: bounded active alerts and recent trigger context.
- **Drift (tab 3)**: stage status (`OK` / `DEGRADED`) and alarm summary.
- **Runs (tab 4)**: last 20 runs from local run registry.

All views are bounded and deterministically ordered.

## Run

```bash
cargo run -p ucf-console
```

Optional local gateway settings:

```bash
export UCF_GATEWAY_TOKEN="<token>"
cargo run -p ucf-console -- --endpoint tcp://127.0.0.1:44991
```

If gateway health is unavailable, the console falls back to local artifacts (`out/` + `ess/runs`).

## Keybindings

- `1`..`4`: switch tabs
- `r`: refresh snapshot
- `e`: export current tab JSON to `./out/console_export.json`
- `q`: quit

No execute/promote/enable controls are implemented.

## CI / one-shot mode

Use one-shot mode for smoke checks:

```bash
cargo run -p ucf-console -- --once --out ./out/console_once.json
```

This writes a single Overview snapshot and exits.

## Bug reports

Attach these files when reporting issues:

- `./out/console_export.json`
- `./out/console_once.json`
- optionally `./out/alerts_report.json` and `./out/drift_report.json`
