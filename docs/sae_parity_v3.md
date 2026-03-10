# SAE Parity v3

Scope: exactly the configured second slot from `docs/series_state_snapshot.md` (current repo: `sae`).

## Compared backends
- Primary (decision source): `stub_sae_v1`
- Shadow compare: `candle_sae_v1` (required when SAE parity is enabled)
- Optional shadow compare: `burn_sae_v1` (only when Burn support is already available; otherwise `SKIP`)

## Window semantics
SAE parity windows reuse the same compare-window contract family as World parity:
- bounded latest windows (`<=10`)
- deterministic ordering by `(t1, window_id)`
- deterministic backend ordering by `backend_id`
- canonical parity digest per window

`SaeParityRecordV1` includes:
- run/window bounds (`run_id`, `window_id`, `t0`, `t1`)
- primary backend id
- bounded compared backend entries (max 2)
- delta evidence (`spike_count_delta_mean_q`, `spike_count_delta_max_q`, `magnitude_delta_mean_q`)
- digest/invalid counters and bounded sample prefixes
- backend status (`OK|WARN|SEVERE`)

## Drift and eligibility relation
- SAE parity is shadow evidence only.
- WARN/SEVERE parity windows are consumable by drift/eligibility evidence paths.
- Parity never alters decision outputs directly.

## Active safety
This report does **not** enable Active by itself. Active remains separately gated by active-evidence requirements.

## Command
```bash
cargo run -p ucf-ops -- models parity --slot sae --run <id> --out ./out/sae_parity_report.json
```
