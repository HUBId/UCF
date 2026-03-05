# Soak Suite v1 (Long-Run Stability)

`ucf-ops soak run` provides a hardware-neutral soak harness for long-duration stability checks.

## Command

```bash
cargo run -p ucf-ops -- soak run --duration 2h --scenario golden_a --out ./out/soak_run/
```

## Monitored signals

- Health polling (default every 5s): `health_status`, `last_tick_age_ms`.
- Counters:
  - `drift_alarms`
  - `fallbacks`
  - `gateway_abuse`
  - `emergency_active_ticks`
- RSS memory sampling (best effort, default every 60s):
  - Linux: `/proc/self/status` (`VmRSS`)
  - Windows/non-Linux: unsupported fallback (`null` samples)

## Incident injection (deterministic, test/dev usage)

Use repeated `--inject` flags:

```bash
cargo run -p ucf-ops -- soak run \
  --duration 2m \
  --scenario golden_a \
  --inject timeout:llm@t=200 \
  --inject drift:ssm@t=400 \
  --inject gateway_auth_fails@t=600 \
  --out ./out/soak_injected/
```

## Leak sentinels

Leak sentinel marks failure when both conditions hold:

- RSS slope exceeds threshold (`24 MB/hour`), and
- sustained increasing windows exceed threshold.

Output fields:

- `leak_sentinel.slope_mb_per_hour`
- `leak_sentinel.sustained_growth_windows`
- `leak_sentinel.leak_suspected`

## Output artifacts

- `soak_report.json` (summary)
- `soak_timeseries.json` (downsampled time series, max 256 points)
- on failure (or `--postmortem`): `postmortem_<timestamp>.zip`

The postmortem archive contains redaction-safe JSON only:

- `diagnostics_bundle.json`
- `repro_pack.json`
- `alerts_report.json`
- `drift_report.json`
- `health_snapshot.json`
- `manifest.json` (sha256 checksums)

## Interpreting status

- `pass`: no injected incidents and no leak sentinel trigger.
- `fail`: incident injection and/or leak sentinel trigger.

## Short soak (validation)

```bash
cargo run -p ucf-ops -- soak run --duration 2m --scenario golden_a --inject timeout:llm@t=20 --out ./out/soak_short/
```

Inspect bundle:

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile('./out/soak_short/postmortem_<timestamp>.zip')
print('\n'.join(z.namelist()))
PY
```
