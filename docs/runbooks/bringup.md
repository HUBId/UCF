# Bringup

## Prerequisites
- Rust toolchain with workspace dependencies.
- Offline mode supported by default (stub/toy compute + inproc isolation).

## One-command release spine
```bash
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out
```

## Legacy smoke command
```bash
cargo run -p ucf-ops -- bringup --demo --ticks 100
```

## Runtime profile
`UCF_PROFILE=dev|test|prod`.

Resolved config lives in `./.ucf/config_resolved.json` and merges profile defaults with env overrides.

## Expected output artifacts
- `.ucf/ess/ess_fixture.json`
- `.ucf/ess/run_metadata_record.json`
- `out/metrics_summary.json`
- `out/explain_tick_last.json`
- `out/replay_verify.json` (default)
- `out/run_metadata_record.json`
