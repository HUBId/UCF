# Bringup

## Prerequisites
- Rust toolchain with workspace dependencies.
- Offline mode supported by default (stub compute + inproc isolation).

## Command
```bash
cargo run -p ucf-ops -- bringup --demo --ticks 100
```

## Config
`./.ucf/config_resolved.json`
- `compute_backend` (default `stub`)
- `compute_seed` (fixed deterministic default)
- `compute_budget_profile` (default `default`)
- `isolation_runtime` (default `inproc`)
- `capabilities_default` (default `deny`)

## Expected output
- PID/status line
- ESS fixture path (`.ucf/ess/ess_fixture.json`)
- log file (`.ucf/logs/bringup.log`)
- deterministic digest for the demo run
