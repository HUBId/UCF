# Diagnostics

## Command
```bash
cargo run -p ucf-ops -- diag
cargo run -p ucf-ops -- diag --json
```

## Checks
- `workspace_build_tag`: git/build metadata available.
- `config_resolved`: deterministic/safe config loaded.
- `ess_health`: fixture ESS can be opened and parsed.
- `audit_chain`: audit checkpoints readable if present.
- `compute_probe`: one deterministic compute call succeeds.
- `sandbox_runtime`: isolation mode validated.
- `metrics_tracing`: log/telemetry artifacts available.

## Failure remedies
- Run `bringup --demo` to create fresh local state.
- Set `capabilities_default=deny` in resolved config.
- Use `compute_backend=stub` for offline and deterministic diagnostics.
