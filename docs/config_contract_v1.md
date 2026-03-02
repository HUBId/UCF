# Config Contract v1

`ConfigV1` defines the runtime configuration schema for `dev`, `test`, and `prod` profiles.

## Guarantees

- strict schema parsing via `serde(deny_unknown_fields)`
- deterministic validation
- numeric ranges enforced:
  - `runtime.llm_max_tokens`: `1..=8192`
  - `runtime.probe_timeout_ms`: `1..=60000`
- no hardware-specific assumptions

## Schema overview

Top-level keys:

- `profile_name` (`dev|test|prod`)
- `policy_overlay` (overlay id)
- `device_profile` (`small|medium|large`)
- `slot_modes` (`ebm = shadow|active|off`)
- `paths` (bundle-relative references)
- `strictness` (`determinism_lock`, `stage_isolation`)
- `runtime` (backend/runtime toggles)

## Validate a config

```bash
cargo run -p ucf-ops -- config validate --in configs/test.toml
```

## Migrate legacy config

```bash
cargo run -p ucf-ops -- config migrate --in old.toml --out new.toml --diff ./out/config_diff.txt
```

Migration behavior:

- permissive parse of legacy keys
- unknown legacy keys become warnings
- strict validation of generated `ConfigV1`
- emits bounded actionable diff output
