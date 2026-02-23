# Determinism Lock v1

UCF enforces deterministic execution by policy.

## Policy

`DeterminismPolicyV1` is part of policy-pack loading and policy graph digesting.

- `allowed_rng_sites`: explicit allowlist of RNG sites.
- `allowed_mode`: defaults to `deterministic_only`.
- `global_seed_source`: derived from `run_id + policy_graph_digest`.

The determinism policy digest is persisted in policy graph provenance.

## RNG Registry

RNG usage is deny-by-default.

- Runtime code must request RNG by `RngSiteId`.
- Seed derivation: `H(run_id || policy_graph_digest || site_id)`.
- No OS entropy access is used.
- Both allow and deny paths are represented by audit-safe digest-only records.

## Sampling Enforcement

LLM sampling is blocked by default:

- `temperature > 0` => denied
- `top_p != 1` => denied
- `sampling_enabled = true` => denied

Error code: `SAMPLING_DISABLED`.

## CI / Local Scan

Use:

```bash
cargo run -p ucf-ops -- determinism scan
```

The scan fails on runtime uses of `thread_rng`, `rand::random`, `getrandom`, `OsRng`.

## Dev-only Exceptions

If a dev profile requires RNG, add an explicit `RngSiteId` allowlist entry in `determinism.toml` and keep it out of production pack overlays.
