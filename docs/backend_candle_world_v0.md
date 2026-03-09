# Backend Candle World V0

`backend-candle` introduces a feature-gated, CPU-only Candle adapter skeleton for the `world_jepa` slot.

## Feature flag

Default builds stay on stubs/toy backends.

```bash
cargo build -p ucf-compute
cargo build -p ucf-compute --features backend-candle
cargo run -p ucf-ops --features backend-candle -- models probe --manifest models/MANIFEST.toml --out ./out/probe_report.json
```

## Weight spec (strict)

When promoted world weights are present, `CandleWorldAdapterV0` validates strict `WeightSpec` tensors:

- `w1`: `[D,H]` `f32`
- `b1`: `[H]` `f32`
- `w2`: `[H,D]` `f32`
- `b2`: `[D]` `f32`

If the promoted weights are missing or invalid, the adapter is disabled and returns `BACKEND_DISABLED`.

## Shadow-only semantics

The adapter is probe-first and shadow-first:

- default runtime remains unaffected without the feature
- world probe prefers candle adapter only when `backend-candle` is enabled **and** verified weights exist
- decision path remains unchanged; compare windows report primary vs shadow deltas (`mean_delta_q`, `p95_delta_q`)

## Probe interpretation

`models probe` includes a `backend_id` carrying pack + world backend component id:

- format: `<pack_name>:<pack_digest_prefix>/world:<component_id>`
- if candle world cannot initialize from local promoted weights, probe falls back to stub/toy path and remains non-crashing
