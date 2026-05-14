# Backend Burn World V0


Status note: this is an optional-real compile/runtime seam for a bounded Burn skeleton, not a production backend or runtime-inference proof without a pinned local artifact-backed fixture and deterministic runtime golden test.
`backend-burn` adds a feature-gated, CPU-only Burn skeleton for exactly one slot: `world_jepa`.

## Build / probe

```bash
cargo build -p ucf-compute --features backend-burn
cargo run -p ucf-ops --features backend-burn -- models probe --slot world_jepa --manifest models/manifest.toml --out ./out/probe_world_burn.json
```

## Contract and fixture

`BurnWorldAdapterV0` implements the existing `WorldPredictorV1` contract without schema drift:

- `prediction_digest`
- `prediction_error_q`
- `surprise_q`
- same contract version/backends fields used by existing stage v1 flow

The adapter validates the same `WeightSpec` tensor layout as Candle world v0:

- `w1 [D,H]`
- `b1 [H]`
- `w2 [H,D]`
- `b2 [D]`

Tiny promoted fixtures are sufficient; no large model download is required.

## Probe-first and shadow-only

- Probe-first: Burn world is intended to be exercised through `models probe` before any rollout.
- Missing or invalid world fixture disables the Burn adapter path (`BACKEND_DISABLED`) without crash.
- Shadow-only default: Burn world diagnostics are allowed in shadow; decision impact is unchanged.
- Active mode for Burn world is intentionally denied in v0 (`ACTIVE_DENIED_BACKEND_NOT_YET_ALLOWED`).

This v0 is architectural parity scaffolding for backend interchangeability, not performance tuning.
