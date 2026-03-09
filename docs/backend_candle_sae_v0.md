# Backend Candle SAE V0

`backend-candle` now includes a feature-gated, CPU-only SAE adapter for the `sae` slot.

## Build and probe

```bash
cargo build -p ucf-compute --features backend-candle
cargo run -p ucf-ops --features backend-candle -- models probe --slot sae --manifest models/manifest.toml --out ./out/probe_sae_candle.json
```

## WeightSpec (strict)

The Candle SAE adapter validates only the encoder tensors:

- `sae.w_enc` with shape `[F,D]` and dtype `f32`
- `sae.b_enc` with shape `[F]` and dtype `f32`

Decoder tensors are intentionally excluded in v0.

A tiny deterministic fixture is available for probe-first workflows:

- `fixtures/weights/sae_candle_small.safetensors.hex`

## Deterministic inference contract

`CandleSaeAdapterV0` performs:

1. `z = W_enc * x + b_enc` on CPU via Candle.
2. Rust-side deterministic top-k ranking by:
   - value descending,
   - tie-break by `feature_id` ascending.
3. UQ0_16 quantization for spike magnitudes.
4. `spikes_digest` bound to:
   - promoted model hash,
   - SAE input digests,
   - canonicalized spike list.

## Shadow-only semantics

The adapter is shadow-safe by default:

- Missing/invalid weights return `BACKEND_DISABLED` and fall back without crash.
- Real-time decisions remain unaffected in shadow mode.
- Compare windows continue to report mismatch counters (`digest_mismatch_count`) without changing pressure/decision pathways.
