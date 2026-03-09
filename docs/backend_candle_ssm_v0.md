# backend-candle SSM v0 Adapter (CPU, Deterministic Scan)

This document specifies the v0 Candle-backed SSM adapter path for `backend-candle`.

## Scope

- slot: `ssm`
- lane: CPU-only
- mode: shadow-first (observational)
- determinism: scan update remains a deterministic Rust loop

## WeightSpec

The SSM v0 Candle adapter validates `candle_safetensors` with strict names/dtypes/shapes:

- required:
  - `ssm.a: [N] f32`
  - `ssm.b: [N] f32`
- optional:
  - `ssm.c: [N] f32`
  - `ssm.init: [N] f32`

Probe fixture (tiny deterministic, text-safe for review tooling):

- `fixtures/weights/ssm_candle_small.safetensors.hex`
- fixture size uses `N=16`

## Deterministic scan rationale

Even when Candle is enabled, the state update loop is executed in Rust in fixed index order:

- `s[i] = a[i] * s[i] + b[i] * inp_i (+ c[i] optional)`
- clamped to `[-1, 1]`

This avoids backend-dependent reduction/reordering drift in the core selective scan semantics.

Candle is used for safe tensor loading/validation and optional elementwise parameter handling.

## Probe

Run probe with candle feature and slot pinning:

```bash
cargo run -p ucf-ops --features backend-candle -- models probe --slot ssm --manifest models/manifest.toml --out ./out/probe_ssm_candle.json
```

SSM probe exports:

- `pressure_q`
- `state_digest_prefix`

## Shadow compare + drift

Window compare records include pressure and digest disagreement via existing compare fields:

- pressure delta through compare `mean_delta_q`/`p95_delta_q` (pressure-based for SSM)
- digest mismatch through `digest_mismatch_count` (SSM digest prefix)

If severe drift persists, shadow is disabled according to compare/drift policy thresholds.

## Failure behavior

If the SSM slot is enabled in the model manifest, missing/invalid candle weights fail closed with
`BACKEND_DISABLED`. If the SSM slot is disabled, candle lane keeps deterministic hash-derived toy
initialization for compatibility in non-model-slot test paths.


To materialize a local `.safetensors` file for manual experiments:

```bash
python - <<'PY'
from pathlib import Path
import binascii
hex_path = Path("fixtures/weights/ssm_candle_small.safetensors.hex")
out_path = hex_path.with_suffix("")
out_path.write_bytes(binascii.unhexlify(hex_path.read_text().strip()))
print(out_path)
PY
```
