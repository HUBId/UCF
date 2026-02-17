# Model Slots (Local-only, hash-locked)

`ucf-compute` supports local model slots (`llm`, `world_jepa`, `sae`, `lfm`, `ssm`) via `models/manifest.toml`.

## Guarantees
- no network fetch path (filesystem only)
- allowlisted root (`allowlist_root`, default `models/`)
- canonicalized path checks (reject traversal / outside root)
- max-bytes cap per slot
- SHA-256 must match expected hash
- mismatch disables slot (safe fallback to toy/stub)

## Manifest
Use `models/manifest.toml`:

```toml
allowlist_root = "models"

[slots.llm]
enabled = true
path = "llm.bin"
expected_sha256 = "<64 hex chars>"
max_bytes = 67108864
format = "candle_bin"
device = "cpu_only"
```

## Env overrides
- `UCF_MODEL_<SLOT>_PATH`
- `UCF_MODEL_<SLOT>_SHA256`
- `UCF_MODEL_<SLOT>_MAX_BYTES`
- `UCF_MODEL_<SLOT>_ENABLED`

`<SLOT>`: `LLM`, `WORLD_JEPA`, `SAE`, `LFM`, `SSM`.

## Verify
```bash
cargo run -p ucf-ops -- models verify --manifest models/manifest.toml
```

## SHA-256
```bash
sha256sum models/llm.bin
```

Only the hash + metadata are persisted in records (not model bytes).
