# Model Slots (Local-only, hash-locked)

`ucf-compute` supports local model slots (`llm`, `world_jepa`, `world_vljepa`, `sae`, `lfm`, `ssm`, `ebm_reasoner`) via `models/manifest.toml`.

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
- Canonical manifest source remains `models/manifest.toml`.
- `UCF_MODEL_MANIFEST` is a **legacy/explicit compatibility override only**; production bootstrap should keep the canonical path.

Per-slot overrides:
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

## Candle safetensors weight specs (v1)
For `format = "candle_safetensors"`, `ucf-compute` validates required tensor names, exact shapes, and dtypes before enabling a slot.

### `world_jepa` (JEPA v1)
Required tensors:
- `W1: [D,H] f32`
- `b1: [H] f32`
- `W2: [H,D] f32`
- `b2: [D] f32`

### `sae` (SAE v1)
Required tensors:
- `W: [F,D] f32`
- `b: [F] f32`

### `ssm` (SSM v1)
Required tensors:
- `A: [N,N] f32` (v1 uses deterministic diagonal/structured scan path)
- `B: [N] f32`
- `C: [N] f32`

### `lfm` (LFM LNN v1)
Required tensors:
- `alpha: [N] f32`
- `Wx: [N,N] f32`
- `Wu: [N] f32`
- `b: [N] f32`

### `llm` (Candle CPU v1 tiny)
Required tensors:
- `tok_emb: [32,64] f32`
- `lm_head: [64,32] f32`

Tokenizer asset (hash-locked, offline) is required for active LLM slot loading:
- default path: `runtime/ucf-compute/fixtures/llm_v1_tiny_vocab.json`
- override: `UCF_LLM_TOKENIZER_PATH`
- hash override: `UCF_LLM_TOKENIZER_SHA256`

If tokenizer hash verification fails, slot creation falls back safely to stub/toy backend.

Dimension symbols (`D/H/F/N`) are slot-local bind variables and must stay consistent across tensors in a slot.

## Runtime compatibility + failure semantics

`runtime/ucf-compute` resolves each slot into a structured runtime status:

- `used`
- `disabled`
- `unavailable`
- `verification_failed`
- `incompatible`

Failure codes are emitted per slot and distinguish:

- `disabled`
- `missing_path`
- `missing_expected_hash`
- `hash_mismatch`
- `oversized`
- `path_violation`
- `artifact_unavailable`
- `artifact_incompatible`

Canonical pipeline failures map these slot outcomes into explicit failure kinds:

- `artifact_unavailable`
- `artifact_verification_failed`
- `artifact_incompatible`
- `backend_disabled`
- `stage_contract_mismatch`
- `degraded_fallback`
