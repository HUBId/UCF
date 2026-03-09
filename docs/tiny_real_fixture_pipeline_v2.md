# Tiny Real Weights Fixture Pipeline v2 (One Slot, Probe-First)

## Scope

This workflow is intentionally limited to exactly one slot:

- Slot: `world_jepa` (World Predictor)
- Fixture source: `fixtures/weights/world_real_tiny.safetensors.hex`
- Materialized stage dir: `fixtures/weights/world_real_tiny_dir/model.safetensors` (generated deterministically)
- Metadata: `fixtures/weights/world_real_tiny.metadata.toml`

No additional slots are introduced in this prompt.

## Tiny fixture contract

The tiny real fixture is deterministic and offline:

- Tensor `w1`: shape `[16, 2]`, `f32`
- Tensor `b1`: shape `[2]`, `f32`
- Tensor `w2`: shape `[2, 16]`, `f32`
- Tensor `b2`: shape `[16]`, `f32`

Expected fixture SHA-256 is pinned in metadata.

## Probe-first promotion flow

1. Stage

```bash
python - <<'PY'
from pathlib import Path
import binascii
root = Path('.')
hex_path = root / 'fixtures/weights/world_real_tiny.safetensors.hex'
out_dir = root / 'fixtures/weights/world_real_tiny_dir'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'model.safetensors').write_bytes(binascii.unhexlify(hex_path.read_text().strip()))
PY
cargo run -p ucf-ops -- models stage --slot world --path fixtures/weights/world_real_tiny_dir
```

2. Verify manifest/promoted state

```bash
cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_world_tiny.json
```

3. Probe staged hash (no active mutation)

```bash
cargo run -p ucf-ops -- models probe --slot world --hash <H> --out ./out/probe_world_staged.json
```

4. Promote

```bash
cargo run -p ucf-ops -- models promote --slot world --hash <H> --out ./out/models_promote_world_tiny.json
```

5. Probe active (must resolve to promoted hash)

```bash
cargo run -p ucf-ops -- models probe --slot world --out ./out/probe_world_active.json
```

## Required invariants

- Staged probe does not mutate active manifest hash.
- Runtime load path is promoted-only (`models/promoted/...`), never staging.
- Promotion updates `models/MANIFEST.toml` and writes deterministic history under
  `models/manifests/history/`.

## Shadow-only default policy

Default slot mode for world remains Shadow-oriented after promotion:

- Promotion updates model lifecycle state only.
- It does **not** imply active decision impact.
- Decision impact requires explicit operator mode change later.

## Audit trail inspection

- Promotion records: `./out/model_promotion_records.json`
- Manifest history: `models/manifests/history/*.toml`
- Probe reports:
  - `./out/probe_world_staged.json`
  - `./out/probe_world_active.json`

Probe reports include:

- `backend_id`
- `manifest_digest_prefix`
- `model_hash_prefix`
