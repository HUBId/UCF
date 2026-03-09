# Tiny Real Weights Fixture v2 — SAE Slot

## Decision (exactly one slot)

This prompt extends tiny real fixture support to **SAE only**.

Reason: SAE already had a mature and deterministic probe contract (`spike_count`, `spikes_digest`, bounded top-k style semantics), so adding fixture lifecycle support required fewer invasive changes than SSM.

## Fixture + metadata

- Fixture source: `fixtures/weights/sae_real_tiny.safetensors.hex`
- Materialized stage dir: `fixtures/weights/sae_real_tiny_dir/model.safetensors`
- Metadata: `fixtures/weights/sae_real_tiny.metadata.toml`

Pinned SHA-256:

- `0f1ea81381690179efb5058ff06379423142265b2e6e80ca731ecd8ad8330c57`

Tensor contract:

- `sae.w_enc: [64,16] f32`
- `sae.b_enc: [64] f32`

## Deterministic offline lifecycle

```bash
python - <<'PY'
from pathlib import Path
import binascii
root = Path('.')
hex_path = root / 'fixtures/weights/sae_real_tiny.safetensors.hex'
out_dir = root / 'fixtures/weights/sae_real_tiny_dir'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'model.safetensors').write_bytes(binascii.unhexlify(hex_path.read_text().strip()))
PY

cargo run -p ucf-ops -- models stage --slot sae --path fixtures/weights/sae_real_tiny_dir
cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_sae_tiny.json
cargo run -p ucf-ops -- models probe --slot sae --hash <H> --out ./out/probe_sae_staged.json
cargo run -p ucf-ops -- models promote --slot sae --hash <H> --out ./out/models_promote_sae_tiny.json
cargo run -p ucf-ops -- models probe --slot sae --out ./out/probe_sae_active.json
```

## Required invariants

- Staged probe does not mutate active manifest hash.
- Runtime load remains promoted-only (`models/promoted/...`).
- Promotion updates manifest + deterministic history.
- Probe output remains deterministic for fixed slot/hash bytes.

## Shadow-only + no decision impact

- SAE real tiny fixture is **shadow-only by default**.
- Decision path remains unchanged in shadow.
- Compare windows/drift are observational only for this stage.

## Active deny at this stage

Attempting active enablement for SAE in this stage must be denied with stable code:

- `ACTIVE_NOT_ENABLED_FOR_SLOT_STAGE`

In strict mode this remains a hard fail path for active requests at this stage.
