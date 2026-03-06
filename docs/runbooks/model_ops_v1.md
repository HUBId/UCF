# Model Ops Runbook v1 (Offline)

This runbook documents deterministic, human-in-the-loop model lifecycle operations.

## Commands

Stage candidate:

```bash
cargo run -p ucf-ops -- models stage --slot llm --path ./tmp/model --out ./out/models_stage.json
```

Verify lifecycle manifest:

```bash
cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify.json
```

Promote staged hash (requires staged verify PASS):

```bash
cargo run -p ucf-ops -- models promote --slot llm --hash <sha256> --out ./out/models_promote.json
```

Rollback to explicit promoted hash:

```bash
cargo run -p ucf-ops -- models rollback --slot llm --to <sha256> --out ./out/models_rollback.json
```

Rollback to previous promoted hash from history:

```bash
cargo run -p ucf-ops -- models rollback --slot llm --steps 1 --out ./out/models_rollback.json
```

## Expected artifacts

- `./out/models_stage.json`
- `./out/models_verify.json`
- `./out/models_promote.json`
- `./out/models_rollback.json`
- `models/MANIFEST.toml`
- `models/manifests/history/<ts>_<digest>.toml`

## Safety model

- Promotion never activates staging directly.
- Runtime consumes only `models/promoted/...` via `active_hash` in `models/MANIFEST.toml`.
- Rollback is operator-triggered; no automatic rollback path is enabled in runtime.
