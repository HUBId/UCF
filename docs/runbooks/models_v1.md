# Models Lifecycle Runbook v1 (Scaffold, Offline)

This runbook describes the deterministic operator sequence for the v1 **models lifecycle scaffold**.

> Scope note: v1 model lifecycle is governance scaffolding (hash-locked manifests, probes, promotion/rollback flow). It does **not** claim production-grade model quality by itself.

## Preconditions

- Work from repository root.
- Keep operation offline (no network dependencies).
- Use repository-local artifact paths under `./out/`.

## 1) Stage candidate bytes

```bash
cargo run -p ucf-ops -- models stage --slot llm --path ./fixtures/models_dummy/llm --out ./out/models_stage_llm.json
```

Expected result:

- staged artifact hash emitted in report
- candidate bytes copied under model staging storage

## 2) Verify manifest integrity

If your repo uses lowercase manifest:

```bash
cargo run -p ucf-ops -- models verify --manifest models/manifest.toml --out ./out/models_verify.json
```

If your repo uses uppercase manifest:

```bash
cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify.json
```

## 3) Probe staged candidate (no activation)

```bash
cargo run -p ucf-ops -- models probe --slot llm --hash <staged_sha256> --out ./out/probe_llm_staged.json
```

Promote only after probe status is `PASS`.

## 4) Promote staged hash

```bash
cargo run -p ucf-ops -- models promote --slot llm --hash <staged_sha256> --out ./out/models_promote_llm.json
```

## 5) Probe active/promoted slot

```bash
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm_active.json
```

## 6) Rollback (if needed)

Rollback to explicit known-good hash:

```bash
cargo run -p ucf-ops -- models rollback --slot llm --to <known_good_sha256> --out ./out/models_rollback_llm.json
```

Or rollback to previous promoted hash:

```bash
cargo run -p ucf-ops -- models rollback --slot llm --steps 1 --out ./out/models_rollback_llm.json
```

## Artifacts you should see

- `./out/models_stage_llm.json`
- `./out/models_verify.json`
- `./out/probe_llm_staged.json`
- `./out/models_promote_llm.json`
- `./out/probe_llm_active.json`
- `./out/models_rollback_llm.json` (only when rollback executed)
- `models/manifest.toml` or `models/MANIFEST.toml`
- `models/manifests/history/<timestamp>_<digest>.toml`

## Troubleshooting

- Verify fails:
  - check manifest path casing (`models/manifest.toml` vs `models/MANIFEST.toml`)
  - rerun `models verify` and fix reported hash/path mismatch
- Probe fails:
  - keep slot non-promoted; inspect `envelope_checks` in probe report
  - restage candidate after fixing fixture/weights
- Promote fails:
  - ensure hash came from stage report and probe report is `PASS`
  - rerun readiness checks before reattempting promotion
