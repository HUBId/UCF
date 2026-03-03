# Public Bug Report Kit (BugKit)

`bugkit` packages reproducible, redaction-safe diagnostics into one uploadable `.zip`.

## Build

```bash
cargo run -p ucf-ops -- bugkit build --run <run_id> --out ./out/bugkit_<run_id>.zip
```

Default behavior:
- Includes `repro_pack.zip` (built automatically if missing).
- Includes `diagnostics_bundle.zip` (redacted diagnostics collection).
- Includes `docs/spec_snapshot.md` and `policy_graph_ref.json`.
- Includes available top-level diagnostics artifacts (strict/gate/drift/docs lint) as optional entries.
- Enforces `50MB` default cap (`--max-bytes` overrides).
- Drops optional artifacts first if over cap and records warnings in `BUGKIT_MANIFEST.json`.

## Verify offline

```bash
python scripts/verify_bugkit.py --bugkit ./out/bugkit_<run_id>.zip
```

The verifier:
1. Extracts the kit into a temporary directory.
2. Verifies per-file `sha256` checksums from `BUGKIT_MANIFEST.json`.
3. Verifies `bugkit_digest` integrity.
4. Runs `ucf-ops repro verify` against bundled `repro_pack.zip`.

No network access is required.

## Include flags

By default, sensitive payload and weight exports are disabled.

- `--include_payload` (default `false`): marks manifest and emits warning.
- `--include_weights` (default `false`): discouraged; manifest warning is emitted.

Example:

```bash
cargo run -p ucf-ops -- bugkit build --run <run_id> --out ./out/bugkit_<run_id>.zip --include_payload
```

## One-file upload workflow

1. Build bugkit zip.
2. Verify locally with `scripts/verify_bugkit.py`.
3. Upload single `bugkit_<run_id>.zip` to issue tracker.
4. Share optional warning context from `BUGKIT_MANIFEST.json` if size-capped.
