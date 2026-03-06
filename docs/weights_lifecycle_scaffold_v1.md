# Weights Lifecycle Scaffold v1

This document defines the offline-first, hardware-neutral v1 weights lifecycle scaffold.

## Layout

All paths are relative to repository/bundle root:

- `models/staging/<slot>/<sha256>/...`
- `models/promoted/<slot>/<sha256>/...`
- `models/MANIFEST.toml`
- `models/manifests/history/<ts>_<digest>.toml`

Runtime loads only promoted paths for enabled model slots.

## Manifest schema (v1)

`models/MANIFEST.toml` contains:

- `manifest_version`
- `created_at` (optional; excluded from digest)
- `manifest_digest`
- `slots = []`, where each slot entry contains:
  - `slot_id`
  - `active_hash` (optional)
  - `files[]` with `path`, `sha256`, `size_bytes`
  - `max_bytes`
  - `contract_versions_supported[]`

### Canonical digest rules

`manifest_digest` is SHA-256 over canonical JSON encoding of:

- `manifest_version`
- sorted `slots` by `slot_id`
- each slot `files` sorted by `path`
- each slot `contract_versions_supported` sorted lexicographically

`created_at` and `manifest_digest` are excluded from digest input.

## Ops commands

- Stage:
  - `cargo run -p ucf-ops -- models stage --slot llm --path ./tmp/model --out ./out/models_stage.json`
- List:
  - `cargo run -p ucf-ops -- models list --slot llm`
- Verify:
  - `cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify.json`

All commands are offline-only and do not download artifacts.

## Runtime behavior

- **Stub-only mode (v0-compatible):** no enabled real slots => manifest may be absent.
- **Any real slot enabled:** startup enforces manifest presence/parse and promoted-only path resolution (`active_hash` or explicit pin).

## Bounds

- Manifest size cap: 1 MiB.
- Per-slot file list cap: 512 entries.

## Example dummy flow

```bash
mkdir -p /tmp/ucf_dummy
printf 'dummy' > /tmp/ucf_dummy/model.safetensors
cargo run -p ucf-ops -- models stage --slot llm --path /tmp/ucf_dummy --out ./out/models_stage.json
cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify.json
```
