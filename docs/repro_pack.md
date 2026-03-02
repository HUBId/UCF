# Repro Pack v1

`Repro Pack` is a deterministic, redaction-safe, offline bundle for incident reproduction.

## Contents

Default `ReproPackV1` zip includes:

- `repro_pack_manifest.json` (canonical manifest + `repro_pack_digest`)
- `config_resolved.json` (`ConfigV1`-derived resolved runtime config)
- `policy_ref.json` (`policy_graph_digest`, base/overlay identifiers)
- `models_ref.json` (`manifest_digest` + active slot hashes only; no weights bytes)
- `ess_slice.json` (bounded ESS slice; default last 2048 records)
- `segment_roots.json` (Merkle segment roots for included slice)
- `run_certificate.json` (optional, when present)
- `readiness_gate_report.json` (optional, when present)

## Build

```bash
cargo run -p ucf-ops -- repro pack --run <run_id> --range last --out ./out/repro_<run_id>.zip
```

Notes:
- `--range last` is accepted for workflow compatibility and currently maps to bounded last records behavior.
- The pack is deterministic: file set/order is canonicalized and digested.

## Verify/Rehydrate

```bash
cargo run -p ucf-ops -- repro verify --pack ./out/repro_<run_id>.zip --out ./out/repro_verify.json
```

Verification flow:
1. unpack to temp dir
2. verify all artifact SHA-256 values from manifest
3. verify `repro_pack_digest`
4. verify `policy_graph_digest` and `manifest_digest` cross-file consistency
5. run replay in verify-only mode on included ESS slice
6. emit PASS/FAIL JSON report with first divergence (if any)

## Bug-report attachment guidance

Attach:
- `./out/repro_<run_id>.zip`
- `./out/repro_verify.json`

This is safe-by-default for offline sharing because it excludes raw model weights and raw payload extras.

## Optional manual weights handoff

If full model-byte rehydration is required, transmit model files separately via your secure channel and do **not** add them to the default repro pack.
