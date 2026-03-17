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
- `evidence/backend_evidence_snapshot.json` (optional, included only if context-consistent)
- `evidence/active_review_snapshot.json` (optional, included only if context-consistent)
- `evidence/operator_signoff.json` (optional, included only if context-consistent)

## Build

```bash
cargo run -p ucf-ops -- repro pack --run <run_id> --range last --out ./out/repro_<run_id>.zip
```

Notes:
- `--range last` is accepted for workflow compatibility and currently maps to bounded last records behavior.
- The pack is deterministic: file set/order is canonicalized and digested.
- `repro_pack_manifest.json` now contains additive evidence reference blocks (`backend_evidence_snapshot`, `active_review_snapshot`, `operator_signoff`, `backend_resolution`) and a bounded `evidence_context` digest-prefix summary.
- Evidence refs are explicit by status: `INCLUDED`, `MISSING`, or `EXCLUDED` (with `reason_code`).

## Verify/Rehydrate

```bash
cargo run -p ucf-ops -- repro verify --pack ./out/repro_<run_id>.zip --out ./out/repro_verify.json
cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_<run_id>.zip --out ./out/export_roundtrip_check.json
```

Verification flow:
1. unpack to temp dir
2. verify all artifact SHA-256 values from manifest
3. verify `repro_pack_digest`
4. verify `policy_graph_digest` and `manifest_digest` cross-file consistency
5. verify included evidence artifact hashes and context consistency against manifest `evidence_context`
6. run replay in verify-only mode on included ESS slice
7. emit PASS/FAIL JSON report with first divergence (if any)

## Bug-report attachment guidance

Attach:
- `./out/repro_<run_id>.zip`
- `./out/repro_verify.json`

This is safe-by-default for offline sharing because it excludes raw model weights and raw payload extras.

## Optional manual weights handoff

If full model-byte rehydration is required, transmit model files separately via your secure channel and do **not** add them to the default repro pack.


## Export normalization v6

This surface participates in canonical export normalization (shared `CanonicalExportArtifactRefV1` and `CanonicalExportContextV1`) and is validated via `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`. See `docs/export_normalization_v6.md` for semantics.

## Bundle-Spine-Validierung

Nach dem Build sollte zusätzlich die Spine-Prüfung laufen:

```bash
cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_<run_id>.zip --out ./out/bundle_spine_check.json
```

