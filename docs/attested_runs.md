# Attested Runs (v1)

`ucf-ops` supports offline run certificates bound to policy + model manifest + Merkleized ESS logs.

## Certificate contents

`RunCertificateV1` includes:

- `schema_version`
- `run_id`
- `started_at`, `ended_at`
- `policy_graph_digest`
- `manifest_digest`
- `final_checkpoint_root` (final Merkle segment root)
- `record_count`
- `summary`
  - `mean_risk_q`
  - `mean_uncertainty_q`
  - `max_governor_tier`
  - `total_violations_count`
- `certificate_digest` (SHA-256 over canonical certificate bytes with digest/signature fields cleared)
- `signature` (Ed25519 over digest bytes)
- `signer_key_id`
- `signer_public_key`

The payload is deterministic; only signature bytes depend on local key material.

## Key handling (local-only)

Generate local keys (if absent):

```bash
cargo run -p ucf-ops -- attest keys
```

Files:

- `.ucf/keys/attestation_ed25519.key` (private)
- `.ucf/keys/attestation_ed25519.pub` (public)

No network or PKI is required. Keys are local and future PKI integration can be layered on top via `signer_key_id`.

## Generate certificate

```bash
cargo run -p ucf-ops -- attest run --run <run_id> --out ./out/run_cert_<run_id>.json
```

This command:

1. Loads run metadata and ESS fixture.
2. Rebuilds/validates Merkle segment chain.
3. Computes bounded summary fields.
4. Computes certificate digest and signs it.
5. Persists an attestation index record to `.ucf/ess/run_attestations.json`.

## Verify certificate (offline)

```bash
cargo run -p ucf-ops -- attest verify --cert ./out/run_cert_<run_id>.json --ess ./.ucf/ess/ess_fixture.json
```

Verification checks:

- signature validity
- certificate digest recomputation
- policy digest match
- model manifest digest match
- Merkle segment chain validity
- final root match

Returns PASS/FAIL and reason list.

## Export redaction-safe bundle

```bash
cargo run -p ucf-ops -- attest bundle --run <run_id> --out ./out/bundle_<run_id>.zip
```

Bundle includes:

- `run_certificate.json`
- `final_checkpoint.json` (digest-only fields)
- `segment_roots.json` (root-only segment manifests)
- optional `readiness_gate_report.json`

No raw output text payloads are included.

## Relation to proof-carrying logs

Attested runs complement Merkleized logs by binding the terminal root state to policy/model provenance and bounded governance metrics in a signed artifact suitable for external, offline verification.
