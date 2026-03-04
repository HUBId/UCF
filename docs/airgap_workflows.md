# Air-Gapped Artifact Import/Export Workflows

`ucf-ops airgap` provides offline-only artifact transport over USB/fileshare.
No network services or remote compute are required.

## Export commands

```bash
ucf-ops airgap export policies --pack policies/packs/base_v1 --overlay policies/packs/overlays/test --out ./out/airgap_policies.zip
ucf-ops airgap export models --slot llm --hash <sha256> --out ./out/airgap_models_llm.zip
ucf-ops airgap export run-cert --run <run_id> --out ./out/airgap_cert_<run_id>.zip
ucf-ops airgap export repro --run <run_id> --out ./out/airgap_repro_<run_id>.zip
```

Every export zip includes:

- bounded artifact files
- `AIRGAP_MANIFEST.json`
- `AIRGAP_MANIFEST.sig`

Manifest fields include deterministic file SHA-256 digests, overall manifest digest,
signer key id/public key, and export timestamp.

## Import commands

```bash
ucf-ops airgap import policies --in ./out/airgap_policies.zip --out ./out/airgap_import_report.json --mode staging
ucf-ops airgap import models --in ./out/airgap_models_llm.zip --out ./out/airgap_import_report.json --mode promoted
ucf-ops airgap import run-cert --in ./out/airgap_cert_<run_id>.zip --out ./out/airgap_import_report.json
ucf-ops airgap import repro --in ./out/airgap_repro_<run_id>.zip --out ./out/airgap_import_report.json
```

Import verification steps:

1. verify manifest signature (ed25519)
2. verify all file SHA-256 values
3. enforce trusted signer allowlist (`airgap_trusted_signer_key_hashes`)
4. run artifact-specific validation (policy validate / model lifecycle / attestation verify / repro verify)
5. write import audit record and report

Use `--allow-untrusted` only for controlled non-prod workflows.

## Trust model and key management

- Signing keys are local files under `<workdir>/keys/attestation_ed25519.{key,pub}`.
- Export signs the manifest digest with this key pair.
- Import computes signer key hash (`sha256(pubkey_bytes)`) and checks allowlist.
- Configure trusted key hashes in policy packs:

```toml
[values]
airgap_trusted_signer_key_hashes = "<hash1>,<hash2>"
```

## Recommended USB workflow

1. On source host: run `airgap export ...`.
2. Move zip via USB/fileshare.
3. On target host: run `airgap import ... --mode staging`.
4. Review `./out/airgap_import_report.json` and `ess/airgap_import_records.json`.
5. Promote staged artifacts through normal governance gates where required.

## Offline guarantees

- Works with local filesystem only.
- No network protocol dependency.
- Deterministic ordering for packaged file entries.
- Audit files contain digests/status only (redaction-safe by default).
