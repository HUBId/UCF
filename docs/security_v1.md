# Security Hardening v1

## Scope
- Runtime is offline-by-design: no network capability issuance and no process-exec issuance.
- Policy decisions are bound to a verified immutable policy bundle.

## Policy bundle workflow
1. Bundle files live in `policies/bundle_v1/`.
2. `policies/manifest.toml` pins per-file SHA-256 and `bundle_sha256`.
3. Startup requires `UCF_POLICY_BUNDLE_SHA256=<bundle hash>` and fails fast on mismatch.
4. Runtime writes a `PolicyProvenanceRecord` including run_id, version, bundle hash, enabled features.

## Sandbox FS v1
- Capability types:
  - `FileRead { root_id }`
  - `FileWrite { root_id }` (reserved/disabled by default)
- Roots are policy allowlisted (`models_root`, `out_root`, `ess_root`).
- `SandboxFs` enforces:
  - normalized relative paths
  - traversal rejection (`..`)
  - canonical root containment (symlink escape protection)

## Audit/security chain verification
- Security-critical records include policy bundle hash and are linked through audit digests.
- Verify with:
  - `ucf-ops security verify-chain --from 0 --to <tick>`
- Command reports first break as an error.
