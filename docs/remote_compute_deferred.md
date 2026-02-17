# Remote Compute v2 (Deferred)

## Status

Remote compute remains **disabled in v1**. This document describes the deferred architecture skeleton only.
No network path is enabled by default.

## Enablement Requirements (Future)

Remote compute can only be activated when **all** of the following are true:

1. Compile-time feature `remote-compute` is enabled.
2. Runtime operator opt-in is set (`UCF_REMOTE_ENABLE=1`).
3. Policy allowlist enables remote compute and includes the active policy bundle hash.

If any condition is missing, remote backend construction is denied.

## Security Model

- Explicit proxy model for remote compute client wiring.
- Signed requests (node signer) over canonical request bytes.
- Response integrity verification (request id + signature).
- Zero-trust stance: allowlisted endpoints/stages only (schema in policy allowlist).
- Bounded payload sizes, timeout budget, concurrency/rate controls (governor skeleton).
- Audit schemas include remote call and denial records.

## v1 Invariant

- `network.enabled` remains false in policy defaults.
- `remote_compute.enabled` remains false in policy defaults.
- No tests require network connectivity.
