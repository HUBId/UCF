# Canonical Bundle Sweep v9

`ucf-ops exports bundle-spine-sweep` is the final canonical proof that covered export surfaces use one bundle authority path:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`

## Covered surfaces

- `repro pack build`
- `repro verify`
- `bugkit build`
- `exports roundtrip-check`
- `exports bundle-spine-check`
- export readiness/build guard helpers
- operator roundtrip-chain helpers

## Why secondary bundle derivation is blocked

Canonical flows fail closed when spine/context/refs are missing or inconsistent.
Legacy/secondary derivations are reported with explicit mismatch categories and cannot become canonical authority.

## Command

```bash
cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json
```

For final canonical primary blocking/remediation semantics across governance/readiness/bundle/review/export/interop/gate surfaces, see `docs/primary_semantics_sweep_v9.md` and run `ucf-ops primary-semantics-sweep`.
