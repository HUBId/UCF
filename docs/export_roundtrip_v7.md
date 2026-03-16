# Export Round-Trip Consistency v7

`ucf-ops exports roundtrip-check` validates that canonical export bundles reconstruct one shared consumption context for verify/inspect flows.

## Command

```bash
cargo run -p ucf-ops -- exports roundtrip-check --in <bundle.zip> --out ./out/export_roundtrip_check.json
```

The command auto-detects Repro Pack vs BugKit, rebuilds `CanonicalBundleConsumptionContextV1`, and emits `BundleRoundTripConsistencyV1`.

## What “round-trip consistency” means

For the same exported bundle bytes, bundle consumption must deterministically recover the same:

- applied scope digest prefix
- policy graph digest prefix
- model manifest digest prefix
- related artifact reference semantics and included-state interpretation
- governance/review/signoff reference consistency

## Mismatch categories

- `BUNDLE_SCOPE_MISMATCH`: applied scope digest context differs.
- `BUNDLE_POLICY_MISMATCH`: policy digest context differs.
- `BUNDLE_MANIFEST_MISMATCH`: manifest/context digest does not reconstruct canonically.
- `BUNDLE_ARTIFACT_REF_MISMATCH`: artifact reference digest/state mismatch.
- `BUNDLE_INCLUDED_STATE_MISMATCH`: included artifact semantics are inconsistent.
- `LEGACY_BUNDLE_LAYOUT`: legacy export layout detected.
- `LEGACY_BUNDLE_TRANSLATED`: deterministic legacy translation path was used.
- `LEGACY_BUNDLE_UNSUPPORTED`: legacy bundle could not be safely translated (fail closed).

All checks are offline, deterministic, bounded, and read-only.
