# Final Bundle Consumer Sweep v10

`ucf-ops final-bundle-consumer-sweep` proves that canonical export consumers are anchored to one final bundle authority chain:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`

## Covered canonical consumers

- repro pack build
- repro verify
- bugkit build
- exports roundtrip check
- exports bundle spine check
- operator roundtrip-chain helpers
- export readiness/build guards
- interop/export matrix helpers
- v10 prep gate helpers

## What the sweep proves

- every covered consumer carries identical scope/governance/readiness prefixes from final authority inputs
- canonical bundle spine + canonical bundle authority prefixes are present in major export manifests
- legacy bundle layout inputs are demoted from canonical truth and reported as `LEGACY_PRESENT`
- failures are fail-closed and produce deterministic mismatch categories

Mismatch categories:

- `CONSUMER_SKIPPED_FINAL_BUNDLE_AUTHORITY`
- `CONSUMER_USED_LEGACY_BUNDLE_INPUT`
- `FINAL_BUNDLE_SCOPE_MISMATCH`
- `FINAL_BUNDLE_SPINE_MISMATCH`
- `FINAL_BUNDLE_EXPORT_CONTEXT_MISMATCH`
- `LEGACY_BUNDLE_INPUT_PRESENT`

## Command

```bash
cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json
```

## v11 residual cleanup

v11 adds `bundle-residual-sweep` to remove and block the last residual bundle-reconstruction paths in canonical export flows.
