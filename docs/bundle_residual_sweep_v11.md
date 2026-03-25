# Bundle Residual Sweep v11

`bundle-residual-sweep` proves that canonical export consumers no longer derive bundle truth from
legacy manifest parsing, ad-hoc inspect/report summaries, or roundtrip reconstruction paths as
primary authority.

## Covered consumers

- repro pack build
- repro verify
- bugkit build
- exports roundtrip check

All covered consumers are evaluated against the final authoritative bundle input chain:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`

## Why residual reconstruction is disallowed

Canonical export flows must fail-closed when final bundle inputs are missing, stale, or contradictory.
This prevents latent divergence where local manifest hints or inspect-only summaries become accidental
authority.

## Command

```bash
cargo run -p ucf-ops -- bundle-residual-sweep --out ./out/bundle_residual_sweep.json
```

## v12 residual-free consumer finalization

v12 adds `residual-free-bundle-sweep` and `ResidualFreeBundleConsumerAuthorityV1`, which fail-closed canonical export consumers unless residual-free final bundle inputs are the sole authority.


## v13 note

v13 closes remaining canonical consumer lineage paths by requiring the absolute residual-free bundle input chain via `bundle-absolute-sweep`.
