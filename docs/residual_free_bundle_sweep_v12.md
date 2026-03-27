# Residual Free Bundle Sweep (v12)

`residual-free-bundle-sweep` is the v12 proof artifact that canonical export consumers derive bundle truth only from residual-free final bundle inputs.

## Covered authoritative inputs

Canonical consumers must resolve and validate:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`

The command emits `ResidualFreeBundleConsumerAuthorityV1` with deterministic PASS/FAIL/LEGACY_PRESENT status.

## Command

```bash
cargo run -p ucf-ops -- residual-free-bundle-sweep --out ./out/residual_free_bundle_sweep.json
```

## Why this exists

v12 blocks the last historical, implicit, and bundle-local reconstruction traces in canonical export flows. If required final inputs are missing, stale, or contradictory, canonical flows fail closed.


## v13 note

v13 adds `bundle-absolute-sweep`, which consumes `ResidualFreeBundleConsumerAuthorityV1` and removes the final historical/bundle-local lineage traces from canonical export consumers.

## v14 terminal note

The v14 terminal sweep closes the final canonical export consumer residual paths by requiring absolute final bundle inputs across covered consumers.


## v15 note

Canonical export consumers now reference `bundle_ultimate_sweep_digest_prefix`, and cache/mirror/snapshot reconstruction paths are blocked from canonical use.
