# Bundle Seal Sweep v20

`ucf-ops bundle-seal-sweep` emits `BundleSealSweepV1` as the bounded proof that covered canonical export consumers derive authoritative bundle truth only from the closure-complete final-consolidated stabilized canonical bundle chain.

## Chain required in-order

`BundleSealSweepV1` requires, in order:

1. `CanonicalExportArtifactRefV1`
2. `CanonicalExportContextV1`
3. `CanonicalBundleConsumptionContextV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`
8. `ResidualFreeBundleConsumerAuthorityV1`
9. `ResidualFreeBundleAbsoluteSweepV1`
10. `AbsoluteFinalBundleTerminalSweepV1`
11. `TerminalBundleUltimateSweepV1`
12. `BundleConvergenceSweepV1`
13. `BundleStabilizationSweepV1`
14. `BundleFinalConsolidationSweepV1`
15. `BundleClosureSweepV1`

Any missing, stale, or contradictory input fails closed with `CLOSURE_COMPLETE_FINAL_CONSOLIDATED_STABILIZED_CANONICAL_BUNDLE_INPUTS_REQUIRED`.

## Covered canonical consumers

- `ReproManifest`
- `BugKitManifest`
- `ReproVerify`
- `RoundTripCheck`
- `InspectSummary`
- `Continuity`
- `ExportReadinessGuard`
- `InteropConsistencyMatrix`
- `V20PrepGateHelper`

## Why shells/bridges/auxiliary views are blocked

Canonical flows cannot source bundle truth from compatibility-shells, bridge layers, or auxiliary export views because those create competing authority paths outside the canonical chain. Residual paths are reported as:

- `CONSUMER_SKIPPED_BUNDLE_SEAL`
- `CONSUMER_USED_BUNDLE_SHELL_PATH`
- `BUNDLE_INPUT_SCOPE_MISMATCH`
- `BUNDLE_INPUT_SPINE_MISMATCH`
- `BUNDLE_INPUT_EXPORT_CONTEXT_MISMATCH`
- `BUNDLE_SHELL_PATH_PRESENT`

Stable denial/translation codes:

- `CLOSURE_COMPLETE_FINAL_CONSOLIDATED_STABILIZED_CANONICAL_BUNDLE_INPUTS_REQUIRED`
- `CANONICAL_EXPORT_ARTIFACT_REFS_REQUIRED`
- `CANONICAL_EXPORT_CONTEXT_REQUIRED`
- `BUNDLE_SHELL_PATH_BLOCKED`
- `BUNDLE_SHELL_PATH_TRANSLATED`
- `BUNDLE_SHELL_PATH_REJECTED`

## Command

```bash
cargo run -p ucf-ops -- bundle-seal-sweep --out ./out/bundle_seal_sweep.json
```
