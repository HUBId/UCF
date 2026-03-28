# Bundle Convergence Sweep v16

`ucf-ops bundle-convergence-sweep` emits `BundleConvergenceSweepV1` as the compact proof that
canonical export consumers derive authoritative bundle truth only from the terminal canonical chain.

## What this proves

The sweep resolves and validates this exact chain before checking consumers:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalBundleSpineV1`
5. `CanonicalBundleAuthorityV2`
6. `FinalBundleConsumerAuthorityV1`
7. `FinalBundleResidualSweepV1`
8. `ResidualFreeBundleConsumerAuthorityV1`
9. `ResidualFreeBundleAbsoluteSweepV1`
10. `AbsoluteFinalBundleTerminalSweepV1`
11. `TerminalBundleUltimateSweepV1`

The report fails closed when these inputs are missing, stale, or contradictory.

## Covered consumers

Current canonical consumer checks include:

- `ReproManifest` (`out/repro_pack_manifest.json`)
- `BugKitManifest` (`out/bugkit_manifest.json`)
- `ReproVerify` (`out/repro_verify.json`)
- `RoundTripCheck` (`out/export_roundtrip_check.json`)
- `BundleSpineCheck` (`out/bundle_spine_check.json`)
- `ContinuityArtifacts` (`out/ultimate_terminal_absolute_final_input_continuity_sweep.json`)
- `InspectSummary` (`out/export_inspect_report.json`)

## Why memoized/copied/derived bundle truth is blocked

Canonical export flows must not treat any of the following as primary bundle authority:

- memoization caches
- copied manifest/export records
- derived export mirrors
- bundle lineage/history echoes
- inspect/report summaries

These surfaces are now mismatch classes and produce deterministic non-pass outcomes.

## Command

```bash
cargo run -p ucf-ops -- bundle-convergence-sweep --out ./out/bundle_convergence_sweep.json
```
