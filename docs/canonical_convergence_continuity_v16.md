# Canonical Convergence Continuity v16

`CanonicalConvergenceContinuityAuthorityV1` is the single converged top-level continuity proof for canonical operator/export/build/verify flows after v16 convergence cleanup.

## What it proves

`ucf-ops canonical-convergence-continuity-sweep` binds one deterministic authority chain over:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1` + `GovernanceConvergenceSweepV1`
3. `CanonicalReadinessSpineV1` + `ReadinessConvergenceSweepV1`
4. `CanonicalBundleSpineV1` + `BundleConvergenceSweepV1`
5. `CanonicalPrimarySemanticsAuthorityV1` + `PrimarySemanticsConvergenceSweepV1`
6. `OperatorReviewPacketV1`, `OperatorSignoffDecisionV1`, `OperatorWorkflowChainV1`
7. `CanonicalRoundTripChainV1`
8. `UltimateTerminalAbsoluteFinalInputContinuityAuthorityV1` (subordinate legacy contributor)

Top-level PASS/FAIL/LEGACY is emitted only by `CanonicalConvergenceContinuityAuthorityV1`.

## Why this is now the sole top-level proof

All canonical operator/export/build/verify continuity claims must roll up to `canonical-convergence-continuity-sweep`.
Legacy/older continuity surfaces are retained only as subordinate diagnostics and contributors.

## Subordinate contributors

The following are no longer top-level continuity truth surfaces:

- `ultimate-terminal-absolute-final-input-continuity-sweep`
- `terminal-absolute-final-input-continuity-sweep`
- `absolute-final-input-continuity-sweep`
- `final-input-continuity-sweep`
- `CanonicalRoundTripChainV1` and bundle-spine/roundtrip checks

## Command

```bash
cargo run -p ucf-ops -- canonical-convergence-continuity-sweep --bundle <path> --out ./out/canonical_convergence_continuity_sweep.json
```

## Canonical operator/export/build/verify sequence

1. Produce governance/readiness/bundle/primary convergence sweeps.
2. Produce operator review/signoff/workflow artifacts.
3. Verify bundle continuity contributors (`bundle-spine`, `roundtrip`).
4. Run `canonical-convergence-continuity-sweep`.
5. Use only this output for top-level continuity PASS/FAIL.
