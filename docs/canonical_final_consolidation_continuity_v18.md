# Canonical Final Consolidation Continuity (v18)

`CanonicalFinalConsolidationContinuityAuthorityV1` is the **sole top-level continuity proof** for canonical operator/export/build/verify flows after v18 final consolidation cleanup.

## What it proves
`ucf-ops canonical-final-consolidation-continuity-sweep` binds one deterministic chain over:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `GovernanceFinalConsolidationSweepV1`
4. `CanonicalReadinessSpineV1`
5. `ReadinessFinalConsolidationSweepV1`
6. `CanonicalBundleSpineV1`
7. `BundleFinalConsolidationSweepV1`
8. `CanonicalPrimarySemanticsAuthorityV1`
9. `PrimarySemanticsFinalConsolidationSweepV1`
10. `OperatorReviewPacketV1`, `OperatorSignoffDecisionV1`, `OperatorWorkflowChainV1`
11. `CanonicalRoundTripChainV1`
12. `CanonicalStabilizationContinuityAuthorityV1` (subordinate contributor)

It fail-closes on residual paths, legacy top-level continuity surfaces, and cross-chain digest drift.

## Command
```bash
cargo run -p ucf-ops -- canonical-final-consolidation-continuity-sweep --bundle <path> --out ./out/canonical_final_consolidation_continuity_sweep.json
```

## Canonical sequence
1. Run governance/readiness/bundle/primary final-consolidation sweeps.
2. Run operator review/signoff/workflow.
3. Build/verify bundle.
4. Run `canonical-final-consolidation-continuity-sweep`.
5. Use only this output for top-level continuity PASS/FAIL.

## Subordinate continuity contributors
These surfaces remain diagnostic/subordinate only:
- `canonical-stabilization-continuity-sweep`
- `canonical-convergence-continuity-sweep`
- `ultimate-terminal-absolute-final-input-continuity-sweep`
- `terminal-absolute-final-input-continuity-sweep`
- `absolute-final-input-continuity-sweep`
- `final-input-continuity-sweep`
- `CanonicalRoundTripChainV1`, bundle spine/roundtrip checks
