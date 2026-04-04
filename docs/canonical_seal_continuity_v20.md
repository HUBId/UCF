# Canonical Seal Continuity v20

`CanonicalSealContinuityAuthorityV1` is the sole top-level continuity proof for canonical operator/export/build/verify flows after seal cleanup.

## What it proves

`ucf-ops canonical-seal-continuity-sweep` binds one deterministic, offline authority chain over:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `GovernanceSealSweepV1`
4. `CanonicalReadinessSpineV1`
5. `ReadinessSealSweepV1`
6. `CanonicalBundleSpineV1`
7. `BundleSealSweepV1`
8. `CanonicalPrimarySemanticsAuthorityV1`
9. `PrimarySemanticsSealSweepV1`
10. `OperatorReviewPacketV1`, `OperatorSignoffDecisionV1`, `OperatorWorkflowChainV1`
11. `CanonicalRoundTripChainV1`
12. `CanonicalClosureContinuityAuthorityV1` (subordinate)

Top-level PASS/FAIL for canonical operator/export/build/verify now comes only from `CanonicalSealContinuityAuthorityV1`.

## Command

```bash
cargo run -p ucf-ops -- canonical-seal-continuity-sweep --bundle <path> --out ./out/canonical_seal_continuity_sweep.json
```

## Canonical sequence

1. Produce seal sweeps (`governance/readiness/bundle/primary-semantics`).
2. Produce operator artifacts (`operator review-packet`, `operator signoff`, `operator workflow`).
3. Produce bundle evidence (`exports bundle-spine-check`, `exports roundtrip-check`).
4. Run `canonical-seal-continuity-sweep`.
