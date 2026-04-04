# Canonical Stabilization Continuity v17

`CanonicalStabilizationContinuityAuthorityV1` is the sole stabilized converged canonical top-level continuity proof for canonical operator/export/build/verify flows in v17.

## What it proves

`ucf-ops canonical-stabilization-continuity-sweep` binds exactly one deterministic authority chain over:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1` + `GovernanceStabilizationSweepV1`
3. `CanonicalReadinessSpineV1` + `ReadinessStabilizationSweepV1`
4. `CanonicalBundleSpineV1` + `BundleStabilizationSweepV1`
5. `CanonicalPrimarySemanticsAuthorityV1` + `PrimarySemanticsStabilizationSweepV1`
6. `OperatorReviewPacketV1`, `OperatorSignoffDecisionV1`, `OperatorWorkflowChainV1`
7. `CanonicalRoundTripChainV1`
8. `CanonicalConvergenceContinuityAuthorityV1` as subordinate continuity contributor

Top-level PASS/FAIL/LEGACY is emitted only by `CanonicalStabilizationContinuityAuthorityV1`.

## Why this is the only top-level proof now

After v17 stabilization cleanup, all canonical operator/export/build/verify continuity claims must roll up to `canonical-stabilization-continuity-sweep`.
No adapter/translation/projection/memoized/copied/local reconstruction fallback path may remain outside this authority.

## Command

```bash
cargo run -p ucf-ops -- canonical-stabilization-continuity-sweep --bundle <path> --out ./out/canonical_stabilization_continuity_sweep.json
```

## Canonical operator/export/build/verify sequence

1. Run stabilization contributors (`governance-stabilization-sweep`, `readiness-stabilization-sweep`, `bundle-stabilization-sweep`, `primary-semantics-stabilization-sweep`).
2. Produce operator artifacts (`operator review-packet`, `operator signoff`, `operator workflow`).
3. Build/inspect bundle contributors (`exports bundle-spine-check`, `operator roundtrip-chain-check`).
4. Run `canonical-stabilization-continuity-sweep`.
5. Use only this output as top-level continuity PASS/FAIL authority.

## Subordinate continuity contributors

The following remain subordinate/legacy diagnostics and are not top-level continuity truth:

- `canonical-convergence-continuity-sweep`
- `ultimate-terminal-absolute-final-input-continuity-sweep`
- `terminal-absolute-final-input-continuity-sweep`
- `absolute-final-input-continuity-sweep`
- `final-input-continuity-sweep`
- `residual-free-continuity-sweep`
- `final-continuity-sweep`
- `continuity-authority-check`


> v19 closure finalization: sole top-level continuity proof is now `canonical-closure-continuity-sweep` (`CanonicalClosureContinuityAuthorityV1`). `canonical-stabilization-continuity-sweep` remains subordinate continuity evidence.

> v20 seal finalization: this remains subordinate continuity evidence under `CanonicalSealContinuityAuthorityV1`.
