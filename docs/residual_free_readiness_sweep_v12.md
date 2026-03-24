# Residual-Free Readiness Sweep v12

`residual-free-readiness-sweep` is the v12 ultimate canonical consumer proof for readiness.

## What it proves

The command proves that covered canonical consumers now derive readiness/reviewability only from these residual-free final inputs:

1. `SlotReviewabilityTruthV1`
2. `ReviewabilityReductionV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalReadinessAuthorityV2`
5. `FinalReadinessConsumerAuthorityV1`
6. `FinalReadinessResidualSweepV1`

It emits `ResidualFreeReadinessConsumerAuthorityV1` with:

- applied supported-set digest prefix
- canonical governance entry digest prefix
- canonical readiness spine digest prefix
- canonical readiness authority digest prefix
- final readiness consumer authority digest prefix
- final readiness residual sweep digest prefix
- covered consumer count
- residual path count
- authority status (`PASS | FAIL | LEGACY_PRESENT`)
- authority digest

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V12PrepGateHelper`

## Why historical/implicit/aggregate readiness reconstruction is disallowed

Canonical flows must fail closed if residual-free final readiness inputs are missing, stale, or contradictory. Historical reconstruction (aggregate snapshot as primary readiness truth, stage-history reconstruction, raw evidence hints, implicit stage traces) is treated as blocked legacy behavior, not canonical authority.

## Command

```bash
cargo run -p ucf-ops -- residual-free-readiness-sweep --out ./out/residual_free_readiness_sweep.json
```


## v13 note

v13 adds the readiness absolute sweep to remove any remaining historical or aggregate readiness lineage traces from canonical consumers.
