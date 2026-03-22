# Readiness Residual Sweep v11

`readiness-residual-sweep` is the final canonical readiness residual cleanup authority for v11.

## What it proves

The sweep proves that covered canonical consumers derive readiness/reviewability from final readiness inputs only:

1. `SlotReviewabilityTruthV1`
2. `ReviewabilityReductionV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalReadinessAuthorityV2`
5. `FinalReadinessConsumerAuthorityV1`

It emits `FinalReadinessResidualSweepV1` with:

- applied/governance/spine/authority/final-consumer digest prefixes
- covered consumer count
- residual path count
- sweep status (`PASS | FAIL | LEGACY_PRESENT`)
- deterministic sweep digest

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V11PrepGateHelper` (v10 gate report linkage check)

## Why residual readiness reconstruction is no longer allowed

Canonical flows are fail-closed when final readiness inputs are missing, stale, contradictory, or replaced by legacy snapshot/stage/raw-evidence readiness reconstruction.

`require_final_readiness_inputs(...)` is the shared fail-closed helper for canonical consumers and sweep tooling.

## Command

```bash
cargo run -p ucf-ops -- readiness-residual-sweep --out ./out/readiness_residual_sweep.json
```

## v12 residual-free consumer finalization

v12 adds `residual-free-readiness-sweep` and `ResidualFreeReadinessConsumerAuthorityV1` to prove that canonical consumers no longer retain historical, implicit, or aggregate readiness reconstruction in canonical flows.
