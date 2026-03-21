# Governance Residual Sweep v11

`governance-residual-sweep` is the final canonical residual-governance check for v11.

## What it proves

The sweep verifies that canonical governance consumers are anchored to final governance inputs:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`

It emits `FinalGovernanceResidualSweepV1` with digest prefixes, coverage count, residual path count, status, and `sweep_digest`.

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V11PrepGateHelper`

## Why residual reconstruction is blocked

Canonical flows must fail closed when final governance inputs are missing or contradictory, and may no longer reconstruct governance truth from legacy execution/reevaluation/policy/evidence/export internals as primary substrate.

Stable residual denial/mismatch categories:

- `FINAL_GOVERNANCE_INPUTS_REQUIRED`
- `APPLIED_SCOPE_REQUIRED`
- `CANONICAL_GOVERNANCE_ENTRY_REQUIRED`
- `RESIDUAL_GOVERNANCE_PATH_BLOCKED`
- `CONSUMER_SKIPPED_FINAL_GOVERNANCE_INPUTS`
- `CONSUMER_USED_RESIDUAL_GOVERNANCE_PATH`
- `GOVERNANCE_INPUT_SCOPE_MISMATCH`
- `GOVERNANCE_INPUT_ENTRY_MISMATCH`
- `RESIDUAL_GOVERNANCE_PATH_PRESENT`

## Command

```bash
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
```
