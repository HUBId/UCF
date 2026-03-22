# Governance Absolute Sweep v13

`governance-absolute-sweep` is the final residual-free governance consumer proof for canonical governance/review/export/gate consumers.

## What this proves

`ResidualFreeGovernanceAbsoluteSweepV1` proves that covered canonical consumers derive governance context only from the residual-free final governance input chain:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`

The sweep is deterministic, bounded, and fail-closed. Missing/stale/contradictory inputs produce FAIL/LEGACY status.

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V13PrepGateHelper` (`v12_gate_report` helper surface)

## Why historical/embedded reconstruction is blocked

Canonical flows must not reconstruct governance truth from historical lineage, reevaluation history, embedded rationale summaries, raw evidence-first paths, or bundle-local governance notes.

v13 enforces direct residual-free final governance inputs as authoritative for consumer-level governance continuity.

## Command

```bash
cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json
```
