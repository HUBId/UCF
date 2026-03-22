# Residual-Free Governance Sweep v12

## Purpose

`residual-free-governance-sweep` proves that canonical governance consumers only trust residual-free final governance inputs:

- `AppliedSupportedSetContextV1`
- `CanonicalGovernanceEntryV1`
- `CanonicalGovernanceEntryAuthorityV2`
- `FinalGovernanceConsumerAuthorityV1`
- `FinalGovernanceResidualSweepV1`

The command emits `ResidualFreeGovernanceConsumerAuthorityV1` plus per-consumer status and fails closed when required inputs are missing, stale, or contradictory.

`SupportedScopeExecutionV7` consumes this authority directly; expansion is denied unless residual-free governance authority is `PASS` and digest prefixes match current applied/canonical/final governance inputs.

## Covered canonical consumers

- ActiveReviewSnapshot
- OperatorSignoff
- OperatorReviewPacket
- OperatorWorkflowChain
- InteropConsistencyMatrix
- v12 prep helper (`v11_gate_report.json` probe)

## Why historical/implicit reconstruction is blocked

Canonical flows must not rebuild governance truth from execution metadata, reevaluation/policy history, or raw historical hints. Any mismatch is classified and surfaced as deterministic failures (`FAIL`) or legacy presence (`LEGACY_PRESENT`).

## Command

```bash
cargo run -p ucf-ops -- residual-free-governance-sweep --out ./out/residual_free_governance_sweep.json
```

## v13 follow-up

The v13 `governance-absolute-sweep` consumes this authority and removes the last canonical historical/embedded governance lineage traces from covered consumers.

