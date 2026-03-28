# Governance Convergence Sweep v16

`ucf-ops governance-convergence-sweep` is the v16 convergence proof that covered canonical governance consumers only consume governance truth from the terminal canonical chain:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`
9. `TerminalGovernanceUltimateSweepV1`

## What this sweep proves

- No covered canonical consumer may treat governance memoization, copied authority records, or derived governance mirrors as canonical governance truth.
- Canonical consumers fail closed if terminal canonical governance inputs are missing, stale, or contradictory.
- Residual memoized governance pathways are surfaced deterministically as explicit mismatch categories.

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V16PrepGateHelper` (`v15_gate_report.json`)

## Deterministic mismatch categories

- `CONSUMER_SKIPPED_GOVERNANCE_CONVERGENCE`
- `CONSUMER_USED_GOVERNANCE_MEMO_PATH`
- `GOVERNANCE_INPUT_SCOPE_MISMATCH`
- `GOVERNANCE_INPUT_ENTRY_MISMATCH`
- `GOVERNANCE_MEMO_PATH_PRESENT`

## Run command

```bash
cargo run -p ucf-ops -- governance-convergence-sweep --out ./out/governance_convergence_sweep.json
```

A PASS requires all covered consumers to align with the terminal canonical governance chain and report `convergence_status=PASS`.
