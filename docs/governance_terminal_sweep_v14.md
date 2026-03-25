# Governance Terminal Sweep v14

`ucf-ops governance-terminal-sweep` ist der terminale Nachweis, dass kanonische Governance-Consumer keine Decision-Echos, Scope-Lineage-Echos oder eingebetteten Governance-Summaries mehr als primäre Wahrheit verwenden.

## Was v14 beweist

Der Sweep ist **PASS** nur wenn alle abgedeckten Consumer zuerst diese autoritativen Inputs verwenden:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`

`AbsoluteFinalGovernanceTerminalSweepV1` fasst das Ergebnis kompakt zusammen (inkl. `sweep_digest`).

## Rolle für Supported Scope Execution v14

`SupportedScopeExecutionV9` darf nur expandieren, wenn `AbsoluteFinalGovernanceTerminalSweepV1` PASS und digest-aligned zu den anderen Governance-Inputs ist. Bei FAIL oder Legacy-Status wird fail-closed auf `REAFFIRM_FREEZE` entschieden.

## Abgedeckte Consumer

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V14PrepGateHelper`

## Kommando

```bash
cargo run -p ucf-ops -- governance-terminal-sweep --out ./out/governance_terminal_sweep.json
```
