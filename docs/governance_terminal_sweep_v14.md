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

## Abgedeckte Consumer

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V14PrepGateHelper`

## Warum Echo/Summary/Lineage nicht mehr zulässig sind

Kanonische Flows müssen Governance-Truth fail-closed aus finalen absoluten Inputs auflösen. Historische Rekonstruktion über Echos, Summaries, Lineage-Spuren oder eingebettete Hinweise ist kein autoritativer Einstieg mehr.

## Kommando

```bash
cargo run -p ucf-ops -- governance-terminal-sweep --out ./out/governance_terminal_sweep.json
```
