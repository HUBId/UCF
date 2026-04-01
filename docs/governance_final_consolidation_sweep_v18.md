# Governance Final Consolidation Sweep (v18)

`ucf-ops governance-final-consolidation-sweep` ist der v18 Final-Consolidation-Beweis, dass kanonische Governance-Consumer ihre autoritative Governance-Sicht ausschließlich aus der stabilisierten konvergierten Governance-Kette ableiten.

## Was der Sweep beweist

`GovernanceFinalConsolidationSweepV1` bindet deterministisch:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`
9. `TerminalGovernanceUltimateSweepV1`
10. `GovernanceConvergenceSweepV1`
11. `GovernanceStabilizationSweepV1`

und validiert, dass keine Governance-Facade, Alias-Layer oder Shadow-View als konkurrierende Governance-Wahrheit in den abgedeckten kanonischen Flows verbleibt.

## Abgedeckte Consumer

- ActiveReviewSnapshot
- OperatorSignoff
- OperatorReviewPacket
- OperatorWorkflowChain
- ExportReadinessGuard
- BundleBuildVerifyOrchestration
- InteropConsistencyMatrix
- V18PrepGateHelper

## Warum Facades/Aliase/Shadow-Views nicht mehr erlaubt sind

Diese Pfade können eine konkurrierende Governance-Wahrheit erzeugen (z. B. Facade-Notizen, Alias-Mappings über Authority-Artefakten oder Shadow-Views) und würden den deterministischen Nachweis der kanonischen Governance-Kette brechen.

Der Sweep meldet deterministische Mismatch-Kategorien:

- `CONSUMER_SKIPPED_GOVERNANCE_FINAL_CONSOLIDATION`
- `CONSUMER_USED_GOVERNANCE_FACADE_PATH`
- `GOVERNANCE_INPUT_SCOPE_MISMATCH`
- `GOVERNANCE_INPUT_ENTRY_MISMATCH`
- `GOVERNANCE_FACADE_PATH_PRESENT`

## Command

```bash
cargo run -p ucf-ops -- governance-final-consolidation-sweep --out ./out/governance_final_consolidation_sweep.json
```

## Scope execution coupling
Ab v18 muss `models supported-scope-execute-v13` diesen Sweep im PASS-Zustand nachweisen; sonst ist nur `REAFFIRM_FREEZE` zulässig.
