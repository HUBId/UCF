# Supported Scope Execution v17

`SupportedScopeExecutionV12` ist die aktuelle autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v17.

## Ausführungs-Hierarchie

- `SupportedRealSlotSetPolicyV2`: Policy-Absicht, nie direkt anwendbar.
- `SupportedScopeReevaluationV1`: Reevaluation unter aktueller Applied-Scope-Basis, nie direkt anwendbar.
- `SupportedScopeExecutionV3`–`V11`: Historie/Audit.
- `SupportedScopeExecutionV12`: einzige zulässige Autorisierung für `models supported-set-apply` in v17.

## Warum Governance Stabilization jetzt zwingend ist

Expansion ist nur zulässig, wenn die stabilisierte konvergierte kanonische Governance-Kette aktuell PASS + digest-aligned ist:

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

Zusätzlich gilt fail-closed:

- Kandidat muss vollständig scaffolded sein.
- Keine Adapter-/Translation-/Projection-Governanceabhängigkeit.
- Export/Bundle/Roundtrip/Continuity ohne Sonderpfad.
- Genau ein valider Kandidat, sonst `REAFFIRM_FREEZE`.

## Befehle

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-convergence-sweep --out ./out/governance_convergence_sweep.json
cargo run -p ucf-ops -- governance-stabilization-sweep --out ./out/governance_stabilization_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v12 --out ./out/supported_scope_execute_v12.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze-Reaffirmation ist ein gültiger Erfolg

`REAFFIRM_FREEZE` ist first-class success, wenn Governance-Stabilization, Kandidatenqualität oder Kontinuitätskette keine saubere Expansion erlauben.

## Wie man erkennt, ob Scope geändert wurde

- `out/supported_scope_execute_v12.json`
  - `execution_decision=EXECUTE_EXPAND_BY_ONE` + `chosen_candidate_slot` => Expansion autorisiert.
  - `execution_decision=REAFFIRM_FREEZE` => Freeze explizit bestätigt.
- `out/supported_set_apply.json`
  - `decision=EXPANDED` => Applied Scope geändert.
  - `decision=FROZEN` => Applied Scope unverändert.

> v18 update: `SupportedScopeExecutionV13` ersetzt v12 als aktuelle Apply-Autorität unter zusätzlicher `GovernanceFinalConsolidationSweepV1`-Pflicht.
