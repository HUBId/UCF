# Supported Scope Execution v15

`SupportedScopeExecutionV10` war die autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v15.
In v16 wurde diese Rolle von `SupportedScopeExecutionV11` übernommen.

## Ausführungs-Hierarchie

- `SupportedRealSlotSetPolicyV2`: Policy-Absicht, nie direkt anwendbar.
- `SupportedScopeReevaluationV1`: aktuelle Reevaluation gegen Applied Scope, nie direkt anwendbar.
- `SupportedScopeExecutionV3`–`V9`: Historie/Audit.
- `SupportedScopeExecutionV10`: v15-Ausführungsebene.
- `SupportedScopeExecutionV11`: aktuelle v16-Ausführungsebene mit zusätzlicher Governance-Convergence-Pflicht.

## Warum Governance-Cache-Eliminierung zwingend ist

Expansion ist nur erlaubt, wenn **aktuelle** finale residual-freie terminale Governance-Inputs PASS sind und digest-aligned bleiben:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`
9. `TerminalGovernanceUltimateSweepV1`

Zusätzlich muss genau ein Kandidat vollständig scaffolded sein und ohne historische/implizite/echo-/mirror-/cache-/lineage-Governancepfade durch Export/Bundle/Continuity-Ketten tragbar sein.

## Befehle

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-ultimate-sweep --out ./out/governance_ultimate_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v10 --out ./out/supported_scope_execute_v10.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze-Reaffirmation ist ein gültiger Erfolg

`REAFFIRM_FREEZE` ist first-class success, wenn kein exakt ein valider Kandidat existiert oder irgendein terminaler finaler Governance-Input nicht PASS/aligned ist.

## Wie man erkennt, ob Scope geändert wurde

- `out/supported_scope_execute_v10.json`
  - `execution_decision=EXECUTE_EXPAND_BY_ONE` + `chosen_candidate_slot` => Expansion autorisiert.
  - `execution_decision=REAFFIRM_FREEZE` => Freeze explizit bestätigt.
- `out/supported_set_apply.json`
  - `decision=EXPANDED` => Applied Scope geändert.
  - `decision=FROZEN` => Applied Scope unverändert.
