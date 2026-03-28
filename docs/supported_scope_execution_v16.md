# Supported Scope Execution v16

`SupportedScopeExecutionV11` ist die aktuelle autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v16.

## Ausführungs-Hierarchie

- `SupportedRealSlotSetPolicyV2`: Policy-Absicht, nie direkt anwendbar.
- `SupportedScopeReevaluationV1`: aktuelle Reevaluation gegen Applied Scope, nie direkt anwendbar.
- `SupportedScopeExecutionV3`–`V10`: Historie/Audit.
- `SupportedScopeExecutionV11`: einzige zulässige Autorisierung für `models supported-set-apply` in v16.

## Warum Governance Convergence jetzt zwingend ist

Expansion darf nur ausgeführt werden, wenn die komplette konvergierte kanonische Governance-Kette **aktuell PASS und digest-aligned** ist:

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

Zusätzlich muss exakt ein Kandidat vollständig scaffolded sein und ohne Memoization-/Copy-/Mirror-Governancepfade durch Review/Export/Bundle/Continuity-Ketten tragbar sein.

## Befehle

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-convergence-sweep --out ./out/governance_convergence_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v11 --out ./out/supported_scope_execute_v11.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze-Reaffirmation ist ein gültiger Erfolg

`REAFFIRM_FREEZE` ist first-class success, wenn:

- Governance-Convergence nicht PASS/aligned ist, oder
- kein exakt ein valider Kandidat existiert, oder
- Export-/Continuity-Kompatibilität nicht sauber erfüllt ist.

## Wie man erkennt, ob Scope geändert wurde

- `out/supported_scope_execute_v11.json`
  - `execution_decision=EXECUTE_EXPAND_BY_ONE` + `chosen_candidate_slot` => Expansion autorisiert.
  - `execution_decision=REAFFIRM_FREEZE` => Freeze explizit bestätigt.
- `out/supported_set_apply.json`
  - `decision=EXPANDED` => Applied Scope geändert.
  - `decision=FROZEN` => Applied Scope unverändert.
