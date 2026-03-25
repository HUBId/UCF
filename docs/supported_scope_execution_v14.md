# Supported Scope Execution v14

`SupportedScopeExecutionV9` ist die aktuelle, autoritative Ausführungsentscheidung für Supported-Scope-Änderungen nach terminaler Governance-Echo-Eliminierung.

## Hierarchie (v14)

- `SupportedRealSlotSetPolicyV2`: Policy-Empfehlung, **nicht** direkt anwendbar.
- `SupportedScopeReevaluationV1`: aktuelle Reevaluation, **nicht** direkt anwendbar.
- `SupportedScopeExecutionV3`-`V8`: Historie/Audit.
- `SupportedScopeExecutionV9`: einzige zulässige Autorisierung für `supported-set-apply` in v14.

## Warum v14 zusätzlich notwendig ist

Ab v14 reicht Residual-Free-Governance allein nicht mehr. Eine Expansion darf nur ausgeführt werden, wenn die **terminalen absoluten residual-freien finalen Governance-Inputs** aktuell PASS sind und sauber zum aktuellen Applied Scope passen:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`

Zusätzlich muss der Kandidat vollständig scaffolded sein und durch aktuelle Export/Bundle/Continuity-Ketten ohne Sonderpfade tragbar bleiben.

## Kommandos

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-terminal-sweep --out ./out/governance_terminal_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v9 --out ./out/supported_scope_execute_v9.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze-Reaffirmation ist Erfolg

`REAFFIRM_FREEZE` bleibt ein gewünschter, erfolgreicher Endzustand, sobald:

- kein exakt ein valider Kandidat vorhanden ist,
- ein terminaler Governance-Input nicht PASS/aligned ist,
- Echo/Lineage-Abhängigkeiten bestehen,
- oder Export-/Continuity-Ketten Lücken zeigen.

## Wie man Scope-Änderungen erkennt

- `out/supported_scope_execute_v9.json`
  - `execution_decision=EXECUTE_EXPAND_BY_ONE` + `chosen_candidate_slot` => Expansion autorisiert.
  - `execution_decision=REAFFIRM_FREEZE` => keine Expansion autorisiert.
- `out/supported_set_apply.json`
  - `decision=EXPANDED` => Applied Scope wurde geändert.
  - `decision=FROZEN` => Applied Scope unverändert.
