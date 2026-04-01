# Supported Scope Execution v18 (Final Consolidation Gated)

`SupportedScopeExecutionV13` ist die aktuelle autoritative Ausführungsentscheidung für Supported-Scope-Änderungen in v18.

## Hierarchie
- **Policy** (`SupportedRealSlotSetPolicyV2`): bewertet Kandidatenlage, aber führt keine Änderung aus.
- **Reevaluation** (`SupportedScopeReevaluationV1`): vorbereitende Plausibilisierung.
- **Prior execution artifacts** (z. B. v12): Historie/Chain-Link, aber **nicht** mehr direkte Apply-Autorität.
- **Current execution** (`SupportedScopeExecutionV13`): einzig gültige Autorisierung für `models supported-set-apply` in v18.

## Warum Governance Final Consolidation Pflicht ist
Expansion darf nur erfolgen, wenn die final-konsolidierte, stabilisierte Governance-Kette PASS ist (inkl. `GovernanceFinalConsolidationSweepV1`) und der Kandidat ohne Facade-/Alias-/Shadow-View-Governancepfade durch kanonische Consumer- und Export-/Continuity-Ketten passt.

## Kommando
```bash
cargo run -p ucf-ops -- models supported-scope-execute-v13 --out ./out/supported_scope_execute_v13.json
```

## Freeze-Reaffirmation ist Erfolg
`REAFFIRM_FREEZE` ist ein gültiges, gewünschtes Ergebnis, wenn die aktuelle Governance-Lage oder Kandidatenlage keine saubere Ein-Slot-Expansion erlaubt.

## Scope-Änderung erkennen
1. In `supported_scope_execute_v13.json` prüfen:
   - `execution_decision`
   - `chosen_candidate_slot`
2. Danach `models supported-set-apply` ausführen und `resulting_slots` vs `previous_slots` vergleichen.
