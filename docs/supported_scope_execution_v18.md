# Supported Scope Execution v18 (Final Consolidation Gated)

`SupportedScopeExecutionV13` bleibt die v18-Execution-Stufe.

## Hinweis zur Nachfolge in v19
Ab v19 ist `SupportedScopeExecutionV14` die neue Apply-Autorität und erweitert die v18-Kette um `GovernanceClosureSweepV1` als zusätzliche harte Voraussetzung vor jeder Expansion.

## Hierarchie
- **Policy** (`SupportedRealSlotSetPolicyV2`): bewertet Kandidatenlage, aber führt keine Änderung aus.
- **Reevaluation** (`SupportedScopeReevaluationV1`): vorbereitende Plausibilisierung.
- **Prior execution artifacts** (z. B. v12): Historie/Chain-Link, aber **nicht** mehr direkte Apply-Autorität.
- **Current execution** (`SupportedScopeExecutionV13`): einzig gültige Autorisierung für `models supported-set-apply` in v18.

## Kommando (v18)
```bash
cargo run -p ucf-ops -- models supported-scope-execute-v13 --out ./out/supported_scope_execute_v13.json
```
