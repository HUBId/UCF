# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` bleibt ein intermediäres Reevaluation-Artefakt.

## Current hierarchy (v15)

- Reevaluation darf den Scope nie direkt anwenden.
- `SupportedScopeExecutionV10` ist die autoritative Execution-Entscheidung.
- `supported-set-apply` akzeptiert in v15 nur Autorisierung aus aktuellem `SupportedScopeExecutionV10`.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-ultimate-sweep --out ./out/governance_ultimate_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v10 --out ./out/supported_scope_execute_v10.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```
