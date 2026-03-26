# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` bleibt ein intermediäres Reevaluation-Artefakt.

## Current hierarchy (v14)

- Reevaluation darf den Scope nie direkt anwenden.
- `SupportedScopeExecutionV9` ist die autoritative Execution-Entscheidung.
- `supported-set-apply` akzeptiert in v14 nur Autorisierung aus aktuellem `SupportedScopeExecutionV9`.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-terminal-sweep --out ./out/governance_terminal_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v9 --out ./out/supported_scope_execute_v9.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```
