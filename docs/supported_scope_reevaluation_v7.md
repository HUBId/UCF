# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` bleibt ein intermediäres, nicht-anwendendes Artefakt.

## Current hierarchy (v19)

- Reevaluation darf den Scope nie direkt anwenden.
- `SupportedScopeExecutionV14` ist die aktuelle autoritative Execution-Entscheidung.
- `supported-set-apply` akzeptiert in v19 nur Autorisierung aus aktuellem `SupportedScopeExecutionV14`.

## Commands (v19 execution path)

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-closure-sweep --out ./out/governance_closure_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v14 --out ./out/supported_scope_execute_v14.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

Reevaluation bleibt vorbereitend; ausführen/anwenden darf ausschließlich die aktuelle Execution-Stufe unter closure-kompletter Governance-PASS-Lage.
