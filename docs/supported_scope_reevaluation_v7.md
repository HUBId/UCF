# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` bleibt ein intermediäres Reevaluation-Artefakt.

## Current hierarchy (v16)

- Reevaluation darf den Scope nie direkt anwenden.
- `SupportedScopeExecutionV12` ist die autoritative Execution-Entscheidung.
- `supported-set-apply` akzeptiert in v16 nur Autorisierung aus aktuellem `SupportedScopeExecutionV12`.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-convergence-sweep --out ./out/governance_convergence_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v12 --out ./out/supported_scope_execute_v12.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```


Ab v17 bleibt Reevaluation weiterhin nur vorbereitend; ausführen darf ausschließlich `SupportedScopeExecutionV12` unter Governance-Stabilization-PASS.
