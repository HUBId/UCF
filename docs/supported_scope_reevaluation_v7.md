# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` remains an intermediate reevaluation artifact.

## Current hierarchy (v13)

- Reevaluation does not apply scope directly.
- `SupportedScopeExecutionV8` is the authoritative execution artifact.
- `supported-set-apply` accepts only current `SupportedScopeExecutionV8` authorization.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v8 --out ./out/supported_scope_execute_v8.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```
