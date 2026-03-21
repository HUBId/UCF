# Supported Scope Reevaluation v7

`SupportedScopeReevaluationV1` remains an intermediate reevaluation artifact (policy intent rechecked against current applied scope).

## Execution hierarchy in v11

- Reevaluation does **not** apply scope directly.
- `SupportedScopeExecutionV6` is the authoritative execution artifact.
- `supported-set-apply` accepts only current `SupportedScopeExecutionV6` authorization.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- models supported-scope-execute-v6 --out ./out/supported_scope_execute_v6.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```
