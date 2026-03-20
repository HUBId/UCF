# Supported Scope Reevaluation v7

## Why reevaluation exists

`SupportedRealSlotSetPolicyV2` is a historical review artifact.
`SupportedRealSlotSetV2` / `AppliedSupportedSetContextV1` are canonical applied-scope artifacts.

v7 introduces `SupportedScopeReevaluationV1` so expansion is re-validated against **current** applied-scope-authoritative governance/export/interop state before any apply step.

## Decision model

`SupportedScopeReevaluationV1` emits exactly one of:

- `REAFFIRM_FREEZE` (default fail-closed)
- `EXECUTE_EXPAND_BY_ONE` (only with exactly one currently valid candidate)

`REAFFIRM_FREEZE` is expected and valid when policy is stale, scaffold is incomplete, authority checks are not clean, or candidates are ambiguous.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

Reevaluation is now an intermediate input. `supported-set-apply` no longer executes reevaluation artifacts directly; it requires `SupportedScopeExecutionV5`, which is bound to canonical governance + final governance-consumer authority. If reevaluation is stale, execution/apply regenerate the current chain deterministically.

## Scope and safety

- No backend/hardware scope is introduced.
- No activation/runtime semantics are introduced.
- Expansion remains capped to one slot and governance-only.
