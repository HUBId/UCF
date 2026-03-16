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
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

The apply step consumes reevaluation output; stale or missing reevaluation is rejected.

## Scope and safety

- No backend/hardware scope is introduced.
- No activation/runtime semantics are introduced.
- Expansion remains capped to one slot and governance-only.
