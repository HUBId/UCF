# Supported Set Apply v6

## Purpose

`models supported-set-apply` executes the governance decision only after a current `SupportedScopeReevaluationV1` result is present and writes the **applied** supported-slot scope as `SupportedRealSlotSetV2`.

This separates review from execution:

- review (`supported-set-review`) is advisory policy output,
- apply (`supported-set-apply`) is the authoritative applied scope artifact.

## FREEZE vs EXPANDED

- `FROZEN`: resulting set remains equal to the previous supported set. A `SupportedSetFreezeRecordV1` is emitted.
- `EXPANDED`: resulting set equals previous set plus exactly one slot, only when scaffolding checks still pass at execution time. A `SupportedSetExpansionRecordV1` is emitted.

If expansion preconditions fail during apply, execution is denied with stable denial codes and falls back to `FROZEN`.

## Command

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

Canonical applied-set artifact path:

- `./out/supported_real_slot_set_applied_v2.json`

## Important non-goals

Applying supported-set governance does **not**:

- activate slots,
- promote models,
- alter active runtime mode.

Newly expanded scope remains probe/shadow-governed and fail-closed.
