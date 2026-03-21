# Supported Set Apply v6

`models supported-set-apply` now requires a current `SupportedScopeExecutionV6` execution artifact and writes canonical applied scope as `SupportedRealSlotSetV2`.

## v11 behavior

- `REAFFIRM_FREEZE` execution emits a fresh frozen applied artifact bound to current governance state.
- `EXECUTE_EXPAND_BY_ONE` may add exactly one slot.
- Stale policy/reevaluation/prior execution artifacts are denied in execution and do not apply directly.

## Command chain

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v6 --out ./out/supported_scope_execute_v6.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```
