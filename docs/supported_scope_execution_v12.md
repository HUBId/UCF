# Supported Scope Execution v12

`SupportedScopeExecutionV7` is the authoritative **current execution** artifact for supported-scope changes after historical-governance cleanup.

## Execution hierarchy

- `SupportedRealSlotSetPolicyV2`: review/policy intent only.
- `SupportedScopeReevaluationV1`: reevaluation against current applied scope.
- `SupportedScopeExecutionV4`/`V5`/`V6`: historical execution artifacts kept only for continuity/audit.
- `SupportedScopeExecutionV7`: current authoritative execution decision.

Only `SupportedScopeExecutionV7` may authorize the next `models supported-set-apply` result in v12.

## Why residual-free final governance is now mandatory

Scope expansion in v12 is fail-closed unless all residual-free final governance inputs are current and PASS:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`

Older policy/reevaluation/prior-execution artifacts alone cannot authorize expansion.

## Command

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- models supported-scope-execute-v7 --out ./out/supported_scope_execute_v7.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze reaffirmation is a successful outcome

`REAFFIRM_FREEZE` is first-class and expected whenever residual-free governance inputs, candidate scaffolding, or continuity/export authority are not cleanly satisfied.

## How to tell whether scope changed

- Check `out/supported_scope_execute_v7.json`:
  - `execution_decision = EXECUTE_EXPAND_BY_ONE` with `chosen_candidate_slot` means one-slot expansion is authorized.
  - `execution_decision = REAFFIRM_FREEZE` means no scope expansion.
- Check `out/supported_set_apply.json`:
  - `decision = EXPANDED` means applied scope changed.
  - `decision = FROZEN` means applied scope is unchanged.
