# Supported Scope Execution v13

`SupportedScopeExecutionV8` is the authoritative current execution artifact after governance-lineage elimination.

## Execution hierarchy

- `SupportedRealSlotSetPolicyV2`: policy recommendation only.
- `SupportedScopeReevaluationV1`: current reevaluation only.
- `SupportedScopeExecutionV4`-`V7`: historical continuity/audit only.
- `SupportedScopeExecutionV8`: **only** execution artifact allowed to authorize `supported-set-apply` in v13.

## Why governance-lineage elimination is required

Expansion is fail-closed unless all current governance inputs are PASS and aligned:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`

Older policy/reevaluation/previous execution artifacts never authorize expansion by themselves.

## Commands

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v8 --out ./out/supported_scope_execute_v8.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze reaffirmation is success

`REAFFIRM_FREEZE` is a first-class successful outcome whenever there is no exactly-one viable candidate, governance inputs are not fully PASS/aligned, or continuity/export chains cannot carry the candidate cleanly.

## How to detect scope change

- `out/supported_scope_execute_v8.json`
  - `execution_decision=EXECUTE_EXPAND_BY_ONE` + `chosen_candidate_slot` => one-slot authorization.
  - `execution_decision=REAFFIRM_FREEZE` => no expansion authorization.
- `out/supported_set_apply.json`
  - `decision=EXPANDED` => applied scope changed.
  - `decision=FROZEN` => applied scope unchanged.
