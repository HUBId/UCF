# Supported Scope Execution v9

## Policy vs Reevaluation vs Execution

- `SupportedRealSlotSetPolicyV2` stays a review recommendation artifact.
- `SupportedScopeReevaluationV1` re-checks policy intent against current applied scope.
- `SupportedScopeExecutionV4` is the **current authoritative execution artifact** that can reaffirm freeze or execute a one-slot expansion.

`SupportedScopeExecutionV4` is the only execution record that may authorize `models supported-set-apply` in v9.

## Why final governance authority is required

Execution v9 is fail-closed unless all of the following are clean for the **current** applied scope:

1. `CanonicalGovernanceEntryV1` PASS.
2. `CanonicalGovernanceEntryAuthorityV2` PASS.
3. Candidate slot still has complete scaffold and can pass export/review/bundle/interop continuity without secondary entry paths.

If any of these checks fail, execution must return `REAFFIRM_FREEZE` or deny stale chains.

## Command sequence

```bash
cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze reaffirmation is a successful outcome

`REAFFIRM_FREEZE` is expected success when:

- no exactly-one viable candidate exists,
- canonical governance entry/authority are not PASS,
- secondary governance entry dependency remains,
- export or bundle continuity gaps exist,
- or candidate is already in applied scope.

This is governance protection, not partial activation.
