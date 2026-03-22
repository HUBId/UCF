# Supported Scope Execution v11

## Execution hierarchy (current, authoritative)

- `SupportedRealSlotSetPolicyV2`: review recommendation only.
- `SupportedScopeReevaluationV1`: reevaluation against current applied scope.
- `SupportedScopeExecutionV4` / `SupportedScopeExecutionV5`: prior execution artifacts kept for continuity.
- `SupportedScopeExecutionV6`: v11 authoritative execution artifact (retained for continuity only in v12).
- `SupportedScopeExecutionV7`: **current authoritative execution artifact in v12**.

In v12, only `SupportedScopeExecutionV7` may authorize the next `models supported-set-apply` result.

## Why v11 requires residual-governance cleanup

`SupportedScopeExecutionV7` is fail-closed unless all current residual-free governance inputs are present and PASS:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`

Older policy/reevaluation/execution outputs are not sufficient by themselves.

## Decision outcomes

- `REAFFIRM_FREEZE`: explicit success outcome and first-class state.
- `EXECUTE_EXPAND_BY_ONE`: allowed only when exactly one fully scaffolded candidate remains valid under current final governance inputs and residual-sweep cleanup.

No active/runtime implications are introduced by this execution step.

## Commands

```bash
cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v7 --out ./out/supported_scope_execute_v7.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## How to tell if applied scope changed

After `supported-set-apply`, compare:

- `previous_slots` vs `resulting_slots` in `./out/supported_set_apply.json`
- `decision` (`FROZEN` means unchanged, `EXPANDED` means exactly one added slot)

## v12 update

`SupportedScopeExecutionV11` documentation is superseded by v12 residual-free governance consumer requirements: canonical consumers may not derive governance from execution or reevaluation history.
