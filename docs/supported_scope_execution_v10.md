# Supported Scope Execution v10

## Artifact hierarchy (current)

- `SupportedRealSlotSetPolicyV2`: review recommendation artifact.
- `SupportedScopeReevaluationV1`: current-scope reevaluation artifact.
- `SupportedScopeExecutionV4`: prior execution artifact (kept for continuity).
- `SupportedScopeExecutionV5`: **current authoritative execution artifact** for freeze reaffirmation vs one-slot expansion.

In v10, `models supported-set-apply` may only apply scope changes authorized by `SupportedScopeExecutionV5`.

## Why final governance-consumer authority is mandatory

`SupportedScopeExecutionV5` requires all final governance inputs to be current PASS:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`

Expansion is fail-closed unless exactly one candidate is still fully scaffolded and continuity-safe for review/export/bundle/roundtrip flows without legacy governance inputs.

## Commands

```bash
cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json
```

## Freeze reaffirmation is valid success

`REAFFIRM_FREEZE` is a first-class success result when no exactly-one viable candidate remains, final governance-consumer authority is not PASS, continuity checks have gaps, or ambiguity exists.

This is governance-state control only:
- no automatic Active implications,
- no runtime-mode mutation,
- no speculative slot activation.
