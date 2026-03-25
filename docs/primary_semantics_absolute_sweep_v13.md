# Primary Semantics Absolute Sweep v13

`primary-semantics-absolute-sweep` is the final v13 proof that canonical governance/readiness/bundle/review/export/interop/gate surfaces consume primary blocking/remediation semantics only from residual-free final primary-semantics inputs.

## Proven authority chain

The sweep resolves and validates, in order:

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`
6. `ResidualFreePrimarySemanticsConsumerAuthorityV1`

The run fails closed if one input is missing, stale, contradictory, or non-PASS.

## Covered canonical surfaces

- `governance_absolute_sweep`
- `readiness_absolute_sweep`
- `bundle_absolute_sweep`
- `operator_signoff`
- `operator_review_packet`
- `operator_workflow_chain`
- `interop_consistency_matrix`
- `v12_gate_report` (v13 prep helper)

## Why historical lineage is blocked

Canonical flows may no longer use historical priority tables, embedded action-hint memory, local cached precedence, or surface-specific override paths as primary semantics truth.

Any such path is treated as lineage and flagged through deterministic mismatch categories (including `SURFACE_USED_HISTORICAL_PRIMARY_SEMANTICS_LINEAGE` and `HISTORICAL_PRIMARY_SEMANTICS_LINEAGE_PRESENT`).

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-absolute-sweep --out ./out/primary_semantics_absolute_sweep.json
```
