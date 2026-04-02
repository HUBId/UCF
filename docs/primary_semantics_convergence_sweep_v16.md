# Primary Semantics Convergence Sweep (v16)

`ucf-ops primary-semantics-convergence-sweep` emits `PrimarySemanticsConvergenceSweepV1` as a bounded proof that canonical surfaces consume primary blocking/remediation semantics only from the terminal canonical chain:

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`
6. `ResidualFreePrimarySemanticsConsumerAuthorityV1`
7. `ResidualFreePrimarySemanticsAbsoluteSweepV1`
8. `AbsoluteFinalPrimarySemanticsTerminalSweepV1`
9. `TerminalPrimarySemanticsUltimateSweepV1`

## What this sweep proves

- Covered canonical governance/readiness/bundle/review/export/interop/gate consumers no longer rely on primary-semantics memoization, precedence copies, or derived remediation mirrors as authoritative truth.
- Any canonical surface missing converged inputs is fail-closed via `ULTIMATE_TERMINAL_ABSOLUTE_PRIMARY_SEMANTICS_INPUTS_REQUIRED`.
- Memoized primary-semantics paths are treated as legacy and blocked/demoted with stable denial codes (`PRIMARY_SEMANTICS_MEMO_PATH_BLOCKED`, `PRIMARY_SEMANTICS_MEMO_PATH_TRANSLATED`, `PRIMARY_SEMANTICS_MEMO_PATH_REJECTED`).

## Covered surfaces

- `governance_ultimate_sweep`
- `readiness_ultimate_sweep`
- `bundle_ultimate_sweep`
- `operator_signoff`
- `operator_review_packet`
- `operator_workflow_chain`
- `interop_consistency_matrix`
- `v15_gate_report` (v16 prep helper)

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-convergence-sweep --out ./out/primary_semantics_convergence_sweep.json
```

The resulting artifact includes `convergence_status`, `residual_path_count`, and `convergence_digest` for deterministic auditability.

## v17 stabilization note

v17 adds `PrimarySemanticsStabilizationSweepV1` and requires canonical surfaces to consume primary blocking/remediation semantics only through the converged canonical primary-semantics chain. Residual adapter/translation/projection paths are no longer canonical authorities.

> v18 update: final consolidation now requires `PrimarySemanticsStabilizationSweepV1` plus `PrimarySemanticsFinalConsolidationSweepV1` for canonical surfaces that expose top-level primary blocking/remediation semantics.
