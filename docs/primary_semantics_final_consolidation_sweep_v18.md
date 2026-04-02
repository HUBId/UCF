# Primary Semantics Final Consolidation Sweep (v18)

`ucf-ops primary-semantics-final-consolidation-sweep` emits `PrimarySemanticsFinalConsolidationSweepV1` as a bounded proof that canonical primary blocking/remediation semantics are sourced only from the stabilized converged canonical primary-semantics chain.

## Chain required before canonical consumption

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`
6. `ResidualFreePrimarySemanticsConsumerAuthorityV1`
7. `ResidualFreePrimarySemanticsAbsoluteSweepV1`
8. `AbsoluteFinalPrimarySemanticsTerminalSweepV1`
9. `TerminalPrimarySemanticsUltimateSweepV1`
10. `PrimarySemanticsConvergenceSweepV1`
11. `PrimarySemanticsStabilizationSweepV1`

## What v18 proves

`PrimarySemanticsFinalConsolidationSweepV1` binds:

- canonical governance/readiness/bundle digest prefixes
- canonical and final primary-semantics authority digest prefixes
- residual/residual-free/absolute/terminal/ultimate/convergence/stabilization digest prefixes
- covered surface count
- residual path count
- final consolidation status (`PASS | FAIL | LEGACY_PRESENT`)
- deterministic `consolidation_digest`

This closes remaining canonical dependencies on primary-semantics facades, alias layers, or shadow remediation views by requiring the stabilized chain as the sole authoritative input source.

## Covered canonical surfaces

- governance final consolidation sweep output
- readiness final consolidation sweep output
- bundle final consolidation sweep output
- operator signoff / review packet / workflow chain outputs
- interop consistency matrix output
- v18 prep helper (`out/v17_gate_report.json`)

## Stable mismatch categories

- `SURFACE_SKIPPED_PRIMARY_SEMANTICS_FINAL_CONSOLIDATION`
- `SURFACE_USED_PRIMARY_SEMANTICS_FACADE_PATH`
- `PRIMARY_BLOCKING_ORDER_MISMATCH`
- `PRIMARY_REMEDIATION_ORDER_MISMATCH`
- `CANONICAL_CONDITION_MAPPING_MISMATCH`
- `PRIMARY_SEMANTICS_FACADE_PATH_PRESENT`

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-final-consolidation-sweep --out ./out/primary_semantics_final_consolidation_sweep.json
```
