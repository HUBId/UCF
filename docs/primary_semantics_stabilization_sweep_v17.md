# Primary Semantics Stabilization Sweep v17

`PrimarySemanticsStabilizationSweepV1` is the bounded v17 proof artifact that canonical surfaces derive primary blocking/remediation semantics only from the converged canonical chain:

- canonical condition model
- canonical remediation registry
- `CanonicalPrimarySemanticsAuthorityV1`
- `FinalPrimarySemanticsConsumerAuthorityV1`
- `FinalPrimarySemanticsResidualSweepV1`
- `ResidualFreePrimarySemanticsConsumerAuthorityV1`
- `ResidualFreePrimarySemanticsAbsoluteSweepV1`
- `AbsoluteFinalPrimarySemanticsTerminalSweepV1`
- `TerminalPrimarySemanticsUltimateSweepV1`
- `PrimarySemanticsConvergenceSweepV1`

## Covered canonical surfaces

The sweep inspects canonical governance/readiness/bundle/review/export/interop/gate surfaces:

- governance/readiness/bundle ultimate outputs
- operator signoff/review/workflow outputs
- export normalize/roundtrip outputs
- interop consistency matrix output
- v17 prep gate helper (`v16_gate_report`)

## What this proves

- Canonical flows do not rely on primary-semantics adapters as authoritative truth.
- Translation/projection remnants are treated as residual adapter paths and force non-PASS outcomes.
- Missing/stale/contradictory converged-chain inputs fail closed via
  `CONVERGED_CANONICAL_PRIMARY_SEMANTICS_INPUTS_REQUIRED`,
  `CANONICAL_CONDITION_MODEL_REQUIRED`,
  `CANONICAL_REMEDIATION_REGISTRY_REQUIRED`, and
  `PRIMARY_SEMANTICS_ADAPTER_PATH_BLOCKED`.

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-stabilization-sweep --out ./out/primary_semantics_stabilization_sweep.json
```

> v18 update: `primary-semantics-final-consolidation-sweep` removes remaining canonical primary-semantics facade/alias/shadow residues from canonical consumer flows and emits `PrimarySemanticsFinalConsolidationSweepV1`.
