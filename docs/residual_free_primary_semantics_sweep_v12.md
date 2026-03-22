# Residual-Free Primary Semantics Sweep v12

## What v12 proves

`residual-free-primary-semantics-sweep` is the final consumer sweep for canonical primary semantics.
It proves that canonical surfaces consume primary blocking/remediation semantics from the residual-free
final input chain only:

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`

The sweep emits `ResidualFreePrimarySemanticsConsumerAuthorityV1` as a bounded proof artifact.

## Covered canonical surfaces

The v12 sweep covers canonical governance/readiness/bundle/review/export/interop/gate consumer
surfaces represented in the final primary semantics consumer status set.

## Why historical/local reconstruction is no longer allowed

Historical main-reason precedence, implicit action-hint precedence, and ad-hoc condition/remediation
reconstruction can diverge across surfaces and reintroduce hidden semantics drift.

v12 therefore fails closed when residual-free final primary semantics inputs are missing, stale,
or contradictory.

## Command

```bash
cargo run -p ucf-ops -- residual-free-primary-semantics-sweep --out ./out/residual_free_primary_semantics_sweep.json
```

The command returns non-zero when residual/historical primary semantics paths are still present.
