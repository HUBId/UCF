# Primary Semantics Ultimate Sweep v15

`ucf-ops primary-semantics-ultimate-sweep` is the v15 proof that canonical governance/readiness/bundle/review/export/interop/gate surfaces consume primary blocking/remediation semantics only from terminal absolute residual-free final primary-semantics inputs.

## What this proves

The sweep validates this authoritative chain in order:
1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`
5. `FinalPrimarySemanticsResidualSweepV1`
6. `ResidualFreePrimarySemanticsConsumerAuthorityV1`
7. `ResidualFreePrimarySemanticsAbsoluteSweepV1`
8. `AbsoluteFinalPrimarySemanticsTerminalSweepV1`

It emits `TerminalPrimarySemanticsUltimateSweepV1` with `sweep_digest` and per-surface statuses, proving canonical surfaces do not rely on primary-semantics echo caches, precedence mirrors, or embedded remediation snapshots as primary truth.

## Covered surfaces

- Governance ultimate sweep output
- Readiness ultimate sweep output
- Bundle ultimate sweep output
- Operator signoff
- Operator review packet
- Operator workflow chain
- Interop consistency matrix
- v15 prep gate helper (`v14_gate_report`)

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-ultimate-sweep --out ./out/primary_semantics_ultimate_sweep.json
```

## Status categories

Mismatch categories include:
- `SURFACE_SKIPPED_ULTIMATE_PRIMARY_SEMANTICS_INPUTS`
- `SURFACE_USED_PRIMARY_SEMANTICS_CACHE_PATH`
- `PRIMARY_BLOCKING_ORDER_MISMATCH`
- `PRIMARY_REMEDIATION_ORDER_MISMATCH`
- `CANONICAL_CONDITION_MAPPING_MISMATCH`
- `PRIMARY_SEMANTICS_CACHE_PATH_PRESENT`

Any missing/stale/contradictory terminal inputs fail closed.
