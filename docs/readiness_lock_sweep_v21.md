# Readiness Lock Sweep v21

`ucf-ops readiness-lock-sweep` emits `ReadinessLockSweepV1` as a bounded proof that canonical readiness consumers derive readiness only from:

1. governance-locked applied scope chain (`GovernanceLockSweepV1` lineage),
2. authoritative supported-scope decision (`SupportedScopeExpansionDecisionV1`),
3. current canonical execution reality evidence digest.

## Covered consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `ExportReadinessGuard`
- `InteropConsistencyMatrix`
- `V21ReadinessGate`

## Why auxiliary readiness is rejected

Canonical readiness must not be inferred from handoff optimism, historical partial success, dormant adapters, implementation intent, or export-local narratives. Any such path is treated as mismatch/legacy and causes fail-closed status in canonical sweep output.

## Command

```bash
cargo run -p ucf-ops -- readiness-lock-sweep --out ./out/readiness_lock_sweep.json
```
