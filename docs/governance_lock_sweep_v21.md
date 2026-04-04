# Governance Lock Sweep v21

`GovernanceLockSweepV1` proves that canonical governance consumers derive authoritative governance only from the seal-complete closure-complete final-consolidated stabilized canonical governance chain:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`
9. `TerminalGovernanceUltimateSweepV1`
10. `GovernanceConvergenceSweepV1`
11. `GovernanceStabilizationSweepV1`
12. `GovernanceFinalConsolidationSweepV1`
13. `GovernanceClosureSweepV1`
14. `GovernanceSealSweepV1`

## Covered canonical consumers

- ActiveReviewSnapshot
- OperatorSignoff
- OperatorReviewPacket
- OperatorWorkflowChain
- ExportReadinessGuard
- BundleBuildVerifyOrchestration
- InteropConsistencyMatrix
- V21PrepGateHelper

## Why compatibility frames/relay/aux projections are blocked

Canonical flows must fail closed if governance truth is reconstructed from compatibility frames, relay layers, or auxiliary governance projections. v21 marks remaining frame paths as blocked/rejected unless deterministic and lossless translation is explicitly possible.

## Command

```bash
cargo run -p ucf-ops -- governance-lock-sweep --out ./out/governance_lock_sweep.json
```

## v21 scope decision note

Governance lock alone does not widen supported scope. Scope changes require an explicit binary `supported-scope-decision` result (`SCOPE_EXPANSION_APPLIED` or `SCOPE_FREEZE_REINFORCED`).


> v21 update: readiness consumers must also pass `ReadinessLockSweepV1` and align with `SupportedScopeExpansionDecisionV1` plus canonical execution reality evidence.
