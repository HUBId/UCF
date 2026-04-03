# Governance Stabilization Sweep v17

`ucf-ops governance-stabilization-sweep` is the v17 stabilization proof that canonical governance consumers use only the converged canonical governance chain and no adapter/translation/projection path as authoritative governance truth.

## Canonical authority inputs

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

The command emits `GovernanceStabilizationSweepV1` with deterministic digest ordering and per-consumer status.

## Covered canonical consumers

- ActiveReviewSnapshot
- OperatorSignoff
- OperatorReviewPacket
- OperatorWorkflowChain
- ExportReadinessGuard (`operator_export_chain`)
- InteropConsistencyMatrix
- v17-prep helper (`v16_gate_report.json` path continuity check)

## Why adapters/translators/projections are blocked

Canonical flows must now fail closed if converged canonical governance inputs are missing, stale, or contradictory. Governance adapter/translation/projection paths are therefore treated as legacy residues and surfaced through explicit mismatch categories.

## Run command

```bash
cargo run -p ucf-ops -- governance-stabilization-sweep --out ./out/governance_stabilization_sweep.json
```


## Beziehung zu Supported-Scope-Execution v17

`GovernanceStabilizationSweepV1` ist in v17 ein Pflichtinput für `models supported-scope-execute-v12`. Ohne PASS + digest alignment muss die Scope-Execution `REAFFIRM_FREEZE` ausgeben.

> v18 update: `governance-final-consolidation-sweep` removes remaining governance-facade, alias-layer, and shadow-governance residues from canonical consumers and emits `GovernanceFinalConsolidationSweepV1`.

## v19 closure note

v19 builds on stabilization by enforcing closure: canonical consumers must use the final consolidated chain and cannot use governance wrapper or crosswalk paths as authority.

> v20 extends stabilization lineage with a seal step (`governance-seal-sweep`) that blocks residual governance shells/bridges/auxiliary views in canonical consumer paths.
