# Primary Semantics Seal Sweep v20

`ucf-ops primary-semantics-seal-sweep` emits `PrimarySemanticsSealSweepV1`.

It proves that covered canonical governance/readiness/bundle/review/export/interop/gate surfaces derive authoritative primary blocking/remediation semantics exclusively from the closure-complete final-consolidated stabilized canonical primary-semantics chain:

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
12. `PrimarySemanticsFinalConsolidationSweepV1`
13. `PrimarySemanticsClosureSweepV1`

## What this sweep proves

- Compatibility-shell, bridge-layer, and auxiliary remediation-view primary-semantics paths are no longer authoritative in canonical flows.
- Covered surfaces are fail-closed when closure-complete canonical primary-semantics inputs are missing, stale, or contradictory.
- Remaining shell paths are classified using deterministic mismatch categories:
  - `SURFACE_SKIPPED_PRIMARY_SEMANTICS_SEAL`
  - `SURFACE_USED_PRIMARY_SEMANTICS_SHELL_PATH`
  - `PRIMARY_BLOCKING_ORDER_MISMATCH`
  - `PRIMARY_REMEDIATION_ORDER_MISMATCH`
  - `CANONICAL_CONDITION_MAPPING_MISMATCH`
  - `PRIMARY_SEMANTICS_SHELL_PATH_PRESENT`

## Covered surfaces

- `GovernanceSealSweep`
- `ReadinessSealSweep`
- `BundleSealSweep`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V20PrepGateHelper` (`out/v19_gate_report.json`)

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-seal-sweep --out ./out/primary_semantics_seal_sweep.json
```
