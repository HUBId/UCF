# Readiness Ultimate Sweep v15

`readiness-ultimate-sweep` ist der finale, deterministische Consumer-Sweep für kanonische Readiness-Flows.

## Was v15 beweist

`TerminalReadinessUltimateSweepV1` bindet ausschließlich diese autoritativen Inputs:

1. `SlotReviewabilityTruthV1`
2. `ReviewabilityReductionV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalReadinessAuthorityV2`
5. `FinalReadinessConsumerAuthorityV1`
6. `FinalReadinessResidualSweepV1`
7. `ResidualFreeReadinessConsumerAuthorityV1`
8. `ResidualFreeReadinessAbsoluteSweepV1`
9. `AbsoluteFinalReadinessTerminalSweepV1`

Der Sweep liefert einen kompakten Nachweis, dass im abgedeckten kanonischen Consumer-Set keine sekundäre Readiness-Wahrheit mehr aus Cache/Mirror/Snapshot rekonstruiert wird.

## Abgedeckte kanonische Consumer

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V15PrepGateHelper` (v14 Gate Report Oberfläche)

## Nicht mehr erlaubt in kanonischen Flows

- readiness echo caches
- stage mirrors
- embedded reviewability snapshots
- indirekte readiness reconstruction aus cacheartigen Zwischenständen

Verstöße werden deterministisch als mismatch categories ausgewiesen (z. B. `CONSUMER_USED_READINESS_CACHE_PATH`, `READINESS_CACHE_PATH_PRESENT`) und fail-closed behandelt.

## Command

```bash
cargo run -p ucf-ops -- readiness-ultimate-sweep --out ./out/readiness_ultimate_sweep.json
```


## v16 Update

v16 (`readiness-convergence-sweep`) entfernt die letzten memoized, kopierten oder abgeleiteten Readiness-Restpfade aus kanonischen Consumern.

v17 (`readiness-stabilization-sweep`) entfernt verbleibende Readiness-Adapter-/Translations-/indirekte Projektionspfade aus kanonischen Consumern und macht die Stabilisierung über `ReadinessStabilizationSweepV1` auditierbar.

> v18 update: `readiness-final-consolidation-sweep` entfernt verbleibende Readiness-Facade-/Alias-/Shadow-View-Residuen aus kanonischen Consumern und emittiert `ReadinessFinalConsolidationSweepV1`.

