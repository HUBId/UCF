# Readiness Convergence Sweep v16

`readiness-convergence-sweep` liefert den bounded Proof `ReadinessConvergenceSweepV1`, dass kanonische Readiness-Consumer ihre autoritative Sicht ausschließlich aus der finalen Kette beziehen:

1. `SlotReviewabilityTruthV1`
2. `ReviewabilityReductionV1`
3. `CanonicalReadinessSpineV1`
4. `CanonicalReadinessAuthorityV2`
5. `FinalReadinessConsumerAuthorityV1`
6. `FinalReadinessResidualSweepV1`
7. `ResidualFreeReadinessConsumerAuthorityV1`
8. `ResidualFreeReadinessAbsoluteSweepV1`
9. `AbsoluteFinalReadinessTerminalSweepV1`
10. `TerminalReadinessUltimateSweepV1`

## Was der Sweep beweist

- Keine kanonischen Consumer verwenden Readiness-Memoization als Primärsubstrat.
- Keine Stage-Kopien, abgeleiteten Reviewability-Mirrors oder Aggregate-Memory-Pfade werden als Autorität akzeptiert.
- Fehlende, stale oder widersprüchliche Inputs failen geschlossen.

## Abgedeckte kanonische Consumer

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V16PrepGateHelper`

## Warum Memo-/Copy-/Mirror-Pfade nicht mehr erlaubt sind

Kanonische Flows müssen eine eindeutige Readiness-Authority besitzen. Memoized/copy/derived Pfade können semantische Divergenzen erzeugen und sind deshalb nur blockiert/abgewiesen oder explizit nicht-kanonisch.

## Command

```bash
cargo run -p ucf-ops -- readiness-convergence-sweep --out ./out/readiness_convergence_sweep.json
```

v17 (`readiness-stabilization-sweep`) entfernt verbleibende Readiness-Adapter-/Translations-/indirekte Projektionspfade aus kanonischen Consumern und macht die Stabilisierung über `ReadinessStabilizationSweepV1` auditierbar.

