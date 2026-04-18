# Serie K Abschluss: Compute-facing Integration in breitere UCF-Surfaces

Status: **technisch abgeschlossen (narrow integration closure)** auf Basis des aktuellen Repo-Stands.

## 1) Harte Abschlussmatrix (repo-basiert)

| Bereich | Zustand | Repo-Anker | Kurzbewertung |
|---|---|---|---|
| Compute execution contract (`submit/status/drain_scheduler`) | stable outward-facing integration surface | `runtime/ucf-compute/src/reference_map.rs` (`compute_execution_contract`), `runtime/ucf-compute/src/service_surface.rs` | Tragfähig als outward compute entry auf der kanonischen `result/fault/status`-Linie. |
| Status/diagnostics export (`status_evidence_export_surface` status-view) | stable outward-facing integration surface | `runtime/ucf-compute/src/service_surface.rs`, `docs/compute_status_evidence_export_surface_serie_k_v2.md` | Einheitliche Top-Level Status-/Trust-/Caveat-Semantik für angrenzende UCF-Subsysteme. |
| Evidence/reference export (`status_evidence_export_surface` evidence-view) | integration-usable but constrained | `runtime/ucf-compute/src/service_surface.rs`, `docs/compute_status_evidence_export_surface_serie_k_v2.md` | Referenz-/Metadatenfläche ist stabil; volle interne Diagnostik bleibt absichtlich nicht outward-facing. |
| Integration-safe hooks (`integration_hook_view`) | integration-usable but constrained | `runtime/ucf-compute/src/service_surface.rs`, `docs/compute_integration_safe_hooks_serie_k_v3.md` | Outward nur read-only (`read_only_integration_safe`, `caveated_conditional`); Mutationspfade bleiben non-outward. |
| Expert runtime control (`replay_with_entry`, `run_operation_with_entry`) | partial / internal-facing | `runtime/ucf-compute/src/reference_map.rs`, `runtime/ucf-compute/src/service_surface.rs` | Technisch wichtig, aber bewusst high-trust/internal und nicht generische Integrationsfläche. |
| Compatibility/legacy lanes (`build_backend kind=stub|candle`, `domains/ai*`, worker lane) | intentionally deferred | `runtime/ucf-compute/src/reference_map.rs` | Bewusst nicht load-bearing für outward compute integration; bleiben compatibility/internal boundary. |

## 2) Explizite Integrationsabschlusslinie

Serie K schließt mit folgender Linie:

1. **Outward-facing technisch tragfähig**
   - compute execution contract über `CanonicalComputeEntryPoint::{submit,status,drain_scheduler}`
   - canonical status/evidence export über `CanonicalComputeEntryPoint::status_evidence_export_surface()`
   - outward read-only hook Klassifikation über `integration_hook_view()`

2. **Bewusst expert/internal-only**
   - mutierende Expert-Pfade (`run_operation_with_entry(..., ExpertHighTrust)`, replay/runtime-control internals)
   - internal dev/test mutation path (`InternalClearReplayRegression`)

3. **Deferred/compatibility Punkte (nicht mehr load-bearing für Serie-K-Ziel)**
   - compatibility backend lane (`stub|candle`) als non-canonical outward boundary
   - legacy domain/worker compatibility lane (`domains/ai*`, worker/internal lane)

Alle outward-facing Serie-K-Flächen bleiben an der kanonischen Final-Linie gebunden:

`submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 3) Nächste Richtungen (1–3, technisch priorisiert)

1. **Serie L — narrow exit review / final hardening wrap-up**
   - Ziel: letzte enge Konsistenzhärtung der outward surfaces (Contracts ↔ Export ↔ Hook-Klassifikation) inkl. drift-fester Checks.
2. **Serie N — cross-system operational integration checks**
   - Ziel: gezielte Betriebs-/Ops-Integrationstests über angrenzende UCF-Surfaces ohne Semantikaufweitung.
3. **Serie M — targeted domain integration after compute-core completion**
   - Ziel: domänenspezifische Integration erst nach abgeschlossener exit-Härtung.

## 4) Priorisierte nächste Richtung

**Priorität: Serie L zuerst.**

Kurzbegründung:
- Höchster Hebel direkt nach Serie K, weil vorhandene outward contracts bereits stehen und jetzt mit schmaler Exit-Härtung gegen spätere Drift abgesichert werden können.
- Serie N ist wertvoll, aber nachrangig, da sie auf stabilen exit-definierten Flächen aufbauen sollte.
- Serie M ist am stärksten spezialisierend und deshalb erst nach Abschluss der systemweiten Exit-/Ops-Konsolidierung sinnvoll.
