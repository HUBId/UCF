# Serie O: Compute Core Maintenance-Only Boundary v1

Stand: Repo-Zustand am 2026-04-19.

Ziel: den finalisierten Real-Compute-Kern explizit als **maintenance-only** Bereich markieren, ohne neue Governance-/Release-/Policy-Struktur.

Diese Boundary bleibt auf derselben technischen Linie aus Serie J/L/M/N:
- finale technical production line: `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- final reference line: `docs/final_reference_line_serie_j_v1.md`
- outward-facing integration line: `status_evidence_export_surface` + `integration_hook_view` (read-only/caveated)
- erste Post-Core-Integrationslinie: `docs/compute_consumer_integration_map_serie_m_v1.md`
- breitere Review-Linie: `docs/broader_system_integration_map_serie_n_v1.md`

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_FINAL_REFERENCE_LINE`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`

## 1) Abgeschlossene Flächen (maintenance-only geschützt)

Als abgeschlossen gelten und bleiben unverändert maßgeblich:

1. Canonical execution core line (`submit -> compute_canonical -> result/fault/status -> execution_snapshot`).
2. Cross-cutting Kerninvarianten + handoff semantics auf derselben Kernsprache.
3. Outward status/evidence export + integration-safe hook boundary als Adapter, nicht als zweite Kernlogik.
4. Boundary, dass compatibility/internal/legacy lanes keine outward Produktionsautorität werden.

## 2) Maintenance-only boundary view (minimal)

Nur drei Klassen:

- `maintenance_safe_change`
- `maintenance_safe_with_care`
- `not_maintenance_only_requires_new_integration_or_buildout`

## 3) Zulässige maintenance-only Änderungsarten

### `maintenance_safe_change`

1. **bug fix**
   - eng begrenzte Fehlerkorrektur innerhalb bestehender Kernsemantik, ohne neue Laufzeitfähigkeit.
2. **small contract consistency fix**
   - kleine Konsistenzkorrektur an bestehender status/evidence/contract Sprache.
3. **narrow drift correction**
   - Driftbehebung zwischen Code und finaler Referenz-/Exit-/Integrationsdoku.
4. **doc/readiness/reference alignment**
   - Doku-/Readiness-Abgleich auf bestehende Source-of-Truth-Konstanten.
5. **small guard/check hardening**
   - kleine fail-closed Guard-/Check-Härtung auf bereits bestehenden Pfaden.

### `maintenance_safe_with_care`

Änderungen sind nur dann in Serie O zulässig, wenn sie strikt schmal bleiben:
- Korrekturen an load-bearing Integrationskanten (z. B. orchestrator intake),
- keine neue Semantikschicht,
- keine neuen outward contracts,
- kein Umbau der finalen Kernlinie.

## 4) Explizit außerhalb maintenance-only

Folgende Klassen sind **nicht** Serie O:

1. `new runtime feature`
2. `broader new integration`
3. `new backend/device capability expansion`
4. `new workflow/control surface`
5. `architectural reshaping`

Wenn eine Änderung in eine dieser Klassen fällt, ist sie neue Integrations- oder Ausbauarbeit und nicht maintenance-only.

## 5) Sichtbare Spiegelung in Referenz-/Exit-/Integrationslinie

- Referenzlinie (Serie J) bleibt die technische Kernautorität.
- Exit-Dossier (Serie L) bleibt die Abschlussaussage, dass Core-Abschluss erreicht ist.
- Serie-M/N-Dokumente bleiben Integrationssicht; Serie O verhindert nur Re-Opening des Kerns.
- Diese Datei erzeugt **keine zweite Wahrheitsquelle**, sondern benennt explizit die zulässige Änderungsbreite nach dem Abschluss.

## 6) Minimaler Konsistenzcheck

Für Serie O genügt ein kleiner Check:
- Boundary-Klassen bleiben exakt dreiteilig.
- Alle maintenance-safe Änderungsarten bleiben explizit benannt.
- Out-of-scope Klassen bleiben explizit benannt.
- Referenz auf finale Kernlinie (`submit -> compute_canonical -> result/fault/status -> execution_snapshot`) bleibt enthalten.

## 7) Konkrete Within-vs-Outside Beispiele

Innerhalb (`maintenance_safe_change` / `maintenance_safe_with_care`):
- Null-Deref/Status-Mapping-Bug auf bestehendem submit/result/fault/status Pfad reparieren.
- Kleine Konsistenzkorrektur bei caveated/partial/degraded Benennung zwischen Code und Doku.
- Driftkorrektur, wenn Integrationsdoku eine nicht mehr bestehende Funktion referenziert.
- Kleine Guard-Härtung, die fehlende Evidence-Refs sauber fail-closed meldet.

Außerhalb (`not_maintenance_only_requires_new_integration_or_buildout`):
- Neuer Laufzeitmodus oder neues Pipeline-Feature.
- Neuer outward Runtime-Consumer mit eigenem Contract.
- Erweiterung auf neue Backend-/Device-Fähigkeiten.
- Neue Workflow-/Control-Surface für Experten oder Operator.
- Architekturumbau, der die Kernlinie oder ihre Autoritätsgrenzen neu ordnet.

## 8) Serie-O Folgearbeit (3–5 direkte nächste Schritte)

1. Kleine Drift-/Konsistenzkorrekturen auf der finalen Referenzlinie kontinuierlich sammeln und schmal halten.
2. Bei Änderungen an load-bearing Integrationskanten immer gegen die drei Boundary-Klassen prüfen.
3. `ops_compute_probe` als canonical Referenzanker unverändert klar halten.
4. Outward Contracts (`submit`, `status_evidence_export_surface`, `integration_hook_view`) nicht stillschweigend erweitern.
5. Nur dann aus Serie O aussteigen, wenn bewusst neue Integration/Buildout freigegeben wird.
