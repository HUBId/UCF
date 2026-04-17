# Real Compute Stack Abschlussmatrix (Serie I Prompt 4)

Stand: Repo-Zustand am 2026-04-17.

Ziel: harte technische Abschlusslinie für **Narrow final cleanup / canonical reference consolidation** (Serie I) und klare Priorisierung der nächsten technischen Serie.

## Canonical readiness authority (für diese Ausbaustufe)

Diese Datei ist die kanonische **Readiness-/Abschlussklassifikationsfläche** für den aktuellen Real-Compute-Stand.

Sie ist absichtlich gekoppelt mit:

- technischer Referenzfläche:
  - `docs/real_compute_reference_surface_v1.md`
  - `runtime/ucf-compute/src/reference_map.rs`
- Status-/Transition-Fläche:
  - `docs/roadmap/AI_MODEL_PIPELINE_STATUS.md`
  - `docs/roadmap/REAL_COMPUTE_TRANSITION.md`

Roadmap-Dateien (`docs/roadmap/AI_STACK.md`, `docs/roadmap/AI_BACKENDS.md`) bleiben Kontext und
müssen mit den oben genannten Flächen ausgerichtet bleiben.

## 1) Serie-I Kernprüfung (repo-basiert)

### 1.1 Canonical reference map

- **Real konsolidiert:**
  - `CANONICAL_COMPUTE_REFERENCE_MAP` klassifiziert produktive, expert-, diagnostics- und internal/legacy-Lanes explizit in einem code-pinned Ort.
  - Kanonischer produktiver Referenzpfad bleibt eindeutig (`service_entry` + canonical pipeline core + rollout activation core).
- **Constrained/partial:**
  - compatibility/dev- und internal-worker-Lanes bleiben vorhanden, aber explizit als nicht-produktive Referenzklassen markiert.
- **Bewusst verbleibender Rand:**
  - `stub|candle` und `worker` bleiben als technische Seams/Internals bestehen; sie sind dokumentiert, aber keine zweite Produktionswahrheit.

### 1.2 Shared-core terminology / contract invariants

- **Real konsolidiert:**
  - `request -> job -> run`, `action`, `result/fault/status` sind als shared core Begriffe in runtime docs und Referenzfläche deckungsgleich.
  - Action-Result-Semantik und diagnostics-core (`available|partial|unavailable`) bleiben als eine gemeinsame Vertragsbasis verankert.
- **Constrained/partial:**
  - Expert/diagnostic/internal Erweiterungen bleiben absichtlich als Erweiterungen über demselben Kern und tragen zusätzliche Caveats.
- **Bewusst verbleibender Rand:**
  - Erweiterte Betriebsdiagnostik bleibt technisch nutzbar, aber nicht als zweite Autoritätsquelle für den Kernvertrag.

### 1.3 Docs / status / readiness alignment

- **Real konsolidiert:**
  - Split ist klar: Reference (`real_compute_reference_surface_v1` + code map), Status/Transition (`AI_MODEL_PIPELINE_STATUS`, `REAL_COMPUTE_TRANSITION`), Readiness (diese Datei).
  - `AI_STACK` und `AI_BACKENDS` deklarieren die gleichen kanonischen Flächen als Autorität.
- **Constrained/partial:**
  - Roadmap-Files bleiben bewusst Kontextflächen; sie tragen keine tiefen Vertragsdetails.
- **Bewusst verbleibender Rand:**
  - Einzelne roadmap-nahe Formulierungen können künftig nachgezogen werden, solange keine Autoritätskollision entsteht.

## 2) Serie-I Abschlussmatrix (kurz)

| Bereich | Statusklasse | Repo-basierte Kurzbegründung |
|---|---|---|
| Canonical compute reference core (code map + canonical production lane) | **stable canonical reference core** | Eine code-pinned Referenzkarte klassifiziert alle relevanten Lanes; Produktionspfad bleibt eindeutig. |
| Shared-core terminology + contract invariants | **stable canonical reference core** | Kernbegriffe und Vertragsinvarianten sind quer über service/pipeline/reference konsistent und als Shared-Core verankert. |
| Docs/status/readiness surface split (reference vs status vs readiness) | **mostly aligned with minor caveats** | Autoritätsflächen sind klar definiert und in Stack/Backend-Roadmaps gespiegelt; Roadmap-Kontext bleibt absichtlich weniger detailtief. |
| Compatibility/dev/internal side lanes (`stub|candle`, `worker`, `domains/ai*`) | **partial / still split** | Lanes existieren weiter als Kompatibilitäts-/Internalschicht, aber explizit nicht als kanonische Produktionswahrheit. |
| Deep heterogeneous accelerator specialization / full fleet orchestrator | **intentionally deferred** | Nicht Teil dieser Cleanup-Serie; bleibt bewusst außerhalb des kanonischen Referenzkerns. |

## 3) Explizite Abschlusslinie für Serie I

Serie I ist als **Narrow final cleanup / canonical reference consolidation** abgeschlossen:

1. Der Real-Compute-Referenzkern ist jetzt eindeutig und code-pinned (keine konkurrierende Produktionsreferenz im Kern).
2. Shared-core Terminologie und Vertragsinvarianten sind über die tragenden Flächen konsolidiert.
3. Reference-/Status-/Readiness-Surfaces sind als Autoritäts-Split technisch sauber getrennt.

Restpunkte sind **nicht mehr load-bearing** für diese Cleanup-Serie:

- verbleibende compatibility/internal Seams als dokumentierte Nebenpfade,
- tiefe Accelerator-/Fleet-Orchestrierungsthemen außerhalb des Cleanup-Scope,
- roadmap-kontextuelle Formulierungsfeinschliffe ohne Vertragswirkung.

Weitere Arbeit wird daher nicht mehr als Cleanup-Serie geführt, sondern als **Konvergenz-/Integrationsserie auf dem stabilen Kern**.

## 4) Nächste Serien nach Serie I (Top-Hebel)

1. **Serie J — Final production-readiness convergence**
   - Fokus: Repro/Replay-/Promotion-Entscheidungssicherheit auf dem kanonischen Pfad und fail-closed Kanten für produktive Readiness.
2. **Serie K — Compute-facing integration into broader system surfaces**
   - Fokus: saubere, nicht-duplizierende Anbindung der Compute-Semantik in angrenzende Systemoberflächen (Status/Ops/Consumer-Interfaces).
3. **Serie L — Narrow exit review / final hardening wrap-up**
   - Fokus: begrenzte Endhärtung und Exit-Review nach J/K ohne neue Architekturflanken.

## 5) Exakt priorisierte nächste Serie

**Priorität jetzt: Serie J — Final production-readiness convergence.**

Warum höchster Hebel jetzt:

- baut direkt auf dem konsolidierten Referenzkern auf,
- reduziert die wichtigste Restunsicherheit zwischen „konsolidiert“ und „robust produktionsreif“,
- stärkt sofort die Tragfähigkeit von Promotion-/Replay-/Readiness-Entscheidungen.

Warum K/L nachrangig:

- K lohnt maximal, wenn Readiness-Konvergenz bereits hart sitzt,
- L ist ein Abschluss-/Härtungsschritt und sollte auf J (und ggf. K) folgen.

## 6) Minimale Konsistenzchecks für diese Abschlussaussage

- `CANONICAL_COMPUTE_REFERENCE_MAP` bleibt mit genau einer kanonischen `service_entry`-Produktionslane verankert.
- `stub|candle` bleiben als non-production compatibility lane klassifiziert.
- Onboarding-Referenz bleibt auf Burn gepinnt (`burn`, `burn_toy_v1`).
- Stack/Backend-Roadmaps referenzieren weiter denselben Autoritäts-Split (reference/status/readiness) statt konkurrierender Vertragsdefinition.
