# Serie BB20 Prompt 1: Narrow production-readiness sweep across operational BlueBrain lines

Status: **BB20 etabliert eine schmale, repo-basierte production-readiness Referenz** über die bereits operativen Linien. Kein neuer Plattform- oder Feature-Scope; nur Konsistenz, Guard-Rail-Schärfung und ehrliche Statusmarkierung.

## 1) Operative Linien (hart inventarisiert)

| Linie | Readiness-Klasse | Repo-basierte Einordnung |
|---|---|---|
| BB2 Runtime-/Transition-/Feedback-Semantik | **stable operational** | Kanonische Runtime-Status-/Feedback-Linien sind produktiv gebunden und bleiben von internal-only Pfaden getrennt. |
| BB4 Selection-/Priority-/Deferral-Semantik | **usable with caveats** | Priority bleibt advisory-only; `deferred` bleibt getrennt von `blocked`; keine direkte Ausführungsmacht. |
| BB3/BB8 + BB17 Context/Memory/Reference | **stable operational** | Kanonische Referenztypen, Konsumregeln und Persistence-Boundaries sind stabil; non-canonical bleibt ausgeschlossen. |
| BB11/BB12 bounded Kuramoto dynamics | **advisory-only** | Dynamics liefern bounded Hinweise/Diagnostik ohne direkte Execution-/Selection-Autorität. |
| BB7/BB9 minimale Planning-/Action-Boundaries | **candidate-only** | Plan-/Action-Nähe bleibt diagnostics/readiness-basiert; keine Planner-/Agentenplattform. |
| BB13 minimale echte Execution-Linie | **stable operational** | Schmale operative Execution (`allowed canonical action` → `emit canonical signal`) bleibt die einzige reale Action-Linie. |
| BB14 execution-integrity | **stable operational** | Terminal-/Result-/Reference-Integrität bleibt fail-closed und nicht vermischt. |
| BB15 bounded retrieval/reference | **usable with caveats** | Retrieval/Reference ist operativ bounded; Consolidation bleibt candidate-only/advisory-only. |
| BB16 bounded dynamics ↔ execution | **advisory-only** | Execution-informierte Dynamics sind zulässig, bleiben aber strikt nicht-autoritativ. |
| BB19 runtime/selection contract | **usable with caveats** | Contract-Signale sind explizit getrennt (`advisory/deferred/blocked`), aber bewusst bounded. |
| internal/expert/dev/test Pfade | **non-canonical/internal-only** | Kein Default-Authority-Pfad, keine implizite Hochstufung. |
| Retry-/Queue-/Planner-/Policy-Orchestrierung | **test-only/deferred** | Nicht Teil der operativen BlueBrain-Produktionslinie. |

## 2) Repo-weite operational readiness map (kanonisch)

Die BB20-Map verwendet ausschließlich folgende Klassen:

1. `stable operational`
2. `usable with caveats`
3. `advisory-only`
4. `candidate-only`
5. `test-only/deferred`
6. `non-canonical/internal-only`

Diese Klassen sind technisch (kein Marketing) und dienen als gemeinsame Sprache für Runtime, Selection, Memory/Reference, Dynamics, Execution und Retrieval.

## 3) Begriffs- und Zustandsgrenzen (line-übergreifend)

### Verbindliche Trennung
- `blocked` ≠ `deferred` (blocked ist Guard-/Integritäts-/Verfügbarkeitsgrenze; deferred ist bounded Aufschub).
- `caveated` ≠ `insufficient` (caveated = verwendbar mit Caveats; insufficient = Basis nicht ausreichend).
- `unavailable` ≠ `blocked` (unavailable ist Verfügbarkeitszustand; blocked ist Regel-/Integritätsgrenze).
- `advisory-only` ≠ `candidate-only` (advisory-only liefert Hinweise; candidate-only bleibt vorgeschlagen, aber nicht operativ wirksam).
- `non-canonical/internal-only` bleibt immer außerhalb operativer Authority.

### BB20-Konsolidierungsregel
Jede Linie muss mindestens eine der sechs BB20-Readiness-Klassen tragen; implizite Zwischenzustände ohne Klasse sind zu vermeiden.

## 4) Referenz-, Guard- und Integritätspfad (quer)

### Canonical references
- Canonical reference paths bleiben alleinige outward authority.
- Internal-/expert-/compat-Pfade bleiben sichtbar, aber als non-canonical klassifiziert.

### no-direct-* guard rails (unverändert bindend)
- no direct action bypass
- no direct compute bypass
- no direct memory commit bypass
- no direct retry orchestration bypass

### execution-integrity (unverändert bindend)
- Terminalzustände bleiben getrennt (`completed/failed/cancelled/blocked/unavailable/unsupported/non-canonical`).
- Konflikt-/Duplikat-/Partial-Pfade bleiben fail-closed.

### dynamics und retrieval Grenzen
- bounded dynamics bleibt advisory-only.
- retrieval/reference bleibt bounded; keine implizite Consolidation-/Reasoning-/Ranking-Semantik.

## 5) Doku-Claims gegen Repo-Stand (BB20-Abgleich)

Im BB20-Sweep gilt:
- Operative Claims dürfen nur für `stable operational` oder klar markiert `usable with caveats` stehen.
- `advisory-only`, `candidate-only`, `test-only/deferred`, `non-canonical/internal-only` müssen explizit markiert bleiben.
- Keine zweite Wahrheitsquelle: diese Referenz konsolidiert nur bestehende Linien und ersetzt keine Code-Guards.

## 6) Schmale Härtungen in BB20

Durchgeführt in diesem Sweep:
- Vereinheitlichte Readiness-Klassen in einer zentralen BB20-Referenz.
- Explizite Begriffsgrenzen für `blocked/deferred/caveated/insufficient/unavailable` ergänzt.
- Production-facing Guard-Rails explizit als unverändert/bindend gesammelt.

Nicht durchgeführt (bewusst):
- keine neue Execution-Fläche,
- keine Planner-/Agenten-/Policy-Orchestrierung,
- keine neue Retrieval-/Reasoning-Plattform,
- keine neue Compute-Core-Arbeit.

## 7) Production-facing Guard-Rail Sicht (repo-weit)

Operational wirksam:
- runtime + memory/reference + minimal execution + execution-integrity.

Operational begrenzt:
- selection/runtime contract (`usable with caveats`),
- bounded retrieval/reference (`usable with caveats`).

Nicht operativ-autoritativ:
- bounded dynamics (`advisory-only`),
- planning/action diagnostics (`candidate-only`),
- test/deferred orchestration lanes,
- non-canonical/internal-only lanes.

## 8) Cross-line Prüfmatrix (für BB20)

Pflichtfragen bei Folgeänderungen:
1. Bleiben advisory/candidate/test/deferred/non-canonical klar unterscheidbar?
2. Bleiben no-direct-* Guards ohne Ausnahmen intakt?
3. Wird execution-integrity an keiner Stelle indirekt unterlaufen?
4. Erhält retrieval/reference keine neue Consolidation-/Reasoning-Autorität?
5. Widersprechen sich Code und Doku nicht?

## 9) BB20 production-readiness Referenz (kurz)

Diese Datei ist die schmale technische Referenz für:
- operative Linien,
- BB20 readiness map,
- unveränderte Guard-Rail-Grenzen,
- verbleibende Caveats,
- priorisierte nächste Richtung.

## 10) Genau eine priorisierte nächste Richtung

**Priorität 1: gezielter Cross-line cleanup pass für Runtime/Selection ↔ Retrieval/Reference Terminologiebindung.**

Statusupdate (BB20 Prompt 2): umgesetzt in `docs/blue_brain_bb20_cross_line_terminology_state_harmonization_serie_bb20_prompt2_v1.md`.

Technische Begründung (kurz):
- Die stärksten Restfriktionen liegen nicht in fehlender Kernfunktion, sondern in semantischer Drift zwischen `usable with caveats` Contract-Signalen und bounded retrieval/reference Diagnostik.
- Ein schmaler Cleanup-Pass liefert den höchsten Hebel für Produktionsnähe ohne Scope-Ausweitung.
