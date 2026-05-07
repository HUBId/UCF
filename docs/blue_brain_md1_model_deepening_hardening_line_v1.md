# Architekturpaket MD1 Prompt 3: model-deepening hardening line

Status: schmale Härtung der in MD1 Prompt 2 eingeführten ersten Modellvertiefung. Diese Linie konsolidiert genau `Amygdala ↔ Thalamus` als bounded Kuramoto-like relation-level advisory deepening gegen bestehende Region-/Relations-Surfaces, Diagnostics, Contracts und Guards. Sie öffnet keine zweite Modellvertiefung, keine globale Dynamikplattform, keine HH-Produktivlinie und keine neue Compute-/Planner-/Retry-/Memory-/Policy-Autorität.

Canonical code anchor: `CANONICAL_BLUE_BRAIN_MD1_FIRST_DEEPENING_HARDENING_MAP`, `BlueBrainMd1FirstDeepeningBoundaryState`, `BlueBrainMd1FirstDeepeningContractSupportClass`, and `evaluate_blue_brain_md1_first_model_deepening` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Canonical model-deepening hardening map

MD1 Prompt 3 verwendet nur diese Härtungsklassen:

| Hardening class | Canonical meaning | Boundary |
|---|---|---|
| `hardened deepened input surface` | Nur bestehende bounded Kuramoto-like Inputs aus Runtime, Selection, Context/Reference, Memory-caveat, Evidence und bounded Execution/Reference feedback dürfen gelesen werden. | Keine raw action state-, queue-, memory-store-, compute-backend-, policy- oder safety-state Reads. |
| `hardened deepened state surface` | Der Modellzustand bleibt relation-lokal: `Amygdala ↔ Thalamus`, bounded Kuramoto-like mode, MD1 candidate class, IR1 direct bounded advisory mediation. | Modellzustand ist kein Contract-Zustand und keine Region Authority. |
| `hardened deepened output/advisory surface` | Outputs bleiben `advisory-only`, `caveated advisory-only`, `deferred`, `blocked`, `insufficient`, `diagnostic-only` oder `non-canonical/internal-only`. | Kein Output wird Action-, Execution-, Retry-, Memory-, Compute- oder Safety-Authority. |
| `hardened diagnostic/model boundary` | Modell-Diagnostics bleiben Modell-Diagnostics; `diagnostic-only` ist kein advisory support. | Diagnostic model output wird nicht automatisch Contract support. |
| `hardened region/relation contract boundary` | IR1- und Regions-Contracts bleiben führend; die Vertiefung liest sie bounded und überschreibt sie nicht. | Keine stillschweigende Umstellung von funktionaler Architektur auf Modellarchitektur. |
| `blocked forbidden authority path` | Direct action/execution/retry/memory/compute/safety/platform paths bleiben explizit verboten. | Forbidden paths sind nicht implementierbar über MD1. |
| `non-canonical/internal-only deepening path` | Helper-, Research-, Shortcut- oder unlisted model paths bleiben intern und nicht konsumierbar. | Keine zweite operative Modellwirklichkeit. |

Diese Map ist keine Meta-Plattform und darf nicht auf weitere Regionen oder Relationen generalisiert werden, ohne einen separaten MD1-Rescope.

## 2) Modellzustand vs. Contract-Zustand

Die erste Vertiefung trennt Modell-, Diagnostic- und Contract-Ebene kanonisch:

- Modellzustand (`bounded Kuramoto-like current mode`, candidate class, leverage flags) ist **nicht automatisch Contract-Zustand**.
- Diagnostischer Modelloutput ist **nicht automatisch advisory support**.
- `caveated advisory-only` bleibt schwacher/caveated support und ist **kein starker operativer Input**.
- `diagnostic-only` bleibt diagnostic-only und erzeugt **keinen** advisory support.
- `deferred`, `blocked` und `insufficient` bleiben explizite Nicht-Autoritätszustände.
- Model-deepening state ist **keine** Region Authority; Region-/Relations-Contracts bleiben führend.

Der kanonische Contract-Support wird nur über die separate `BlueBrainMd1FirstDeepeningContractSupportClass` gelesen. Dadurch bleibt sichtbar, ob ein Ergebnis bounded advisory support, caveated support, no support, diagnostic-only no support oder non-canonical no support ist.

## 3) Runtime-/Selection-/Reference-Konsum

Runtime, Selection und Reference lesen die Vertiefung konsistent über dieselbe Consumer-Read-Klasse:

- canonical selected `Amygdala ↔ Thalamus` Ergebnisse: `consistent bounded advisory/diagnostic read`,
- non-canonical/internal-only Ergebnisse: `no canonical consumer read`,
- nicht geöffnete Kandidaten bleiben deferred/blocked und erzeugen keine eigene Runtime-, Selection- oder Reference-Deutung.

Damit existiert keine zweite Interpretation desselben Modellzustands in Runtime, Selection oder Reference.

## 4) Diagnostics-/Contract-Semantik gegen Drift

Die folgenden Semantiken bleiben getrennt:

- `advisory-only` bleibt advisory-only,
- `caveated advisory-only` bleibt caveated und nicht stark,
- `deferred` bleibt deferred,
- `blocked` bleibt blocked,
- `insufficient` bleibt insufficient,
- `diagnostic-only` bleibt diagnostic-only,
- `non-canonical/internal-only` bleibt nicht konsumierbar.

Insbesondere darf `diagnostic-only` nicht als advisory support gelesen werden, und `caveated` darf nicht zu einer erfolgreichen/starken Basis promovieren.

## 5) Region-/Relations-Grenzen

Die Vertiefung bleibt auf die IR1-Relation `Amygdala ↔ Thalamus` begrenzt:

- `Amygdala ↔ Basal Ganglia` bleibt priorisiert aber deferred und wird nicht mitvertieft.
- Hippocampus-, Thalamus-, Basal-Ganglia- und Cerebellum-Surfaces werden nicht durch das Modell übernommen.
- IR1-Relationsklassen und Mediation paths bleiben führend.
- Die inter-region architecture wird nicht umgebaut und nicht durch eine Modellarchitektur ersetzt.

## 6) No-direct-* Guard-Linie

MD1 Prompt 3 hält explizit verboten:

- kein direct action trigger,
- kein direct execution trigger,
- kein direct retry trigger und keine Retry-Orchestrierung,
- kein direct memory commit und keine automatische Memory-Persistenz,
- kein direct compute invocation,
- kein safety override,
- keine implizite zweite Modellvertiefung,
- keine implizite globale Kuramoto-/HH-/Dynamikplattform.

Die darunterliegenden BB10/BB12/BB16 boundary guards müssen weiterhin `false` für action execution, retry orchestration, memory commit, compute invocation und safety override bleiben.

## 7) MD1 next steps

1. Falls externe Evidence benötigt wird, eine schmale golden fixture nur für `Amygdala ↔ Thalamus` ergänzen.
2. `Amygdala ↔ Basal Ganglia` weiter deferred halten und nur nach separatem Rescope öffnen.
3. HH-Pfade weiter simulation-only/diagnostic-only halten; keine Produktivintegration ohne eigene Scope-Entscheidung.
4. Consumer-Read-Klasse bei zukünftigen Runtime-/Selection-/Reference-Änderungen als einheitliche Lesefläche verwenden.
5. IR1- und no-direct-* Tests vor jeder weiteren Modellvertiefung erneut gegen Contract-Drift prüfen.
