# Serie BR6 Prompt 3: Hypothalamus Surface/Diagnostics/Contracts Hardening

Status: `hypothalamus_like_region` bleibt eine bounded, advisory-only UCF-BlueBrain-Anschlussfläche. Diese Datei härtet die BR6-Prompt-2-Surface gegen lose Interpretation und ist die kanonische diagnostics/contract map für Hypothalamus-Zustände. Sie ersetzt keine Rollenkarte und öffnet keine neue Anatomie-, Compute-, Planner-, Agenten-, Retry-, Policy- oder Governance-Plattform.

## 1) Kanonische Hypothalamus-Surface-Klassen

Die kanonische Hypothalamus-Surface besteht genau aus diesen Klassen:

1. `hypothalamus input surface`
2. `hypothalamus state surface`
3. `hypothalamus output/advisory surface`
4. `hypothalamus reference surface`
5. `hypothalamus diagnostics/contract map`
6. `blocked/deferred hypothalamus path`
7. `non-canonical/internal-only hypothalamus path`

Diese Klassen sind Contract- und Diagnostic-Labels. Sie sind keine operative Region-zu-Region-Nachrichtenengine und keine zweite Runtime-/Selection-/Reference-Wirklichkeit.

## 2) Kanonische diagnostics/contract states

Die Hypothalamus-Diagnostics/Contract-Map unterscheidet mindestens und abschließend für BR6 Prompt 3:

| State | Kanonische Bedeutung | Consumer-Lesart |
|---|---|---|
| `hypothalamus advisory-only diagnostic` | bounded positives Signal für drive/homeostasis/urgency/state-pressure; keine direkte Autorität | Runtime/Selection/Reference lesen advisory-only |
| `hypothalamus caveated diagnostic` | schwache, partielle oder caveated Basis; kein starkes positives Signal | Runtime/Selection/Reference lesen caveated |
| `hypothalamus deferred diagnostic` | bounded Aufschub/Zurückstellung, z. B. stale oder pending context/reference | Runtime/Selection/Reference lesen deferred |
| `hypothalamus blocked diagnostic` | begrenzender Contract-/Safety-/Reference-Zustand | Runtime/Selection/Reference lesen blocked |
| `hypothalamus insufficient diagnostic` | keine tragfähige bounded Basis | Runtime/Selection/Reference lesen insufficient |
| `hypothalamus diagnostic-only state` | nur Diagnose-/Reference-Read, keine positive advisory Wirkung | Runtime/Selection/Reference lesen diagnostic-only |
| `hypothalamus bounded contract signal` | kanonischer bounded Contract-Token zwischen Surface und Consumers | keine Action-/Execution-/Memory-/Compute-Autorität |
| `non-canonical/internal-only hypothalamus path` | interner/test-only oder nicht-kanonischer Restpfad | kein kanonischer Consumer-Read |

## 3) Advisory-only vs caveated

`advisory-only` ist ein bounded positives Signal: Es darf drive-state, homeostasis/regulation caveat, urgency modulation oder context-linked state-pressure als Hinweis sichtbar machen, ohne direkte Autorität zu erzeugen.

`caveated` ist kein starkes positives Signal. Caveated entsteht insbesondere aus schwacher Reference-/Selection-/State-Basis, caveated Context Evidence oder partiellem Hypothalamus-Signal. Caveated darf advisory-only nicht implizit ersetzen oder aufwerten; Runtime, Selection und Reference müssen caveated als caveated lesen.

## 4) Deferred vs blocked vs insufficient

Die drei negativen/limitierenden Zustände bleiben getrennt:

- `deferred` = bounded Aufschub/Zurückstellung, z. B. pending stronger evidence, pending context update, stale oder not-persisted.
- `blocked` = begrenzender Contract-/Safety-/Reference-Zustand, z. B. rejected, blocked oder invalidated.
- `insufficient` = keine tragfähige bounded Basis, z. B. insufficient candidate, insufficient reference oder insufficient context.

`deferred ist nicht blocked`. `blocked ist nicht insufficient`. `insufficient ist nicht deferred`. Diese Zustände dürfen nicht unter einem gemeinsamen blocked/deferred Output verschwimmen; die Diagnostic- und Contract-Lesart bleibt getrennt.

## 5) Runtime/Selection/Reference-Konsum

Runtime, Selection, Context und Reference lesen dieselbe kanonische Hypothalamus-Semantik:

- Runtime liest bounded urgency/state-pressure/homeostasis diagnostics, aber mutiert Runtime nicht direkt.
- Selection liest bounded urgency/state-pressure/homeostasis diagnostics, aber erweitert keine allowed-actions, wählt keine Action und gibt keine Execution frei.
- Context/Reference liest bounded state-pressure/regulation/reference diagnostics, aber erzeugt keinen Retrieval-, Consolidation- oder Memory-Commit-Pfad.

Für denselben Hypothalamus-Zustand darf es keine eigene Runtime-Deutung, Selection-Deutung oder Reference-Deutung geben. Consumer-spezifische Felder sind nur Sichtbarkeitsfelder; der canonical contract read bleibt identisch.

## 6) No-direct-authority Contract Guards

Ein `hypothalamus bounded contract signal` ist ausdrücklich:

- kein action request,
- kein action selection signal,
- kein execution trigger,
- kein retry trigger,
- keine Retry-/Queue-/Orchestration-Entscheidung,
- kein memory commit,
- keine implizite Memory-Persistenz,
- kein compute trigger,
- kein safety override,
- keine Policy-/Governance-Autorität,
- keine Planner-/Agenten-Autorität.

Nicht-kanonische oder internal-only Hypothalamus-Pfade dürfen nicht operativ wirken.

## 7) Current model mode

`current model mode remains unchanged`: Der aktuelle Modus bleibt `abstract functional current mode`.

Die folgenden Modellpfade bleiben getrennt und nicht-produktiv:

- `bounded Kuramoto-like candidate` bleibt ein späterer, explizit zu entscheidender Kandidat.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt Diagnose-/Simulation-only und keine Produktivintegration.
- `later selective HH deepening` braucht explizite Re-Entscheidung.
- deferred biologische Details bleiben deferred.

Es gibt keine automatische Kuramoto-Aufweitung, keine Hodgkin-Huxley-Produktivintegration und keine Modell-Drift aus Hypothalamus-Diagnostics heraus.

## 8) Abgrenzung zu übrigen Regionen

Die Regionengrenzen bleiben strikt:

- `hippocampus_like_region`: context/reference/episode/indexing; Hypothalamus schreibt keine Memory-/Reference-Autorität.
- `amygdala_like_region`: salience/valence/caveat/priority; Hypothalamus erzeugt keine emotionale Valenz, Threat-Semantik oder Safety-Override.
- `thalamus_like_region`: relay/gating/routing; Hypothalamus ist kein Relay-/Routing-Hub und ändert Routing nicht direkt.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection; Hypothalamus wählt keine Actions und sperrt oder öffnet keine Execution-Kanäle.
- `cerebellum_like_region`: prediction/timing/correction/mismatch; Hypothalamus ersetzt keine Timing-, Prediction- oder Mismatch-Korrektur.
- `hypothalamus_like_region`: bounded drive/homeostasis/urgency/state-pressure.

Bounded spätere Kopplung bedeutet Komplementarität, nicht Gleichsetzung und keine semantische Dublette.

## 9) BR6-Folgepfad

Diese Härtung wird durch `docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md` abgeschlossen. Die Prompt-4-Datei bleibt die kanonische BR6-Abschluss- und Expansionsgrenze; diese Prompt-3-Datei bleibt die detaillierte diagnostics/contract map.

Nächste sinnvolle Schritte nach dieser Härtung sind daher keine weitere Anatomieöffnung, sondern die dort priorisierte System-Audit/Consolidation-Linie:

1. Consumer-facing Fixtures/Goldens nur ergänzen, falls externe Consumers die Hypothalamus-Diagnostics maschinenlesbar verwenden müssen.
2. Readiness-/Guard-Doku gegen die diagnostics/contract map und die Prompt-4-Expansionsgrenze prüfen.
3. Inter-region-Diagnostics nur dort erweitern, wo die bestehende IR1-Relation eine Hypothalamus-Adjunct-Lesart bereits erlaubt.
4. Optionalen bounded Kuramoto-like oder HH-simulation-only Re-Scope separat entscheiden; keine implizite Produktivvertiefung.
5. Keine nächste echte Hirnregion aus Prompt 3 heraus öffnen.
