# Serie BR5 Prompt 1: Cerebellum als nächste echte Hirnregion funktional festziehen

Status: `cerebellum_like_region` wird nach BR1 Hippocampus, BR2 Amygdala, BR3 Thalamus und BR4 Basal Ganglia als **nächste echte anatomische UCF-Region** funktional festgezogen. Die Linie ist strikt bounded, advisory-only, prediction-/timing-/correction-/mismatch-nah und erzeugt keine neue Ausführungs-, Retry-, Memory-, Policy-, Planner- oder Compute-Autorität.

Diese Datei ist die kanonische BR5-Prompt-1-Rollenkarte für das Cerebellum. Sie ersetzt keine bestehenden BlueBrain-Contract-Linien, sondern hängt an BB10/BB12/BB13/BB14/BB16/BB19/BB21, die BR1–BR4-Regionenabgrenzung und die bestehende DBM-18-Cerebellum-Roadmap an.

## 1) Repo-basierte Cerebellum-Funktionsfläche

Cerebellum-nahe Substanz ist im aktuellen Repo bereits funktional sichtbar über:

- prediction-/timing-/correction-/mismatch-nahe Lesarten in bounded dynamics und execution feedback,
- bounded execution-interface relevance über Execution-Result-, Eligibility-, Transition- und Reference-Guard-Linien,
- error-shaping/coordination semantics als diagnostische Kalibrierung, nicht als Planner oder Action-Orchestrator,
- runtime-/selection-Konsumpunkte, die advisory-only Signale lesen können, ohne direkte Autorität abzuleiten,
- bestehende funktionale Regionenflächen und `dbm_18_cerebellum` als Roadmap-/Modulebene ohne produktive Deep-Simulation.

Der echte Hebel liegt damit bei **Prediction, Timing/Coordination, Error-Correction/Mismatch-Shaping und bounded Execution-Support**. Dieser Hebel bleibt absichtlich unter den bestehenden no-direct-* Guard Rails.

## 2) Kanonische Cerebellum-Rollenkarte

Die kanonische Cerebellum-Rolle in BR5 ist auf genau fünf Klassen begrenzt:

1. `prediction role`
2. `timing/coordination role`
3. `error-correction or mismatch-shaping role`
4. `bounded execution-support role`
5. `non-role / out-of-scope biological detail`

Interpretation:

- Cerebellum ist in UCF eine **advisory prediction/timing/correction/mismatch Region**.
- Sie kann bounded Hinweise liefern, ob erwartete Ausführungs-, Timing-, Feedback- oder Referenzsignale kalibriert, caveated, deferred, blocked oder insufficient gelesen werden sollten.
- Sie darf keine direkte Action wählen, keine Execution freigeben, keinen Retry starten, keine Memory-Referenz schreiben und keinen Compute-Pfad auslösen.
- Sie ist kein biologischer Vollnachbau des Cerebellums und keine neue Meta-Plattform.

## 3) Current integration mode (BR5)

Aktueller Integrationsmodus für das Cerebellum ist:

- `abstract functional current mode` (operativ in BR5)

Explizit weiterhin unterschieden:

- `bounded Kuramoto-like candidate` bleibt ein späterer Kandidat nur für eng begrenzte, advisory-only Timing-/Coordination-Kopplung.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt nicht-operativ und liefert keine Produktivautorität.
- `later selective HH deepening` bleibt separater späterer Re-Scope, falls eine konkrete diagnostische Fragestellung dies rechtfertigt.
- deferred components bleiben deferred; insbesondere Purkinje-/Granule-/Deep-Nuclei-/Microcircuit-/Spiking-Details werden nicht geöffnet.

Damit wird **keine implizite HH-Pflicht**, **kein impliziter Kuramoto-Produktivzwang** und **keine Vollsimulation** erzeugt.

## 4) Spiegelung gegen bestehende BlueBrain-Linien

Cerebellum stützt bestehende Linien klar begrenzt:

- **Prediction/Timing/Correction:** bounded advisory Kalibrierung erwarteter Feedback-, Transition-, Timing- und mismatch-Signale; keine direkte Planung.
- **Bounded dynamics:** darf als bounded Kuramoto-like candidate später geprüft werden, bleibt jetzt aber abstract functional; keine globale Neurodynamikplattform.
- **Execution-interface/Eligibility/Safety:** darf execution-support nur bounded modulieren; no direct execution trigger, no direct action execution und no safety override.
- **Runtime/Selection:** Runtime und Selection dürfen Cerebellum-Signale nur als bounded advisory/diagnostic Contract lesen; no direct compute invocation, no direct action selection und keine Retry-Orchestrierung.
- **Reference/Context:** darf Reference-Signale nur zur mismatch-/correction-nahen Lesart konsumieren; no direct memory commit, keine Retrieval-/Consolidation-Linie und keine implizite Memory-Persistenz.

No-direct-* Grenzen bleiben unverändert:

- no direct action trigger
- no direct action selection
- no direct action execution
- no direct execution trigger
- no retry trigger
- no retry orchestration
- no direct memory commit
- no direct compute invocation
- no safety override

## 5) Abgrenzung zu Hippocampus, Amygdala, Thalamus und Basal Ganglia

Klare Rollentrennung in BR5:

- `hippocampus_like_region`: context/reference/episode/indexing-lastig.
- `amygdala_like_region`: salience/valence/caveat/priority-lastig.
- `thalamus_like_region`: relay/gating/routing-lastig.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection-lastig.
- `cerebellum_like_region`: prediction/timing/correction/mismatch-lastig.

Konsequenz:

- Hippocampus bindet Kontext, Episoden und Referenzen; Cerebellum schreibt keine Memory-Referenz und übernimmt keine Indexing-Autorität.
- Amygdala gewichtet Salienz/Valenz/Caveats/Priorität; Cerebellum entscheidet keine emotionale Priorität und keine Governance.
- Thalamus relayed/routed bounded Signale; Cerebellum relayed nicht als Routing-Hub, sondern formt Prediction-/Timing-/Correction-Mismatch-Lesarten.
- Basal Ganglia stützt action-gating, suppression und channel-selection; Cerebellum wählt keine Action-Kanäle und unterdrückt keine Handlung als Autorität.
- Hypothalamus bleibt deferred und würde eher homeostasis/drive/arousal-nahe Fragen tragen; Cerebellum öffnet diese Linie nicht.
- Spätere bounded Kopplung zwischen Regionen ist komplementär und bedeutet keine Gleichsetzung und keine semantische Dublette.

## 6) Explizit out-of-scope in diesem Schritt

Nicht Teil von BR5 Prompt 1:

- kein vollständiger biologischer Cerebellum-Nachbau,
- keine Purkinje-/Granule-/Deep-Nuclei-/Microcircuit-Vollmodellierung,
- keine HH-Produktivintegration,
- keine direkte Execution-/Safety-/Memory-Autorität,
- keine direkte Action-Auswahl, keine allowed-actions-Erweiterung,
- keine Planner-/Agenten-/Policy-/Governance-/Retry-/Queue-/Orchestration-Plattform,
- keine Retrieval-/Consolidation-/Reasoning-Plattform,
- keine implizite Memory-Persistenz,
- keine neue Compute-Core-Arbeit,
- keine globale Neurodynamikplattform.

## 7) Anschluss- und Konsistenzentscheidung

Repo-seitig taucht keine starke Gegenindikation gegen **Cerebellum vor Hypothalamus** auf, solange die Region strikt abstract-functional und no-direct bleibt. Der BR4-Handoff wird damit eingelöst, ohne eine Vollsimulation oder HH-Produktivpflicht zu öffnen.

Warum Cerebellum jetzt vor Hypothalamus:

1. Die vorhandene Prediction-/Timing-/Correction-/mismatch-nahe Substanz liefert einen direkteren funktionalen Anschluss.
2. Execution feedback und bounded dynamics können advisory kalibriert werden, ohne Action-, Retry- oder Compute-Autorität zu erzeugen.
3. Hypothalamus-nahe homeostasis/drive/arousal-Semantik hätte höhere Scope-Risiken und bleibt deferred.
4. BR5 baut keine neue Plattform, sondern schärft eine bereits vorhandene calibration/mismatch lane.

## 8) BR5-Nächste Schritte (3–5)

1. Cerebellum-spezifische Diagnostics-/Contract-Token für prediction, timing/coordination und correction/mismatch in bestehenden bounded Tests aufnehmen.
2. Runtime/Selection snapshots um advisory Cerebellum markers regressionssicher machen, ohne Action-, Execution-, Retry- oder Compute-Autorität zu erzeugen.
3. Execution-interface-/Reference-Checks ergänzen, damit mismatch/correction nie zu direkter Freigabe, Memory-Commit oder Retry eskaliert.
4. Guard-Prüfungen ergänzen, damit Hippocampus/Amygdala/Thalamus/Basal Ganglia/Cerebellum semantisch getrennt bleiben.
5. Erst danach optionalen Re-Scope für bounded Kuramoto-like timing coupling oder HH simulation-only diagnostics separat entscheiden.
