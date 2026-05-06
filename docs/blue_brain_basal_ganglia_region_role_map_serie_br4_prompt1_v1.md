# Serie BR4 Prompt 1: Basal Ganglia als nächste echte Hirnregion funktional festziehen

Status: `basal_ganglia_like_region` wird nach BR1 Hippocampus, BR2 Amygdala und BR3 Thalamus als **nächste echte anatomische UCF-Region** funktional festgezogen. Die Linie ist strikt bounded, advisory-only, selection-/action-gating-nah und erzeugt keine neue Ausführungs-, Retry-, Memory-, Policy- oder Compute-Autorität.

Diese Datei ist die kanonische BR4-Prompt-1-Rollenkarte für Basal Ganglia. Sie ersetzt keine bestehenden BlueBrain-Contract-Linien, sondern hängt an BB4/BB13/BB14/BB19/BB21 und die BR1–BR3-Regionenabgrenzung an.

## 1) Repo-basierte Basal-Ganglia-Funktionsfläche

Basal-Ganglia-nahe Substanz ist im aktuellen Repo bereits funktional sichtbar über:

- action-gating-/selection-channel-nahe Contract-Semantik ohne direkte Action-Auswahl,
- suppression-/blocked-/deferred-/insufficient-Diagnostik in bestehenden Selection- und Execution-interface-Grenzen,
- priority/deferral competition als bounded advisory Lesart, nicht als Planner- oder Policy-Entscheider,
- bounded execution-interface relevance über Eligibility-/Safety-/Reference-Guard-Linien,
- bestehendes anatomisches Mapping `basal_ganglia_like_region` / `ActionGatingMediation` ohne produktive Deep-Simulation.

Der echte Hebel liegt damit bei **Go/No-Go-artiger action-gating Unterstützung, Inhibitions-/Suppression-Hinweisen und Auswahlkanal-Arbitration**. Dieser Hebel bleibt absichtlich unter den bestehenden no-direct-* Guard Rails.

## 2) Kanonische Basal-Ganglia-Rollenkarte

Die kanonische Basal-Ganglia-Rolle in BR4 ist auf genau fünf Klassen begrenzt:

1. `action gating role`
2. `suppression/inhibition role`
3. `bounded selection-channel arbitration role`
4. `execution-readiness modulation role`
5. `non-role / out-of-scope biological detail`

Interpretation:

- Basal Ganglia ist in UCF eine **advisory action-gating/suppression/channel-selection Region**.
- Sie kann bounded Hinweise liefern, ob bestehende Auswahlkanäle eher verstärkt, unterdrückt, deferred oder blocked gelesen werden sollten.
- Sie darf keine direkte Action wählen, keine Execution freigeben und keinen Retry starten.
- Sie ist kein biologischer Vollnachbau der Basal Ganglia und keine neue Meta-Plattform.

## 3) Current integration mode (BR4)

Aktueller Integrationsmodus für Basal Ganglia ist:

- `abstract functional current mode` (operativ in BR4)

Explizit weiterhin unterschieden:

- `bounded Kuramoto-like candidate` bleibt ein späterer Kandidat nur für eng begrenzte, advisory-only Auswahlkanal-Kopplung.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt nicht-operativ und liefert keine Produktivautorität.
- `later selective HH deepening` bleibt separater späterer Re-Scope, falls eine konkrete diagnostische Fragestellung dies rechtfertigt.
- deferred components bleiben deferred; insbesondere biologische Subnukleus-/Dopamin-/Spiking-Details werden nicht geöffnet.

Damit wird **keine implizite HH-Pflicht**, **kein impliziter Kuramoto-Produktivzwang** und **keine Vollsimulation** erzeugt.

## 4) Spiegelung gegen bestehende BlueBrain-Linien

Basal Ganglia stützt bestehende Linien klar begrenzt:

- **Selection/Action-gating:** bounded advisory Go/No-Go-, channel-selection- und suppression Hinweise; no direct action selection.
- **Priority/Deferral:** darf competition/posture als caveated/deferred/blocked/insufficient Lesart modulieren, aber keine Policy- oder Planner-Entscheidung treffen.
- **Execution-interface/Eligibility/Safety:** darf execution-readiness nur bounded modulieren; no direct execution trigger und no safety override.
- **Reference/Context:** darf Reference/Context nur bounded beeinflussen, etwa als caveat-aware Hinweis auf Auswahlkanal-Kontext; no direct memory commit und keine neue Retrieval-/Consolidation-Linie.
- **Runtime:** Runtime darf Basal-Ganglia-Signale nur als bounded advisory/diagnostic Contract lesen; no direct compute invocation und keine Retry-Orchestrierung.

No-direct-* Grenzen bleiben unverändert:

- no direct action trigger
- no direct action selection
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

## 5) Abgrenzung zu Hippocampus, Amygdala und Thalamus

Klare Rollentrennung in BR4:

- `hippocampus_like_region`: context/reference/episode/indexing-lastig.
- `amygdala_like_region`: salience/valence/caveat/priority-lastig.
- `thalamus_like_region`: relay/gating/routing-lastig.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection-lastig.

Konsequenz:

- Hippocampus bindet Kontext und Referenzen; Basal Ganglia bindet keine Memory-/Reference-Autorität.
- Amygdala gewichtet Salienz/Valenz/Caveats; Basal Ganglia entscheidet keine emotionale Priorität und keine Governance.
- Thalamus relayed/routed bounded Signale; Basal Ganglia arbitriert selection-channel Haltung und Suppression nur advisory.
- Spätere bounded Kopplung zwischen Regionen ist komplementär und bedeutet **keine** Gleichsetzung oder semantische Dublette.

## 6) Explizit out-of-scope in diesem Schritt

Nicht Teil von BR4 Prompt 1:

- kein vollständiger biologischer Basal-Ganglia-Nachbau,
- keine Subnukleus-/Dopamin-/direkter-vs-indirekter-Pfad-Vollmodellierung,
- keine HH-Produktivintegration,
- keine direkte Execution-/Safety-/Memory-Autorität,
- keine direkte Action-Auswahl, keine allowed-actions-Erweiterung,
- keine Planner-/Agenten-/Policy-/Governance-/Retry-/Queue-/Orchestration-Plattform,
- keine Retrieval-/Consolidation-/Reasoning-Plattform,
- keine implizite Memory-Persistenz,
- keine neue Compute-Core-Arbeit,
- keine globale Neurodynamikplattform.

## 7) Anschluss- und Konsistenzentscheidung

Repo-seitig taucht keine starke Gegenindikation gegen **Basal Ganglia vor Cerebellum** auf, solange die Region strikt abstract-functional und no-direct bleibt. Der frühere BR3-Ausblick auf Cerebellum bleibt als historischer BR3-Closure-Kontext lesbar, wird für die nächste aktive BR4-Regionsdefinition aber durch diese engere Basal-Ganglia-Rollenkarte ersetzt.

Warum Basal Ganglia jetzt vor Cerebellum:

1. Die vorhandene Selection-/Priority-/Deferral-/Suppression-Substanz liefert einen direkteren funktionalen Anschluss.
2. Die action-gating Nähe wird durch no-direct action selection, no direct execution trigger und no retry trigger ausreichend begrenzt.
3. Cerebellum bleibt eher Kalibrierungs-/Timing-Kandidat und benötigt aktuell keinen neuen operativen Hebel.
4. BR4 baut keine neue Plattform, sondern schärft eine bereits benannte action-gating mediation lane.

## 8) BR4-Nächste Schritte (3–5)

1. Basal-Ganglia-spezifische Diagnostics-/Contract-Token für action-gating, suppression und channel-selection in bestehenden bounded Tests aufnehmen.
2. Runtime/Selection snapshots um advisory Basal-Ganglia markers regressionssicher machen, ohne Action- oder Execution-Autorität zu erzeugen.
3. Execution-interface-/Eligibility-Checks ergänzen, damit execution-readiness modulation nie zu direkter Freigabe eskaliert.
4. Guard-Prüfungen ergänzen, damit Hippocampus/Amygdala/Thalamus/Basal Ganglia semantisch getrennt bleiben.
5. Erst danach optionalen Re-Scope für bounded Kuramoto-like channel coupling oder HH simulation-only diagnostics separat entscheiden.
