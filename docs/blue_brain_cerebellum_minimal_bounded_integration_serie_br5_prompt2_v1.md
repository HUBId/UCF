# Serie BR5 Prompt 2: Cerebellum minimal und bounded in UCF einhängen

Status: `cerebellum_like_region` ist als fünfte echte anatomische UCF-/Blue-Brain-Region nach BR1 Hippocampus, BR2 Amygdala, BR3 Thalamus und BR4 Basal Ganglia minimal eingehängt. Die Integration bleibt eine kontrollierte, bounded, advisory-only Prediction-/Timing-/Correction-/Mismatch-Linie und erzeugt keine direkte Action-, Retry-, Memory-, Compute-, Execution- oder Safety-Autorität.

Diese Datei ist die kanonische BR5-Prompt-2-Integrationslinie. Sie erweitert die BR5-Prompt-1-Rollenkarte, ersetzt aber keine bestehenden Runtime-, Selection-, Execution-interface-, Reference-/Context-, Dynamics- oder Safety-Verträge.

## 1) Kleinste echte Integrationsfläche

Die kleinste belastbare Cerebellum-Fläche besteht aus genau diesen kanonischen Klassen:

| Klasse | Kanonischer Zweck | Grenze |
| --- | --- | --- |
| `cerebellum input surface` | liest bounded Runtime-/Selection-/Execution-feedback-/Reference-Signale für Prediction, Timing, Coordination, Correction und Mismatch | keine Tool-/Action-Steuerung, keine compute-internen Rohzustände, keine Safety-Override-Eingänge, keine impliziten Memory-Mutationsinputs |
| `cerebellum state surface` | hält nur abstract-functional Lesestände wie active prediction/timing advisory, timing/coordination advisory, correction/mismatch advisory, execution-support caveat, reference-only, deferred, blocked, insufficient oder non-canonical/internal-only | kein Planner-State, kein Queue-/Retry-State, kein Memory-Commit-State, kein Compute-Core-State |
| `cerebellum output/advisory surface` | liefert bounded timing hint, correction hint, mismatch hint, execution-support caveat, reference-bounded signal, blocked/deferred oder insufficient diagnostic output | keine direct action selection, kein direct execution trigger, kein direct retry trigger, kein direct compute invocation |
| `cerebellum reference surface` | konsumiert Reference/Context nur als bounded mismatch-/correction-nahe Basis und markiert stale/caveated/reference-only Fälle diagnostisch | keine zweite Referenzwirklichkeit, keine Retrieval-/Consolidation-Linie, keine implizite Memory-Persistenz |
| `blocked/deferred cerebellum path` | bildet blocked, deferred, stale, insufficient, invalidated oder rejected Fälle fail-closed ab | keine Eskalation in Execution, Retry oder Safety Override |
| `non-canonical/internal-only cerebellum path` | bleibt explizit non-canonical und diagnostisch abgegrenzt | nicht promotable, keine Runtime-/Selection-Autorität |

## 2) Input-Surface und Guards

Zulässige Cerebellum-Inputs sind nur bounded Lesesignale:

- Runtime-nahe prediction signals und timing signals,
- Selection-nahe coordination signals,
- Execution-feedback-/mismatch-Signale als execution-interface-Lesart,
- Context reference signals und Reference validity signals als reference-only/bounded Basis.

Diese Inputs bleiben advisory-only oder reference-only. Explizit unzulässig sind:

- direkte Tool-/Action-Steuersignale,
- compute-interne Rohzustände,
- direkte Safety-Override-Eingänge,
- implizite Memory-Mutationsinputs.

## 3) State- und Output-/Advisory-Surface

Kanonische Cerebellum-States bleiben voneinander unterscheidbar:

- active prediction/timing advisory-only,
- timing/coordination advisory state,
- correction/mismatch advisory state,
- execution-support caveat state,
- reference-only correction state,
- deferred correction state,
- blocked correction state,
- insufficient correction state,
- non-canonical/internal-only state.

Kanonische Outputs sind ausschließlich bounded Hinweise:

- `timing hint`,
- `correction hint`,
- `mismatch hint`,
- `execution-support caveat`,
- `reference-bounded signal`,
- `blocked/deferred`,
- `insufficient diagnostic output`,
- `non-canonical/internal-only`.

Die Output-Surface informiert Runtime, Selection, Execution-interface und Reference höchstens über denselben bounded canonical contract read. Sie erzeugt keine Proposal-, Action-, Execution-, Retry-, Memory-, Compute- oder Safety-Autorität.

## 4) Runtime-/Selection-/Execution-interface-Kopplung

Runtime sieht Cerebellum nur als bounded advisory/diagnostic Prediction-/Timing-Kalibrierung. Selection sieht Cerebellum nur als bounded coordination-/correction-/mismatch-Hinweis. Das Execution-interface darf nur eine execution-support caveat oder einen reference-bounded signal readen; daraus folgt keine direkte Ausführung und keine Action-Freigabe.

Damit gilt:

- Prediction/Timing/Correction fließen nur bounded und advisory-only ein.
- Runtime, Selection, Execution-interface und Reference lesen dieselbe kanonische Cerebellum-Contract-Klasse.
- Es entsteht keine direkte Proposal-, Action- oder Execution-Autorität.
- Es entsteht keine Planner-Logik und keine Retry-Orchestrierung.

## 5) Reference/Context bounded

Cerebellum-relevante Referenzen sind nur bounded Context-/Reference-Basis für mismatch-/correction-nahe Lesarten. Stale, caveated, blocked, insufficient und reference-only Fälle werden als deferred, caveated, blocked, insufficient oder diagnostic-only Contract gelesen.

Grenzen:

- keine implizite Memory-Persistenz,
- kein direct memory commit,
- keine Retrieval-/Consolidation-/Reasoning-Plattform,
- keine zweite Referenzwirklichkeit.

## 6) Modellgrenze

Der aktuelle Modellmodus bleibt unverändert:

- `abstract functional current mode` ist der operative Cerebellum-Modus in BR5.
- `bounded Kuramoto-like candidate` bleibt späterer optionaler Re-Scope für eng begrenzte advisory-only timing/coordination-Kopplung.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt nicht-operativ.
- Spätere Purkinje-/Granule-/Deep-Nuclei-/Microcircuit-/Spiking-Vertiefung braucht eine separate Entscheidung.

Es erfolgt keine implizite Kuramoto-Aufweitung, keine Hodgkin-Huxley-Produktivintegration und keine globale Neurodynamikplattform.

## 7) Abgrenzung zu bestehenden anatomischen Regionen

- `hippocampus_like_region`: context/reference/episode/indexing; Cerebellum schreibt keine Memory-Referenz und übernimmt kein Indexing.
- `amygdala_like_region`: salience/valence/caveat/priority; Cerebellum entscheidet keine emotionale Priorität, keine Governance und keine Safety-Override-Logik.
- `thalamus_like_region`: relay/gating/routing; Cerebellum ist kein Routing-Hub und relayed keine Signale als Autorität.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection; Cerebellum wählt keine Action-Kanäle und unterdrückt keine Handlung als Autorität.
- `cerebellum_like_region`: prediction/timing/correction/mismatch; advisory-only execution-support bleibt begrenzt.

Hypothalamus und weitere anatomische Regionen bleiben deferred; BR5 Prompt 2 öffnet keine sechste Region.

## 8) No-direct-* Grenzen

Bewusst unverändert gesperrt bleiben:

- no direct action trigger,
- no direct action selection,
- no direct action execution,
- no direct execution trigger,
- no direct retry trigger,
- no retry orchestration,
- no direct memory commit,
- no automatic memory persistence,
- no direct compute invocation,
- no safety override,
- no new allowed-actions extension,
- no planner/agent/policy/governance platform.

## 9) BR5-Nächste Schritte

1. Cerebellum-Diagnostics gegen Runtime-/Selection-Snapshots härten, ohne advisory-only zu verlassen.
2. Execution-interface-Reads mit mismatch/correction regression tests weiter pinnen.
3. Reference-only, stale und caveated Cerebellum-Fälle gegen Docs-Lint und readiness gates absichern.
4. Inter-region guards für Hippocampus/Amygdala/Thalamus/Basal Ganglia/Cerebellum weiter verdichten.
5. Danach optionalen Re-Scope für bounded Kuramoto-like timing coupling oder HH simulation-only diagnostics separat entscheiden.
