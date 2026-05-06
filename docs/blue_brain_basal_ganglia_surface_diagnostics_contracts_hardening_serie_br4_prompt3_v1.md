# Serie BR4 Prompt 3: Basal-Ganglia Surface/Diagnostics/Contracts härten

Status: `basal_ganglia_like_region` bleibt die vierte echte anatomische UCF-/Blue-Brain-Region nach BR1 Hippocampus, BR2 Amygdala und BR3 Thalamus. Diese Prompt-3-Linie härtet die in Prompt 2 eingeführte input/state/output/reference surface zu einer kanonischen diagnostics/contract line. Die Linie bleibt schmal, bounded, advisory-only und ohne direkte Action-, Retry-, Memory-, Compute-, Execution- oder Safety-Autorität.

## 1) Repo-basierte Surface-Prüfung und Sichtbarkeit

Die bereits vorhandene Basal-Ganglia-Sichtbarkeit bleibt auf diese vier Surfaces begrenzt:

- `basal-ganglia input surface`: Runtime-readiness, Selection-priority, Selection-deferral, Action-gating-posture, Context-reference und Reference-validity als bounded Lesebasis.
- `basal-ganglia state surface`: region-lokale action-gating/suppression/channel-selection/readiness/reference Zustände ohne eigene operative Autorität.
- `basal-ganglia output/advisory surface`: `gating-hint`, `suppression-hint`, `channel-selection hint`, `execution-readiness caveat`, `reference-bounded signal` und reine diagnostics.
- `basal-ganglia reference surface`: reference-bounded Context-/Reference-validity-Sicht ohne Retrieval-, Consolidation- oder Memory-Persistenz-Linie.

Runtime, Selection und Reference lesen Basal Ganglia nur über denselben kanonischen bounded contract read. Es gibt keine Runtime-eigene, Selection-eigene oder Reference-eigene Umdeutung desselben basal-ganglia state.

## 2) Kanonische basal-ganglia diagnostics/contract map

Die kanonische Map besteht aus genau diesen Klassen:

1. `basal-ganglia advisory-only diagnostic`
2. `basal-ganglia caveated diagnostic`
3. `basal-ganglia deferred diagnostic`
4. `basal-ganglia blocked diagnostic`
5. `basal-ganglia insufficient diagnostic`
6. `basal-ganglia diagnostic-only state`
7. `basal-ganglia bounded contract signal`
8. `non-canonical/internal-only basal-ganglia path`

Diese Map baut keine neue Meta-Plattform. Sie konsolidiert nur, wie die bestehende Basal-Ganglia-Surface diagnostisch und contract-nah gelesen wird.

## 3) Advisory-only vs caveated

`basal-ganglia advisory-only diagnostic != basal-ganglia caveated diagnostic`.

- `basal-ganglia advisory-only diagnostic` ist ein bounded positives Signal für action-gating/suppression/channel-selection Lesarten. Es bleibt hint-only und erzeugt keine direkte Autorität.
- `basal-ganglia caveated diagnostic` ist kein starkes positives Signal. Es kann aus schwacher Reference-, Selection- oder Execution-readiness-Basis oder aus partiellem Basal-Ganglia-Signal entstehen.
- Caveated darf nicht zu advisory-only hochgestuft werden und darf keine Execution-, Retry-, Memory-, Compute- oder Safety-Folge auslösen.

## 4) Deferred vs blocked vs insufficient

`basal-ganglia deferred diagnostic != basal-ganglia blocked diagnostic` und `basal-ganglia blocked diagnostic != basal-ganglia insufficient diagnostic`.

- `basal-ganglia deferred diagnostic` bedeutet bounded Aufschub oder Zurückstellung, etwa wegen stale/deferred Selection-, Deferral- oder Reference-Basis.
- `basal-ganglia blocked diagnostic` bedeutet begrenzender Contract-, Safety- oder Reference-Zustand; er bleibt ein Block-/Diagnostic-Read und startet keine Ausführung.
- `basal-ganglia insufficient diagnostic` bedeutet keine tragfähige bounded Basis. Insufficient ist nicht deferred, nicht blocked und nicht caveated.

Damit können Runtime, Selection und Reference keinen impliziten Positiv-, Retry- oder Safety-Override-Read aus diesen Zuständen ableiten.

## 5) Diagnostic-only und bounded contract signal

- `basal-ganglia diagnostic-only state` beschreibt reference-only/read-only Basal-Ganglia-Sichtbarkeit. Sie ist diagnostisch nützlich, aber nicht ausführungswirksam.
- `basal-ganglia bounded contract signal` ist eine lesbare Contract-Bindung zwischen Runtime, Selection und Reference. Sie ist kein Action-Kanal, kein Execution-Kanal, kein Retry-Kanal und kein Memory-/Compute-Kanal.
- `non-canonical/internal-only basal-ganglia path` bleibt sichtbar, aber operativ unzulässig. Interne/test-only Signale dürfen nicht zu einer zweiten operativen Basal-Ganglia-Wirklichkeit werden.

## 6) Runtime-/Selection-/Reference-Konsum

Die drei Konsumpunkte lesen dieselbe Semantik:

| Konsumpunkt | Erlaubter Read | Ausdrücklich ausgeschlossen |
| --- | --- | --- |
| Runtime | bounded diagnostic/advisory contract read | keine Compute-Invocation, keine Retry-Orchestrierung, kein Safety-Override |
| Selection | advisory action-gating/suppression/channel-selection read | keine Action-Auswahl, kein Planner-/Agentenentscheid, keine allowed-actions-Erweiterung |
| Reference | reference-bounded diagnostic/read-only read | kein Memory-Commit, keine Retrieval-/Consolidation-Linie, keine zweite Referenzwirklichkeit |

Execution-interface darf basal-ganglia Zustände nur als `execution-readiness caveat` lesen. Das ist eine Caveat-/Diagnostic-Lesart, kein Trigger und keine Freigabe.

## 7) Contract-Signale ohne direkte Autorität

Ein `basal-ganglia bounded contract signal` ist ausdrücklich:

- no action request
- no action selection
- no execution trigger
- no retry trigger
- no memory commit
- no compute trigger
- no safety override

Weiterhin ausgeschlossen bleiben Tool-/Action-Steuerung, Planner-/Agentenlogik, Policy-/Governance-Plattformen, Retry-/Queue-/Orchestration-Plattformen, automatische Memory-Persistenz, keine allowed-actions-Flächen, keine neuen allowed-actions-Flächen und keine Compute-Core-Arbeit.

## 8) Modellgrenze

`current model mode remains unchanged`: Basal Ganglia bleibt im `abstract functional current mode`.

Explizit getrennt bleiben:

- `bounded Kuramoto-like candidate` bleibt ein späterer Kandidat nur für eng begrenzte, advisory-only Auswahlkanal-Kopplung.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt nicht-operativ.
- `later selective HH deepening` und `HH-later` brauchen eine explizite spätere Re-Entscheidung.
- deferred Modellpfade bleiben deferred und öffnen keine Runtime-, Compute- oder Produktivsimulation.

Damit verschwimmen abstract/Kuramoto-like/HH-simulation-only/HH-later/deferred nicht, und es entsteht keine Modell-Drift.

## 9) Abgrenzung gegen Hippocampus, Amygdala und Thalamus

- `hippocampus_like_region` bleibt context/reference/episode/indexing-lastig.
- `amygdala_like_region` bleibt salience/valence/caveat/priority-lastig.
- `thalamus_like_region` bleibt relay/gating/routing-lastig.
- `basal_ganglia_like_region` bleibt action-gating/suppression/channel-selection-lastig.

Spätere bounded Kopplung darf diese Rollen ergänzen, aber nicht gleichsetzen. Basal Ganglia erzeugt keine semantische Dublette zu Hippocampus, Amygdala oder Thalamus.

## 10) Non-canonical/internal-only Restpfade

Lose Hilfs-/Shortcut-Pfade, test-only Signale und stärkere biologische Claims gelten als `non-canonical/internal-only basal-ganglia path`, solange sie nicht explizit auf die kanonische bounded Surface heruntergemappt sind. Sie bleiben diagnostisch sichtbar, aber operativ ausgeschlossen.

## 11) BR4 nächste Schritte (3-5)

1. Basal-Ganglia-readiness sweep gegen diese Prompt-3-Contract-Line erstellen.
2. Guard-/Report-Artefakte für advisory-only, caveated, deferred, blocked, insufficient und diagnostic-only Reads erweitern.
3. Cross-region contract matrix für Hippocampus/Amygdala/Thalamus/Basal Ganglia konsolidieren, ohne neue Kopplungsautorität.
4. Non-canonical/internal-only Pfade weiter auditieren und nur diagnostisch sichtbar halten.
5. Erst danach separat entscheiden, ob bounded Kuramoto-like channel coupling oder HH simulation-only diagnostics vertieft werden.
