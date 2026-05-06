# Blue Brain Cerebellum Surface/Diagnostics/Contracts Hardening — Serie BR5 Prompt 3 v1

Status: `cerebellum_like_region` bleibt die fünfte echte anatomische UCF-/Blue-Brain-Region nach Hippocampus, Amygdala, Thalamus und Basal Ganglia. Diese Hardening-Linie konsolidiert die Cerebellum-Surface aus BR5 Prompt 2 zu einer kanonischen diagnostics/contract map. Sie ändert den current model mode nicht und erzeugt keine direkte Action-, Execution-, Retry-, Memory-, Compute-, Planner-, Policy- oder Safety-Autorität.

## 1) Kanonische Cerebellum-Surfaces

| Surface | Kanonische Lesart | Harte Grenze |
| --- | --- | --- |
| `cerebellum input surface` | bounded Runtime prediction/timing signals, Selection coordination signals, Execution-interface mismatch feedback, Reference/Context validity | keine Tool-/Action-Steuerung, keine compute-internen Rohzustände, keine Safety-Override-Eingänge, keine impliziten Memory-Mutationsinputs |
| `cerebellum state surface` | active prediction/timing advisory-only, timing/coordination advisory, correction/mismatch advisory, execution-support caveat, reference-only, deferred, blocked, insufficient, non-canonical/internal-only | kein Planner-State, kein Queue-/Retry-State, kein Memory-Commit-State, kein Compute-Core-State |
| `cerebellum output/advisory surface` | timing hint, correction hint, mismatch hint, execution-support caveat, reference-bounded signal, blocked/deferred, insufficient diagnostic output, non-canonical/internal-only | keine direct action selection, kein direct action trigger, kein direct execution trigger, kein direct retry trigger, kein direct compute invocation |
| `cerebellum reference surface` | bounded mismatch-/correction-nahe Reference-Lesart für current, stale, caveated, blocked, insufficient oder reference-only Basis | keine zweite Referenzwirklichkeit, keine Retrieval-/Consolidation-Linie, kein direct memory commit, keine automatic memory persistence |
| `blocked/deferred cerebellum path` | fail-closed Contract-/Diagnostic-Lesart für bounded Aufschub oder begrenzenden Zustand | keine Eskalation in Execution, Retry, Safety Override oder Allowed-Actions |
| `non-canonical/internal-only cerebellum path` | explizit non-canonical/test-/internal-only und nicht promotable | keine Runtime-/Selection-/Reference-Autorität, keine operative zweite Cerebellum-Wirklichkeit |

## 2) Kanonische diagnostics/contract map

Die kanonische Cerebellum-Map besteht genau aus diesen Klassen:

1. `cerebellum advisory-only diagnostic` — positives bounded Signal für prediction/timing/correction/mismatch; advisory-only, keine direkte Autorität.
2. `cerebellum caveated diagnostic` — schwaches oder partielles Signal, z. B. aus caveated Reference, caveated Selection-/Execution-support oder partieller Cerebellum-Basis; kein starkes positives Signal.
3. `cerebellum deferred diagnostic` — bounded Aufschub/Zurückstellung, z. B. stale oder pending context/reference support; nicht blocked.
4. `cerebellum blocked diagnostic` — begrenzender Contract-/Safety-/Reference-Zustand, z. B. rejected, blocked oder invalidated; nicht insufficient.
5. `cerebellum insufficient diagnostic` — keine tragfähige bounded Basis für prediction/timing/correction/mismatch; nicht blocked.
6. `cerebellum diagnostic-only state` — reference-only oder anderweitig nur beobachtbarer Zustand; keine advisory-Aufwertung.
7. `cerebellum bounded contract signal` — bounded Contract-Nachricht zwischen Runtime/Selection/Execution-interface/Reference und Cerebellum; kein action request und kein trigger.
8. `non-canonical/internal-only cerebellum path` — nicht-kanonischer oder interner Restpfad; nicht promotable und nicht operativ wirksam.

## 3) Advisory-only vs caveated

`advisory-only` ist ein bounded positives Signal ohne direkte Autorität. Runtime, Selection, Execution-interface und Reference dürfen daraus nur denselben canonical contract read ableiten; sie dürfen daraus keine stärkere lokale Interpretation erzeugen.

`caveated` ist ausdrücklich kein starkes positives Signal. Es entsteht bei schwacher Reference-/Selection-/Execution-support-Basis, caveated evidence oder partiellem Cerebellum-Signal. Caveated darf nicht als Advisory-Freigabe, Action-Kandidat, Execution-Freigabe, Retry-Auslöser, Memory-Commit oder Compute-Aufruf gelesen werden.

## 4) Deferred vs blocked vs insufficient

- `deferred` bedeutet bounded Aufschub oder Zurückstellung. Deferred ist nicht blocked und nicht insufficient.
- `blocked` bedeutet begrenzender Contract-/Safety-/Reference-Zustand. Blocked ist nicht deferred und nicht insufficient.
- `insufficient` bedeutet fehlende tragfähige bounded Basis. Insufficient ist nicht blocked und nicht deferred.

Diese drei Zustände bleiben als Cerebellum-Diagnostics getrennt, damit Runtime/Selection/Reference keinen Aufschub mit einem begrenzenden Zustand oder mit fehlender Basis verwechseln.

## 5) Runtime-/Selection-/Reference-Konsum

Runtime, Selection, Execution-interface und Reference lesen denselben Cerebellum canonical contract read:

- Runtime: bounded prediction/timing/correction diagnostic read; keine Runtime-eigene Aufwertung.
- Selection: bounded coordination/mismatch diagnostic read; keine Selection-eigene Action- oder Channel-Wahl.
- Execution-interface: bounded execution-support caveat read; kein execution trigger.
- Reference: bounded reference/correction/mismatch read; keine Memory-Persistenz und keine zweite Reference-Wahrheit.

Damit gibt es keine getrennte Runtime-, Selection- oder Reference-Semantik für denselben Cerebellum-Zustand.

## 6) No-direct-authority Contract

Ein `cerebellum bounded contract signal` ist ausdrücklich:

- kein action request,
- kein direct action trigger,
- keine direct action selection,
- kein direct execution trigger,
- kein direct retry trigger,
- keine retry orchestration,
- kein direct memory commit,
- keine automatic memory persistence,
- kein direct compute invocation,
- kein safety override,
- keine allowed-actions extension,
- keine Planner-/Agenten-/Policy-/Governance-Plattform.

## 7) Current model mode

Der current model mode bleibt unverändert: `abstract functional current mode`. `bounded Kuramoto-like candidate`, `Hodgkin-Huxley simulation-only/diagnostic-only`, `later selective HH deepening` und `deferred/not-suitable-now` bleiben getrennte Modellpfade. Eine spätere Vertiefung braucht eine explizite Re-Entscheidung; BR5 Prompt 3 öffnet keine Hodgkin-Huxley-Produktivintegration und keine globale Neurodynamikplattform.

## 8) Abgrenzung zu bestehenden anatomischen Regionen

- `hippocampus_like_region`: context/reference/episode/indexing-lastig; Cerebellum übernimmt kein Indexing, keine Retrieval-/Consolidation-Linie und keine Memory-Persistenz.
- `amygdala_like_region`: salience/valence/caveat/priority-lastig; Cerebellum setzt keine emotionale Priorität, Governance oder Safety-Override-Logik.
- `thalamus_like_region`: relay/gating/routing-lastig; Cerebellum ist kein Relay- oder Routing-Hub.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection-lastig; Cerebellum wählt keine Action-Kanäle und unterdrückt keine Handlung als Autorität.
- `cerebellum_like_region`: prediction/timing/correction/mismatch-lastig; bounded execution-support bleibt advisory/caveated und nicht direkt wirksam.

Eine spätere bounded Kopplung darf nicht als Gleichsetzung dieser Regionen gelesen werden. Hypothalamus und weitere anatomische Regionen bleiben deferred.

## 9) BR5 Folgepfad

1. Readiness-Sweep für die gehärtete Cerebellum-Surface gegen Docs-Lint/Gates abschließen.
2. Reference-only, stale, caveated, blocked und insufficient Cerebellum-Fälle weiter als Fixtures/Regressionen pinnen.
3. Execution-interface mismatch/correction reads ohne Execution- oder Retry-Autorität weiter absichern.
4. Inter-region boundary checks für Hippocampus/Amygdala/Thalamus/Basal Ganglia/Cerebellum verdichten.
5. Optionalen Re-Scope für bounded Kuramoto-like timing coupling oder HH simulation-only diagnostics nur separat entscheiden.
