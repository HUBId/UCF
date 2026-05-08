# Serie BR6 Prompt 2: Hypothalamus minimal und bounded integrieren

Status: `hypothalamus_like_region` ist die erste minimal integrierte BR6-Anschlussfläche nach der Rollenkarte aus Prompt 1. Der Integrationsmodus bleibt `abstract functional current mode`: keine biologische Vollsimulation, keine globale Neurodynamikplattform, keine Hodgkin-Huxley-Produktivintegration und keine implizite Kuramoto-Aufweitung.

Diese Datei ist die kanonische Hypothalamus-Integration-Map für BR6 Prompt 2. Für die gehärtete Diagnostics-/Contract-Semantik gilt zusätzlich `docs/blue_brain_hypothalamus_surface_diagnostics_contracts_hardening_serie_br6_prompt3_v1.md` als kanonische BR6-Prompt-3-Map. Beide Linien konsolidieren die Hypothalamus-Rolle gegen Runtime, Selection, Context/Reference und die bestehende inter-region architecture, ohne Action-, Execution-, Retry-, Memory-, Compute-, Policy-, Planner-, Agenten- oder Safety-Autorität zu öffnen.

## 1) Kanonische Hypothalamus-Integration-Map

Die Hypothalamus-Integration besteht genau aus diesen Klassen:

1. `hypothalamus input surface`
2. `hypothalamus state surface`
3. `hypothalamus output/advisory surface`
4. `hypothalamus reference surface`
5. `hypothalamus diagnostics/contract map`
6. `blocked/deferred hypothalamus path`
7. `non-canonical/internal-only hypothalamus path`

Diese Klassen sind Schnittstellenlabels, keine neue Meta-Plattform und keine direkte Region-zu-Region-Nachrichtenengine.

## 2) Input Surface

Erlaubte bounded Inputs sind:

- `runtime bounded state signal`: nur als bounded Runtime-Lesart, nicht als Runtime-Mutation.
- `selection bounded state signal`: nur als Selection-/Contract-Diagnostic, nicht als Action-Auswahl.
- `context state-pressure signal`: nur als context-linked state-pressure Caveat.
- `advisory reference signal`: nur als bounded reference/context basis.

Explizit verboten bleiben:

- direkte Tool-/Action-Steuersignale,
- compute-interne Rohzustände,
- direkte Safety-Override-Eingänge,
- implizite Memory-Mutationsinputs.

## 3) State Surface

Der Hypothalamus darf nur bounded Zustände tragen:

- `bounded drive-state advisory-only`,
- `homeostasis/regulation caveat state`,
- `urgency modulation state`,
- `context-linked state-pressure state`,
- `deferred regulation state`,
- `blocked regulation state`,
- `insufficient regulation state`,
- `reference-only regulation state`,
- `non-canonical/internal-only`.

Er darf keine Action-, Execution-, Retry-, Memory-, Compute-, Safety-, Planner-, Policy- oder Agenten-Zustände tragen. Stale, caveated, deferred, insufficient, blocked und reference-only Fälle bleiben als Diagnostics unterscheidbar und werden nicht zu positiver Autorität eskaliert.

## 4) Output / Advisory Surface

Erlaubte bounded Outputs sind:

- `urgency-hint`,
- `state-pressure hint`,
- `bounded regulation caveat`,
- `reference-bounded signal`,
- `blocked/deferred diagnostic`,
- `insufficient diagnostic output`,
- `non-canonical/internal-only`.

Runtime, Selection, Context und Reference dürfen diese Outputs höchstens advisory-only, caveated, deferred, blocked, insufficient oder diagnostic-only gemäß derselben kanonischen Contract-Lesart lesen. Explizit verboten bleiben direct action selection, direct action trigger, direct execution trigger, direct retry trigger, direct memory commit, direct compute invocation und safety override.


## 4a) Diagnostics / Contract Hardening

Die gehärtete BR6-Prompt-3-Map unterscheidet kanonisch:

- `hypothalamus advisory-only diagnostic`,
- `hypothalamus caveated diagnostic`,
- `hypothalamus deferred diagnostic`,
- `hypothalamus blocked diagnostic`,
- `hypothalamus insufficient diagnostic`,
- `hypothalamus diagnostic-only state`,
- `hypothalamus bounded contract signal`,
- `non-canonical/internal-only hypothalamus path`.

`advisory-only ist ein bounded positives Signal` ohne direkte Autorität. `caveated` ist kein starkes positives Signal und darf nicht implizit zu advisory-only aufgewertet werden.

`deferred ist nicht blocked`; deferred bedeutet bounded Aufschub/Zurückstellung. `blocked ist nicht insufficient`; blocked bedeutet begrenzender Contract-/Safety-/Reference-Zustand, während insufficient keine tragfähige bounded Basis hat.

## 5) Runtime / Selection / State-Modulation

Runtime sieht den Hypothalamus ausschließlich als bounded advisory/diagnostic Leser für drive-state, homeostasis/regulation caveat, urgency modulation und context-linked state-pressure. Selection sieht ihn ausschließlich über Selection-/Contract-Diagnostics; urgency modulation darf keine allowed-actions erweitern, keine Action auswählen und keine Execution freigeben.

State-Modulation bleibt bounded und advisory-only: der Hypothalamus darf Hinweise auf state-pressure oder urgency liefern, aber keine Proposal-, Planner-, Agenten-, Queue-, Retry- oder Execution-Autorität erzeugen.

## 6) Reference / Context

Die kanonische Reference-Surface ist eine bounded Reference-/Context-Basis für hypothalamusbezogene state-pressure, regulation caveat und urgency caveat. Reference-only, stale, caveated, blocked und insufficient werden als Diagnostics gelesen. Daraus entsteht keine zweite Referenzwirklichkeit, keine Retrieval-/Consolidation-Linie, kein Memory-Commit und keine automatische Memory-Persistenz.

## 7) Abgrenzung zu anderen Regionen

- `hippocampus_like_region`: context/reference/episode/indexing; Hypothalamus schreibt keine Memory-/Reference-Autorität und liefert nur state-pressure Caveats.
- `amygdala_like_region`: salience/valence/caveat/priority; Hypothalamus erzeugt keine emotionale Valenz, Threat-Semantik oder Safety-Override.
- `thalamus_like_region`: relay/gating/routing; Hypothalamus ist kein Relay-/Routing-Hub und ändert Routing nicht direkt.
- `basal_ganglia_like_region`: action-gating/suppression/channel-selection; Hypothalamus wählt keine Actions und sperrt oder öffnet keine Execution-Kanäle.
- `cerebellum_like_region`: prediction/timing/correction/mismatch; Hypothalamus ersetzt keine Timing- oder Prediction-Engine.

## 8) Inter-region Anschluss

Die bestehende inter-region architecture trägt Hypothalamus nur als bounded adjunct relation:

- Hippocampus ↔ Hypothalamus: `reference-mediated relation`.
- Amygdala ↔ Hypothalamus: `caveated inter-region relation`.
- Thalamus ↔ Hypothalamus: `direct bounded advisory relation`.
- Basal Ganglia ↔ Hypothalamus: `selection-mediated relation`.
- Cerebellum ↔ Hypothalamus: `deferred/not-yet-active relation`.

Diese Relationen sind advisory-only Contract-/Diagnostic-Reads, keine all-to-all Kopplung und kein globales Region-Orchestrierungssystem.

## 9) Modellgrenze und no-direct Guards

Der current mode bleibt `abstract functional current mode`; current model mode remains unchanged. `bounded Kuramoto-like candidate`, `Hodgkin-Huxley simulation-only/diagnostic-only`, `later selective HH deepening` und deferred biologische Details bleiben getrennte spätere Re-Scope-Pfade.

Bewusst out of scope bleiben:

- no direct action trigger,
- no direct action selection,
- no direct execution trigger,
- no direct retry trigger,
- no retry orchestration,
- no direct memory commit,
- no automatic memory persistence,
- no direct compute invocation,
- no safety override,
- no allowed-actions extension,
- no new compute-core work,
- no planner/agent platform,
- no policy/governance platform,
- no retrieval/consolidation/reasoning platform,
- no implicit opening of further anatomical regions.

## 10) BR6 nächste Schritte

1. Hypothalamus diagnostics/contract hardening gegen stale, caveated, blocked, insufficient und reference-only Fälle verdichten.
2. Consumer-facing Runtime/Selection/Context/Reference snapshots für urgency-hint, state-pressure hint und bounded regulation caveat pinnen.
3. Inter-region fixtures für die fünf Hypothalamus-adjunct relations ergänzen, falls externe Consumers die Map maschinenlesbar benötigen.
4. Prüfen, ob Cerebellum ↔ Hypothalamus deferred bleibt oder in einer separaten Timing/Homeostasis-Entscheidung re-scoped wird.
5. Eine spätere Modellvertiefung nur mit expliziter BR6/MD-Spezifikation entscheiden; keine implizite HH- oder Kuramoto-Produktivöffnung.
