# Serie BR6 Prompt 4: Hypothalamus readiness sweep und Expansionsgrenze

Status: BR6 wird hier hart abgeschlossen. `hypothalamus_like_region` ist die **sechste echte anatomische Hirnregion** der UCF-/Blue-Brain-Linie nach Hippocampus, Amygdala, Thalamus, Basal Ganglia und Cerebellum. Die operative Linie bleibt bounded, advisory-only und drive-/homeostasis-/urgency-/state-pressure-nah; sie ist kein biologischer Vollnachbau, keine globale Neurodynamikplattform und keine automatische Hodgkin-Huxley-Produktivintegration.

Diese Datei ist die kanonische BR6-Abschlusslinie. Sie konsolidiert die Rollenkarte, minimale bounded Integration und Surface-/Diagnostics-/Contract-Härtung aus BR6 Prompt 1-3. Die Prompt-3-Datei bleibt die detaillierte diagnostics/contract map; diese Datei zieht die Expansionsgrenze und entscheidet die nächste Roadmap-Richtung, ohne eine zweite Semantik neben Runtime, Selection, Reference/Context, Execution/Reference interaction, bounded Dynamics oder Compute zu erzeugen.

## 1) BR6-expansion-readiness map

| Hypothalamus-Bereich | Readiness-Zustand | Kanonische Bedeutung | Harte Grenze |
| --- | --- | --- | --- |
| input/state/output/reference Surfaces | **stable hypothalamus operational surface** | bounded Lesesignale und bounded Advisory-/Diagnostic-Ausgaben für drive-state, homeostasis/regulation, urgency modulation und context-linked state-pressure sind kanonisch. | keine Action-Steuerung, keine Retry-Orchestrierung, keine Memory-Mutation, keine Compute-Core-Wirkung |
| schwache Reference-/Context-Basis, caveated regulation/state-pressure, partielle urgency basis | **usable with caveats** | nutzbar als schwaches oder caveated Signal, aber nicht als positives Freigabe- oder Autoritätssignal. | keine Aufwertung zu advisory-only, Action-, Retry-, Memory-, Safety-, Policy- oder Compute-Autorität |
| urgency-hint, state-pressure hint, bounded regulation caveat, reference-bounded signal | **advisory-only** | bounded Hinweis für Runtime, Selection, Context und Reference mit identischem canonical contract read. | advisory-only bleibt advisory-only; kein direkter Trigger |
| stale, pending, rejected, invalidated, blocked, insufficient, diagnostic-only, reference-only | **deferred/blocked/insufficient/diagnostic-only/reference-only** | diese Klassen bleiben voneinander getrennt, fail-closed und nicht-promotable. | deferred != blocked, blocked != insufficient, diagnostic-only != advisory-only, reference-only remains read-only and non-actionable |
| Modellmodus | **stable current model mode** | `abstract functional current mode` ist der aktuelle Modellmodus. | `bounded Kuramoto-like candidate`, `Hodgkin-Huxley simulation-only/diagnostic-only`, `HH-later` und `later selective HH deepening` bleiben getrennt und nicht produktiv |
| interne, Expert-, Debug-, Microcircuit- oder nicht-kanonische Restpfade | **non-canonical/internal-only** | nur intern/diagnostisch/reference-only, nicht promotable und nicht operative Hypothalamus-Expansion. | keine Runtime-/Selection-/Reference-/Execution-/Compute-Autorität |

## 2) Harte repo-basierte Gegenprüfung

Stabil und kanonisch sind genau die BR6-Surfaces aus Prompt 2 und die Diagnostics-/Contract-Zustände aus Prompt 3:

1. `hypothalamus input surface`: liest bounded Runtime-/Selection-/Context-/Reference-Signale für drive-state, homeostasis/regulation, urgency modulation und context-linked state-pressure. Nicht kanonisch sind raw biological state, Hormonsystem-/Nukleus-/autonome Rohmodelle, Tool-Control, Retry-Control, Planner-State, Policy-Authority, Safety-Override-Inputs und Compute-Core-Inputs.
2. `hypothalamus state surface`: hält `active urgency modulation state`, `active regulation state`, `context-linked state-pressure state`, `deferred regulation state`, `blocked regulation state`, `insufficient regulation state`, `reference-only regulation state` und `non-canonical/internal-only` getrennt.
3. `hypothalamus output/advisory surface`: liefert `urgency-hint`, `state-pressure hint`, `bounded regulation caveat`, `reference-bounded signal`, `blocked/deferred diagnostic`, `insufficient diagnostic output` oder `non-canonical/internal-only`.
4. `hypothalamus reference surface`: liest Context/Reference bounded für state-pressure, regulation caveat und urgency caveat; stale, caveated, blocked, insufficient, diagnostic-only und reference-only bleiben diagnostisch sichtbar.
5. `hypothalamus diagnostics states`: `hypothalamus advisory-only diagnostic`, `hypothalamus caveated diagnostic`, `hypothalamus deferred diagnostic`, `hypothalamus blocked diagnostic`, `hypothalamus insufficient diagnostic`, `hypothalamus diagnostic-only state`.
6. `hypothalamus contract signals`: `hypothalamus bounded contract signal` und `non-canonical/internal-only hypothalamus path` als harte Consumer-Lesart.
7. `current model mode`: `abstract functional current mode`.

Usable with caveats bleibt schwache, partielle oder caveated Reference-/Context-/Selection-Basis. Advisory-only bleibt bounded positives Signal ohne direkte Autorität. Deferred, blocked, insufficient, diagnostic-only und reference-only bleiben getrennte Limit-/Diagnosezustände. Non-canonical/internal-only Pfade bleiben interne oder Expert-/Debug-/Microcircuit-Restpfade ohne kanonischen Consumer-Read.

## 3) Hypothalamus expansion line

Die Hypothalamus-Expansion ist abgeschlossen als bounded UCF-/Blue-Brain-Anschlussfläche:

- Der Hypothalamus ist in BR6 die nächste echte anatomische Hirnregion und jetzt als sechste Region kanonisch benannt.
- Kanonische Input-/State-/Output-/Reference-Surfaces sind die in Abschnitt 2 genannten Hypothalamus-Surfaces.
- Kanonische Diagnostics- und Contract-States sind die in Abschnitt 2 genannten `hypothalamus ... diagnostic`, `hypothalamus bounded contract signal` und `non-canonical/internal-only hypothalamus path` Tokens.
- Der aktuelle Modellmodus bleibt `abstract functional current mode`.
- Runtime, Selection, Context und Reference dürfen diese Signale nur bounded informieren: urgency/state-pressure/homeostasis/regulation caveats werden gelesen, nicht zu direkter Autorität umgeschrieben.
- Bounded Dynamics dürfen, falls gekoppelt, nur advisory-only Diagnostics liefern; kein Kuramoto- oder Hodgkin-Huxley-Pfad wird produktiv.

Ausdrücklich nicht operativ in BR6 Prompt 4 sind: weitere Hirnregionen, direkte Action-Steuerung, Retry-Steuerung, Planner-/Agentenlogik, Policy-/Governance-Plattform, automatische Memory-Persistenz, Memory-Mutation, Safety-Override-Semantik, Compute-Wirkung und globale Modellplattform.

## 4) Surface-, Diagnostics-, Contract- und Model-Grenzen

Die folgenden Identitäten sind verboten und gelten als Guard-Linie:

- `hypothalamus input surface` is not `hypothalamus state surface`.
- `hypothalamus state surface` is not `hypothalamus output/advisory surface`.
- `hypothalamus output/advisory surface` is not `hypothalamus reference surface`.
- `hypothalamus diagnostics states` are not `hypothalamus contract signals`.
- `hypothalamus bounded contract signal` is not an action, execution, retry, memory, compute, policy, planner, agent, or safety channel.
- `hypothalamus advisory-only diagnostic != hypothalamus caveated diagnostic`.
- `hypothalamus deferred diagnostic != hypothalamus blocked diagnostic`.
- `hypothalamus blocked diagnostic != hypothalamus insufficient diagnostic`.
- `hypothalamus diagnostic-only state != hypothalamus advisory-only diagnostic`.
- `reference-only` remains read-only and non-actionable.
- `abstract functional current mode` is not `bounded Kuramoto-like candidate`.
- `bounded Kuramoto-like candidate` is not `Hodgkin-Huxley simulation-only/diagnostic-only`.
- `Hodgkin-Huxley simulation-only/diagnostic-only` is not `later selective HH deepening` and not productive HH integration.

## 5) No-direct-* und Out-of-scope-Grenzen

BR6 Prompt 4 bestätigt ausdrücklich:

- no direct action execution,
- no direct action trigger,
- no direct action selection,
- no direct execution trigger,
- no retry orchestration or retry trigger,
- no planner/agent logic,
- no Policy-/Governance-Plattform,
- no automatic memory persistence, mutation, or commit,
- no direct memory commit,
- no safety override semantics,
- no allowed-actions extension,
- no direct compute invocation,
- no new compute-core work,
- no seventh anatomical region opened in this step,
- no productive Hodgkin-Huxley integration,
- no global model platform.

Vorhandene DBM-/Microcircuit-/Expert-Pfade bleiben bounded, diagnostic-only, reference-only oder internal-only, solange keine spätere explizite Spezifikation sie anders einordnet. Sie sind keine BR6-Produktivpflicht und keine zweite operative Hypothalamus-Wirklichkeit.

## 6) Absicherung gegen bestehende BlueBrain-Linien

BR6 bleibt an die bestehenden Linien angebunden, ohne sie zu überschreiben:

- **BB2 runtime transition/feedback:** Hypothalamus informiert bounded urgency/state-pressure/regulation diagnostics; keine Runtime-Mutation und keine Retry-Autorität.
- **BB4 selection/priority/deferral:** Selection darf urgency/state-pressure nur advisory/caveated lesen; keine Action-Wahl und keine allowed-actions-Erweiterung.
- **BB8 and BB17 context/memory/reference hardening:** Context/Reference lesen state-pressure/regulation/reference diagnostics; keine Retrieval-/Consolidation-Linie, kein Memory-Commit und keine automatische Persistenz.
- **BB12 bounded dynamics:** Dynamics bleiben advisory-only; Kuramoto-like ist Kandidat, HH bleibt simulation-only/diagnostic-only.
- **BB19 runtime/selection contract line:** Runtime und Selection lesen denselben canonical contract read; keine consumer-spezifische positive Autorität.
- **BB21 execution/reference interaction:** Execution-/Reference-nahe Signale bleiben caveated Diagnostics; kein Execution-Trigger und kein Safety-Override.
- **BR1-BR5 region boundary:** Hippocampus, Amygdala, Thalamus, Basal Ganglia und Cerebellum behalten ihre Rollen; Hypothalamus erzeugt keine semantische Dublette.

Non-canonical/internal-only Pfade dürfen nicht als zweite operative Regionenwirklichkeit erscheinen.

## 7) Compute-Core-Abschlusslinie

BR6 eröffnet den Real Compute Stack nicht erneut. Compute bleibt:

- finale Compute-Linie,
- outward-facing Contracts,
- maintenance-only Core.

Hypothalamus-Signale dürfen Compute nicht direkt aufrufen, keine Compute-internen Rohzustände als kanonische Hypothalamus-Inputs verwenden und keine neue Compute-Core-Arbeit auslösen.

## 8) Entscheidung: weitere Hirnregion oder System-Audit/Consolidation

Priorisiert wird genau eine nächste Richtung: **System-Audit/Consolidation-Pass**.

Technische Begründung:

1. Sechs anatomische Regionen sind jetzt vorhanden; der höhere Hebel liegt nicht in einer siebten Region, sondern in der Prüfung, ob Runtime/Selection/Reference/Execution/Dynamics alle sechs Regionen mit identischer bounded Contract-Lesart konsumieren.
2. Ein System-Audit/Consolidation-Pass reduziert Scope-Risiko, weil er no-direct-* Guards, diagnostic-state Trennung, non-canonical/internal-only Pfade, inter-region Beziehungen und Compute-Maintenance-Grenzen über die vorhandene Basis prüft, ohne neue anatomische Semantik einzuführen.
3. Eine weitere Hirnregion muss warten, weil sie vor der Konsolidierung die Grenzfläche zwischen advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only und reference-only weiter vergrößern würde.
4. HH-lastigere oder schwerere Modellschritte warten, weil `abstract functional current mode` stabil bleibt und HH weiterhin `simulation-only/diagnostic-only` ist.

Damit ist BR6 abgeschlossen: Hypothalamus ist operativ bounded und advisory/caveated nutzbar; weitere anatomische Expansion ist bewusst gestoppt, bis ein gezielter System-Audit/Consolidation-Pass die mehrregionale Blue-Brain-Basis konsistent stabilisiert.
