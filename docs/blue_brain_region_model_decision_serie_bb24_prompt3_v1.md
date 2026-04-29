# Serie BB24 Prompt 3: Region-Model-Decision Line (Kuramoto vs. Hodgkin-Huxley)

Status: **kanonische Modellentscheidungs-Linie für BB24 Prompt 3**.

Diese Datei baut auf BB24 Prompt 1 (Regionenklassen) und BB24 Prompt 2 (Integrationsmodi) auf.
Sie führt **keine** neue Dynamikplattform ein, sondern entscheidet pro Regionenklasse:

- `kuramoto_suitable_bounded`
- `hodgkin_huxley_simulation_only_diagnostic_only`
- `abstract_only`
- `deferred_not_suitable_now`
- `non_canonical_internal_only_model_path`

## 1) Region-Model-Decision Map (kanonisch)

| Regionenklasse (BB24-P1/P2) | Integrationsmodus (BB24-P2) | Modellentscheidung (BB24-P3) | Technische Begründung |
|---|---|---|---|
| Attention/Selection-related | `abstract_functional_integration` | `abstract_only` | Diese Klasse ist primär runtime-/selection-vertraglich; bestehende BB4/BB19-Linien tragen abstrahierte Prioritäts-/Deferral-Logik ohne Bedarf für ein explizites Dynamikmodell. |
| Memory/Context-related | `abstract_functional_integration` | `abstract_only` | Die operative Hauptwirkung liegt auf canonical reference consumption und handoff boundaries (BB8/BB17), nicht auf gekoppelter Oszillator- oder Spiking-Dynamik. |
| Action-readiness/Execution-interface-related | `abstract_functional_integration` | `abstract_only` | Execution-eligibility-/reference-integrity Grenzen (BB13/BB14/BB21) bleiben führend; ein Dynamikmodell würde aktuell keinen sicheren Zusatznutzen liefern und Scope-Risiko erhöhen. |
| Bounded dynamics modulation-related | `bounded_advisory_only_modulation_integration` | `kuramoto_suitable_bounded` | Kuramoto passt als leichtgewichtige, gekoppelte, deterministische Modulationsschicht auf bestehender BB12/BB16-Linie (Hint/Caveat/Diagnostic), ohne direkte Action-/Execution-Autorität. |
| Simulation-only/Diagnostic-only | `simulation_only_diagnostic_only_integration` | `hodgkin_huxley_simulation_only_diagnostic_only` | HH bleibt der schwergewichtige Kandidat für biophysikalisch-/spiking-nahe Simulation und Diagnostik (BB10), ohne operative Runtime-/Selection-/Execution-Promotion. |
| Deferred/Not-suitable-now | `deferred_not_suitable_now` | `deferred_not_suitable_now` | Es fehlt weiterhin eine belastbare operative Kopplung ohne Autoritätserweiterung; daher keine künstliche Modellzuweisung auf Kuramoto oder HH. |

Boundary-Klasse (klasseübergreifend):

- `non_canonical_internal_only_region_path` → `non_canonical_internal_only_model_path`

Diese Boundary-Klasse bleibt intern/expert-only und darf keine kanonische operative Autorität erhalten.

## 2) Kuramoto-Suitability (regionsbezogen, begrenzt)

`kuramoto_suitable_bounded` gilt **nur** für die Regionenklasse
`Bounded dynamics modulation-related`.

Verbindlich:

- Kuramoto bleibt auf advisory-only Modulation begrenzt (Hints/Caveats/Diagnostics).
- Kein direkter Action-/Execution-/Retry-/Memory-/Compute-Write-Pfad.
- Kein Safety-Override, keine Policy-/Governance-Autorität.
- Keine implizite Ausweitung auf abstract-only oder deferred Klassen.

## 3) Hodgkin-Huxley-Suitability (regionsbezogen, begrenzt)

`hodgkin_huxley_simulation_only_diagnostic_only` gilt **nur** für die Regionenklasse
`Simulation-only/Diagnostic-only`.

Verbindlich:

- HH bleibt simulation-only / diagnostic-only.
- HH ist nicht der Default-Standard für andere Regionenklassen.
- Keine stillschweigende produktive Runtime-/Selection-/Execution-Aufwertung.

## 4) Bewusst modellfreie Klassen (kein Defizit)

`abstract_only` ist verbindlich für:

- Attention/Selection-related
- Memory/Context-related
- Action-readiness/Execution-interface-related

Das ist eine **präzise Architekturentscheidung**: funktionale Kopplung über bestehende
Runtime/Selection/Reference/Execution-Verträge statt unnötiger Dynamikmodellpflicht.

## 5) Deferred bleibt deferred

`deferred_not_suitable_now` bleibt verbindlich für `Deferred/Not-suitable-now`.

Fehlende Voraussetzung für spätere Öffnung (heutiger Stand):

- belastbarer, evidenzbasierter Mehrwert gegenüber abstract/bounded Linien,
- klarer Nachweis ohne Autoritätserweiterung,
- klare Runtime-/Execution-Sicherheitsverträglichkeit unter maintenance-only Compute-Rahmen.

Dies ist **keine** implizite Kurzfrist-Roadmap.

## 6) Guard-Abgleich gegen Execution/Safety/Reference-Grenzen

Die Modellentscheidung ändert die bestehenden Guard-Grenzen nicht:

- Kuramoto bleibt bounded advisory-only.
- HH bleibt simulation-only / diagnostic-only.
- Keine Regionenklasse erhält direkte Action-/Retry-/Memory-/Compute-Autorität.
- Keine Safety-Override-Semantik.
- Keine Promotion non-canonical/internal-only Pfade zu kanonischer Runtime-Wirkung.

## 7) Out-of-scope (hart)

- kein vollständiger Blue-Brain-Nachbau,
- keine Vollhirn-/Vollregionssimulation,
- keine neue globale Neurodynamikplattform,
- keine neue Compute-Core-Arbeit,
- keine Planner-/Agenten-/Policy-/Governance-/Orchestration-Plattform,
- keine neue allowed-actions-Erweiterung,
- keine direkte produktive Hodgkin-Huxley-Integration,
- keine Kuramoto-Autorität über bounded advisory-only hinaus.
