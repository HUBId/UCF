# Serie BB24 Prompt 2: Region-Integration-Mode Line (kontrollierter Re-Scope)

Status: **kanonische Integrationsmodus-Linie für BB24 Prompt 2**.

Diese Datei erweitert BB24 Prompt 1 **nicht** in Richtung Vollsimulation. Sie schärft nur, wie
jede kanonische Regionenklasse an bestehende BlueBrain-Linien andocken darf.

## 1) Repo-basierter Anschluss (hart an bestehende Linien)

| Regionenklasse (BB24-P1) | Primäre bestehende Linie | Operative Oberfläche (heute) | Zulässiger Integrationsmodus |
|---|---|---|---|
| Attention/Selection-related | BB4 + BB19 Runtime/Selection Contract | Priority/Deferred/Blocked + advisory contract diagnostics | **abstract_functional_integration** |
| Memory/Context-related | BB8 + BB17 Context/Memory/Reference | Canonical reference consumption, context/memory handoff boundaries | **abstract_functional_integration** |
| Action-readiness/Execution-interface-related | BB13/BB14 + BB21 Execution/Reference Interaction | Execution-eligibility, reference validity, execution-integrity guards | **abstract_functional_integration** |
| Bounded dynamics modulation-related | BB12 (+ BB16 hardening) | Kuramoto bounded modulation hints/caveats (runtime/selection feedback) | **bounded_advisory_only_modulation_integration** |
| Simulation-only/Diagnostic-only | BB10/BB11 diagnostics + HH simulation lane | Diagnostic classes, simulation reports, no runtime authority | **simulation_only_diagnostic_only_integration** |
| Deferred/Not-suitable-now | BB20/BB21 scope boundaries | Explicit unavailable/deferred lane, no operational hook | **deferred_not_suitable_now** |

Zusätzlich gilt als Boundary-Klasse:

- **non_canonical_internal_only_region_path**: intern/expert-only Pfade, nie kanonische operative Autorität.

## 2) Kanonische region-integration-mode map

Kanonische Modi für BB24 Prompt 2:

1. `abstract_functional_integration`
2. `bounded_advisory_only_modulation_integration`
3. `simulation_only_diagnostic_only_integration`
4. `deferred_not_suitable_now`
5. `non_canonical_internal_only_region_path`

Es werden **keine** zusätzlichen Meta-Modi oder Plattform-Layer eingeführt.

## 3) Abstract functional integration (hart)

Gilt für:

- attention/selection-related,
- memory/context-related,
- action-readiness/execution-interface-related.

Verbindliche Grenzen:

- nur funktionale Spiegelung auf Runtime/Selection/Context/Reference/Execution-Flächen,
- keine Pflicht zu biologischen Detailmodellen,
- keine implizite Dynamics-Autorität,
- keine direkte Action-Autorität,
- keine implizite Memory-Persistenz.

## 4) Bounded advisory-only modulation integration (hart)

Gilt nur für:

- bounded dynamics modulation-related class.

Verbindliche Grenzen:

- nur innerhalb bereits bestehender bounded advisory-only Dynamics-Linie,
- Einfluss bleibt advisory-only (Hint/Caveat/Diagnostic),
- keine direkte Action-/Retry-/Compute-Autorität,
- keine direkte Memory-Commit-Autorität,
- keine Safety-Override-Semantik.

## 5) Simulation-only / diagnostic-only integration (hart)

Gilt für:

- simulation-only/diagnostic-only class (inkl. HH-nahe Pfade in aktueller Linie).

Verbindliche Grenzen:

- keine operative Runtime-/Selection-/Execution-Autorität,
- nur Diagnose-/Simulationssignale,
- keine stillschweigende Promotion zu Modulation mit operativer Wirkung,
- dient ausschließlich der späteren, evidenzbasierten Modellentscheidung.

## 6) Deferred / not-suitable-now (bewusst stehen lassen)

Gilt für Klassen ohne tragfähige operative Linie.

Begründung:

- fehlende belastbare Runtime/Selection/Execution-Kopplung ohne Autoritätserweiterung,
- fehlende Notwendigkeit unter maintenance-only Compute-Core-Rahmen,
- fehlender Nachweis, dass operativer Mehrwert die Scope-Risiken übersteigt.

Deferred bedeutet in BB24 Prompt 2:

- kein aktiver Implementierungsauftrag,
- keine stillschweigende Roadmap-Verpflichtung,
- keine Hintertür über non-canonical/internal-only Pfade.

## 7) Guard-Raster gegen Autoritätsausweitung

Für **alle** Regionenklassen verbindlich:

- keine direkte Execution-Autorität aus Regionenklassifikation,
- keine Überschreibung von Safety-/Execution-Integrity-Grenzen,
- keine erzwungene Memory-Persistenz,
- keine Reaktivierung von deferred/not-suitable-now über indirekte Pfade,
- keine Promotion von non_canonical_internal_only_region_path zu kanonischer Runtime-Wirkung.

## 8) Zuordnung zu Runtime / Selection / Memory / Dynamics / Execution

| Regionenklasse | Runtime | Selection | Memory/Context/Reference | Dynamics | Execution |
|---|---|---|---|---|---|
| Attention/Selection-related | advisory runtime contract signals | priority/deferral abstraction | reference-informed context read | keine Pflicht | nur indirekte eligibility-Vorstufe |
| Memory/Context-related | runtime caveat/context usage | selection reference basis | kanonische context/memory/reference Flächen | optional diagnostisch | keine direkte execution-Freigabe |
| Action-readiness/Execution-interface-related | runtime readiness diagnostics | selection gating basis | execution-reference boundary basis | optional diagnostisch | execution-eligibility Spiegelung ohne Autoritätserweiterung |
| Bounded dynamics modulation-related | runtime caveat modulation hint | selection advisory hint | nur referenzbasierte Inputs | Kuramoto advisory-only | keine direkte action-Ausführung |
| Simulation-only/Diagnostic-only | diagnostic feedback only | diagnostic feedback only | diagnostic trace refs only | HH/Kuramoto simulation-only | keine operative Kopplung |
| Deferred/Not-suitable-now | unavailable/deferred marker | unavailable/deferred marker | kein operativer handoff | keine operative dynamics lane | keine operative execution lane |

## 9) Kuramoto-vs-HH Vorbereitung (regionsbezogen, nicht global)

- Regionenklassen mit `bounded_advisory_only_modulation_integration` bleiben Kuramoto-primär.
- Regionenklassen mit `simulation_only_diagnostic_only_integration` bleiben HH-/Simulations-kandidaten.
- `abstract_functional_integration` benötigt per Default **kein** Dynamikmodell.
- `deferred_not_suitable_now` bleibt ohne Modellentscheidung.

Damit ist die spätere Wahl Kuramoto vs Hodgkin-Huxley **regionsbezogen vorbereitbar**, ohne
jetzt eine produktive HH-Integration auszulösen.

Die konkrete kanonische Entscheidung pro Regionenklasse wird in
`blue_brain_region_model_decision_serie_bb24_prompt3_v1.md` festgezogen.

## 10) Out-of-scope (harte Grenze bleibt)

- kein vollständiger Blue-Brain-Nachbau,
- keine Vollhirn-/Vollregionssimulation,
- keine neue globale Neurodynamikplattform,
- keine Planner-/Agenten-/Policy-/Governance-Plattform,
- keine Retry-/Queue-/Orchestration-Plattform,
- keine neue Compute-Core-Entwicklung,
- keine neue allowed-actions-Erweiterung,
- keine implizite Memory-Persistenz,
- keine direkte produktive Hodgkin-Huxley-Integration.
