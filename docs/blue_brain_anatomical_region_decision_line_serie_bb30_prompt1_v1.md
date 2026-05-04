# Serie BB30 Prompt 1: Anatomical Region Decision Line (UCF-funktional, scope-kontrolliert)

Status: **supporting reference** zur BB29-Maintenance-Authority.

Diese Datei führt **keine** Vollhirnsimulation ein. Sie benennt explizite neuroanatomische Regionen als
UCF-relevante Funktionsanker und legt pro Region einen aktuellen Haupt-Integrationsmodus fest.

## 1) Repo-basierter Anschluss (BB2–BB29)

Aus den stabilen Linien folgt:

- Runtime/Selection bleiben über BB4/BB19 vertraglich führend.
- Context/Memory/Reference bleiben über BB8/BB17 führend.
- Execution-/Reference-Integrität bleibt über BB13/BB14/BB21 führend.
- Bounded Dynamics bleibt über BB10/BB12/BB16 advisory-only.
- Drei aktive Regionenexpansionen (Region 1/2/3) bleiben maintenance-stabil (BB29).

Daher ist die anatomische Benennung **rein funktionsgeleitet** und darf keine neue Autorität erzeugen.

## 2) Kanonische anatomical-region map (BB30-P1)

Nur die für UCF sinnvollen Regionen werden kanonisch benannt:

1. `hippocampus_like_region`
2. `amygdala_like_region`
3. `thalamus_like_region`
4. `cerebellum_like_region`
5. `basal_ganglia_like_region`
6. `hypothalamus_like_region`
7. `prefrontal_executive_control_like_region`

Diese Liste ist bewusst nicht biologisch vollständig.

## 3) Region → UCF-Funktionsrolle → BlueBrain-Anbindung

| Anatomische Region (kanonisch) | UCF-Funktionsrolle | Primäre Anbindung | Systemschwerpunkt | Phase |
|---|---|---|---|---|
| `hippocampus_like_region` | Context binding, episodic/reference indexing, handoff-orientierte Gedächtnisstruktur | BB8/BB17 Context-Memory-Reference | Context/Memory/Reference | early viable |
| `amygdala_like_region` | Salienz-/Caveat-Markierung, Risiko-/Relevanzgewichtung als advisory Signal | BB4/BB19 Runtime-Selection Diagnostik | Runtime + Selection | early viable |
| `thalamus_like_region` | Gating-/Relay-Abstraktion zwischen Selection und Execution-Eligibility | BB19 + BB13/BB21 Contract-Flächen | Selection + Execution-interface | early viable |
| `cerebellum_like_region` | Feinkalibrierte Adaptions-/Timing-Optimierung (nur sinnvoll bei höherer Regelungsdichte) | kein belastbarer maintenance-operativer Primärpfad | Dynamics/Execution-nah, aber nicht nötig | deferred |
| `basal_ganglia_like_region` | Go/No-Go-nahe readiness diagnostics ohne direkte Freigabe | BB13/BB14/BB21 Eligibility-Guards | Execution-interface + Selection-Gating | early viable |
| `hypothalamus_like_region` | Globaler Homeostasis-/Drive-Regelanker; in UCF aktuell nur lose anschließbar | keine robuste operative Kopplung ohne Scope-Risiko | Runtime global-modulatory (nicht benötigt) | deferred |
| `prefrontal_executive_control_like_region` | Priorisierung, Deferral-Sensitivität, kontrollierte Fokus-/Arbitrationsstruktur | BB4/BB19 Kernlinie | Runtime + Selection | early viable |

## 4) Current integration mode (genau ein Hauptmodus pro Region)

| Region | Current integration mode |
|---|---|
| `hippocampus_like_region` | `abstract functional` |
| `amygdala_like_region` | `bounded Kuramoto-like dynamics` |
| `thalamus_like_region` | `abstract functional` |
| `cerebellum_like_region` | `deferred/not-suitable-now` |
| `basal_ganglia_like_region` | `abstract functional` |
| `hypothalamus_like_region` | `deferred/not-suitable-now` |
| `prefrontal_executive_control_like_region` | `abstract functional` |

Hinweise:

- `bounded Kuramoto-like dynamics` ist nur als leichte advisory Modulation zulässig.
- `Hodgkin-Huxley simulation-only/diagnostic-only` bleibt ein gültiger Modus, ist in der aktuellen
  regionalen Hauptzuordnung aber **keiner** Region als Default zugewiesen.

## 5) Kuramoto vs HH vs abstract (regionsbezogene Entscheidungslogik)

- **Abstract functional** ist Default, wenn die UCF-Wirkung bereits über bestehende Vertragsflächen
  (Runtime/Selection/Context/Reference/Execution-Eligibility) präzise abbildbar ist.
- **Bounded Kuramoto-like** ist geeignet für leichte, gekoppelte Synchronie-/Gating-Hinweise
  ohne operative Autorität (hier: amygdala-like Salienzmodulation).
- **HH simulation-only/diagnostic-only** ist nur für tiefe excitability-/spiking-/membran-nahe
  Fragestellungen sinnvoll und bleibt nicht-operativ.
- **Later selective HH deepening** bleibt als spätere Option offen, aber nur evidenzbasiert und
  selektiv pro Region statt globaler Pflicht.
- Nicht jede Region braucht ein Dynamikmodell; HH ist explizit **nicht** der Default.

## 6) Früh-/Spät-Priorität (BB30-P1)

| Region | Prioritätsklasse |
|---|---|
| `hippocampus_like_region` | `early viable` |
| `amygdala_like_region` | `early viable` |
| `thalamus_like_region` | `early viable` |
| `basal_ganglia_like_region` | `early viable` |
| `prefrontal_executive_control_like_region` | `early viable` |
| `cerebellum_like_region` | `deferred/not-suitable-now` |
| `hypothalamus_like_region` | `deferred/not-suitable-now` |

Zusatzmarker für spätere Forschung:

- `simulation-only later`: HH-nahe Diagnostik kann selektiv für `thalamus_like_region` oder
  `basal_ganglia_like_region` erwogen werden, bleibt aber nicht-operativ.
- `later viable`: Kuramoto-Kandidatur kann bei nachgewiesenem Mehrwert auf einzelne Gating-nahe
  Regionen ausgeweitet werden, ohne Default-Wechsel auf HH.

## 7) Guard-/Scope-Absicherung (verbindlich)

Auch mit anatomischen Labels gilt unverändert:

- keine direkte Action-Autorität aus Regionsname oder Regionsmodus,
- keine neue Planner-/Agentenlogik,
- keine Retry-/Queue-/Orchestration-/Policy-Governance-Autorität,
- keine neue Compute-Core-Arbeit,
- keine implizite HH-Produktivöffnung,
- keine Vollsimulations-Erwartung.

## 8) Ergebnislinie

Mit dieser BB30-P1-Linie ist kontrolliert möglich:

1. explizite anatomische Benennung UCF-relevanter Regionen,
2. eindeutige Funktionsrollen-Mapping auf bestehende BlueBrain-Linien,
3. klare, regionsspezifische Integrationsmodi,
4. explizite Trennung von abstract/Kuramoto/HH/deferred,
5. belastbarer Startpunkt für eine spätere selektive anatomische Ausbauphase ohne Scope-Drift.
