# Serie BB24 Prompt 1: UCF-relevante Brain-Region-Map (funktional, kontrolliert)

Status: **kanonische Re-Scope-Karte für BB24 Prompt 1**.

Diese Datei führt **keine** Vollsimulation ein. Sie präzisiert nur, welche funktionalen
Regionenklassen für UCF-BlueBrain entlang der bestehenden Runtime/Selection/Context/Memory/
Dynamics/Execution-Linien sinnvoll sind.

## 1) Repo-basierte Ausgangslage (BB2–BB23 Anschluss)

Bereits stabil in der operativen Linie:

- Runtime-Grundlinie inkl. selection/attention-/priority-/deferral-Kopplung.
- Context/Memory/Reference-Flächen inkl. canonical reference consumption und cross-line Handoffs.
- bounded advisory-only dynamics (Kuramoto-Minimallinie) ohne direkte Autorität.
- minimale echte Execution inkl. execution-eligibility und execution/reference interaction.

Daraus folgt: Eine Regionssprache ist nur dort sinnvoll, wo sie bestehende UCF-Funktionsflächen
präziser klassifiziert, **nicht** wo sie neue Plattformen oder Autoritäten eröffnet.

## 2) Kanonische Regionenklassen für UCF

Die folgende Klassifikation gilt als BB24-P1-kanonisch:

1. **Attention/Selection-related class**
2. **Memory/Context-related class**
3. **Action-readiness/Execution-interface-related class**
4. **Bounded dynamics modulation-related class**
5. **Simulation-only/Diagnostic-only class**
6. **Deferred/Not-suitable-now class**

Keine weitere biologische Vollständigkeitsliste ist in diesem Schritt erforderlich.

## 3) Funktionale Zuordnung UCF-relevanter Regionen/Subsysteme

Die Zuordnung ist absichtlich **funktional** (UCF-Rolle) statt anatomisch-vollständig:

| Kandidat-Region / Subsystemklasse | UCF-Funktionale Rolle | Primäre BlueBrain-Anbindung | Integrationsmodus |
|---|---|---|---|
| Präfrontale Kontroll-/Aufmerksamkeits-Systeme (PFC-ähnlich) | Priorisierung, Fokusgewichtung, Deferral/Arbitration-Sensitivität | Selection/Attention + Runtime-Gating | **abstract functional model** |
| Salienz-/Konflikt-Monitoring (ACC/Insula-ähnlich) | Caveat-Risiko-Signale, Konfliktmarkierung, Kontrollaufmerksamkeits-Trigger | Runtime↔Selection Contract + Diagnostics | **abstract functional model** |
| Hippocampus-/Medial-Temporal-ähnliche Gedächtnisfunktionen | Context binding, episodic/reference-indexing, handoff eligibility | Context/Memory/Reference Linien | **abstract functional model** |
| Thalamisch-striatale Gating-/Readiness-Funktionen | Action-readiness Kandidatenfilter, execution-eligibility-Vorstruktur | Selection→Execution Interface (indirekt) | **abstract functional model** |
| Basalganglien-/Motor-Readiness-Analogie | Go/No-Go-artige readiness diagnostics (ohne direkte Action-Freigabe) | Execution-eligibility diagnostics + safety precheck binding | **abstract functional model** |
| Leichtgewichtige Oszillations-/Synchronie-Subsysteme | bounded Modulationshinweise für Selection/Runtime-Caveat | Kuramoto advisory-only Linie | **bounded advisory-only dynamics candidate** |
| Detaillierte Membran-/Spiking-Subsysteme | hochauflösende Neurodynamik, Forschungs-/Erklärungszwecke | keine operative Primärankopplung | **simulation-only / diagnostic-only candidate** |
| Cerebellar feinmotorische/adaptive Vollkopplung, Ganzhirn-Netzwerksimulation etc. | derzeit kein klarer UCF-Mehrwert unter Maintenance-/Bounded-Rahmen | n/a | **deferred / not-suitable-now** |

## 4) Spiegelung gegen bestehende BlueBrain-Linien

- **Runtime andocking:** salienz-/caveat-nahe Klassen und readiness-nahe abstrakte Modelle nur als
  diagnostische bzw. gate-begleitende Signale.
- **Selection andocking:** attention/priority/deferral-nahe Klassen als abstrakte Funktionsmodelle
  innerhalb bestehender Auswahl- und Vertragsgrenzen.
- **Context/Memory andocking:** hippocampus-/binding-nahe Klassen nur als Referenz-/Handoff-
  und Kontextstrukturierung, keine neue Persistenzautorität.
- **Bounded dynamics andocking:** ausschließlich begrenzte Kuramoto-nahe Modulation als advisory-only.
- **Execution andocking:** nur indirekt über bestehende eligibility-/safety-/reference-Surfaces,
  keine direkte regionale Action-Autorität.

## 5) Integrationsmodus pro Klasse (verbindlich für BB24-P1)

- Attention/Selection-related → **abstract functional model**.
- Memory/Context-related → **abstract functional model**.
- Action-readiness/Execution-interface-related → **abstract functional model**.
- Bounded dynamics modulation-related → **bounded advisory-only dynamics candidate**.
- Simulation-only/Diagnostic-only → **simulation-only / diagnostic-only candidate**.
- Deferred/Not-suitable-now → **deferred / not-suitable-now**.

## 6) Regionsbezogene Einordnung Kuramoto vs. Hodgkin-Huxley

- **Kuramoto:** regionsbezogen der primäre leichte Kandidat für bounded, advisory-only
  Modulationshinweise (v. a. selection/runtime-caveat-nahe Klassen).
- **Hodgkin-Huxley:** regionsbezogen nur dort diskutabel, wo ein echter Zusatznutzen über
  Simulation/Diagnostik hinaus nachweisbar wäre; in der aktuellen Linie weiterhin nicht produktiv.
- Nicht jede Region braucht ein Dynamikmodell.
- Nicht jede UCF-relevante Region braucht Hodgkin-Huxley.
- Kein globaler Rollout von Kuramoto oder Hodgkin-Huxley in diesem Schritt.

## 7) Out-of-scope (harte Grenze)

Explizit außerhalb BB24 Prompt 1:

- kein voller Blue-Brain-Nachbau,
- keine vollständige Hirnregionssimulation,
- keine globale Neurodynamikplattform,
- keine neue Agenten-/Planner-/Policy-/Governance-Plattform,
- keine Compute-Core-Neuentwicklung,
- keine direkte Action-/Safety-/Memory-Autorität aus Regionenmodellen,
- keine direkte Hodgkin-Huxley-Produktivintegration.

## 8) Ergebnis für die nächste BB24-Expansion

Diese Karte ermöglicht als nächsten kontrollierten Schritt:

1. regionsklassen-spezifische statt globale Dynamics-Entscheidung,
2. Kuramoto-/HH-Entscheidung pro funktionaler Klasse,
3. saubere Scope-Kontrolle gegen Autoritätserweiterung,
4. konsistente Anbindung an bestehende Runtime/Selection/Context/Memory/Execution-Verträge.
