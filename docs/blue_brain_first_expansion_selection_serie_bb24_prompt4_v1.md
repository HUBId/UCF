# Serie BB24 Prompt 4: First-Expansion-Selection Line (genau eine schmale Regionenklasse)

Status: **kanonische First-Expansion-Selection-Linie für BB24 Prompt 4**.

Diese Datei trifft **genau eine** reale, schmale Priorisierungsentscheidung innerhalb der bereits
festgezogenen BB24-P1/P2/P3 Leitplanken. Es wird keine Mehrfach-Expansion und keine neue
Neurodynamik-/Planner-/Governance-Plattform eingeführt.

## 1) Kandidatenprüfung: Hebel vs. Integrationsrisiko

Bewertung entlang bestehender Linien (BB2/BB4/BB8/BB12/BB13/BB14/BB17/BB19/BB21):

| Regionenklasse (BB24-P1) | Hebel für UCF (jetzt) | Integrationsnähe zu bestehender Linie | Cross-line-/Autoritätsrisiko | Modellschwere | Scope-Drift-Risiko | Ergebnis |
|---|---|---|---|---|---|---|
| Attention/Selection-related | hoch: direkte Wirkung auf priority/deferral/runtime-coupling | sehr hoch (BB4 + BB19 bereits kanonisch) | niedrig bei abstract-only | niedrig | niedrig | **first expansion candidate** |
| Memory/Context-related | hoch: Kontext-/Referenzqualität | hoch (BB8 + BB17) | mittel (Risiko impliziter Persistenz-/Memory-Ausweitung) | niedrig (abstract) | mittel | **viable but not first** |
| Action-readiness/Execution-interface-related | mittel-hoch: eligibility-Konsistenz | hoch (BB13/BB14 + BB21) | mittel-hoch (Nähe zu Execution-Autoritätsgrenzen) | niedrig (abstract) | mittel-hoch | **viable but not first** |
| Bounded dynamics modulation-related | mittel: advisory modulation | mittel-hoch (BB12/BB16) | mittel (falsch integrierte Modulation kann Scope weiten) | mittel (Kuramoto) | mittel | **later-phase candidate** |
| Simulation-only/Diagnostic-only | niedrig operativ, hoch diagnostisch | mittel (BB10) | niedrig operativ, aber hoher Implementierungsaufwand | hoch (HH/sim-heavy) | niedrig operativ, hoch Aufwand | **simulation-only/deferred candidate** |
| Deferred/Not-suitable-now | derzeit niedrig | niedrig | hoch bei vorzeitiger Öffnung | variabel | hoch | **simulation-only/deferred candidate** |
| non_canonical_internal_only_region_path | keine kanonische operative Rolle | n/a (Boundary lane) | hoch bei Promotion | n/a | hoch | **non-canonical/internal-only path** |

## 2) Kanonische first-expansion selection map

Verbindliche Zustände für BB24-P4:

1. `first_expansion_candidate`
2. `viable_but_not_first`
3. `later_phase_candidate`
4. `simulation_only_deferred_candidate`
5. `non_canonical_internal_only_path`

Zuordnung (BB24-P4):

- Attention/Selection-related → `first_expansion_candidate`
- Memory/Context-related → `viable_but_not_first`
- Action-readiness/Execution-interface-related → `viable_but_not_first`
- Bounded dynamics modulation-related → `later_phase_candidate`
- Simulation-only/Diagnostic-only → `simulation_only_deferred_candidate`
- Deferred/Not-suitable-now → `simulation_only_deferred_candidate`
- non_canonical_internal_only_region_path → `non_canonical_internal_only_path`

## 3) Explizite Auswahlkriterien (kompakt, technisch)

Die First-Expansion-Auswahl wird an folgende Kriterien gebunden:

- **Funktionaler UCF-Hebel:** verbessert direkte Qualität der bestehenden selection/priority/deferral-Fläche.
- **Integrationsnähe:** nutzt vorhandene BB4/BB19 Runtime-/Selection-Verträge ohne neue Core-Flächen.
- **Bounded-Verträglichkeit:** bleibt vollständig mit advisory-only, non-authoritative Linien kompatibel.
- **Geringe Autoritätsgefahr:** keine direkte Action-/Retry-/Memory-/Compute-Autorität.
- **Überschaubare Tiefe:** abstract-first Einhängung ohne HH- oder Voll-Dynamics-Druck.

## 4) Genau eine priorisierte Regionenklasse

**Gewählte erste reale Expansion:** `Attention/Selection-related`.

Warum genau diese zuerst:

- maximaler unmittelbarer Hebel auf bestehende Priorisierungs-/Deferral-Qualität,
- beste Anschlussfähigkeit an bereits gehärtete Selection-/Runtime-Contracts,
- geringster Architekturbruch durch `abstract_only`-Pfad,
- klare Trennung von Execution-Autorität und Safety-Entscheidung bleibt erhalten.

Modus der ersten Expansion:

- **abstract-first** (nicht dynamics-first).
- Kuramoto-Lane bleibt davon unberührt und separat begrenzt.
- HH-lastige Klassen bleiben weiterhin simulation-only/deferred, da für die erste reale Expansion
  Aufwand/Risiko unverhältnismäßig und operativer Hebel nicht primär.

## 5) Bewusst nachrangige Einordnung aller übrigen Kandidaten

- `Memory/Context-related`: **viable but not first** (operativ wertvoll, aber höheres Risiko für
  implizite Memory-Scope-Ausweitung).
- `Action-readiness/Execution-interface-related`: **viable but not first** (starker Nutzen,
  aber zu nah an Execution-Integritäts- und Autoritätsgrenzen für den ersten Schritt).
- `Bounded dynamics modulation-related`: **later-phase candidate** (geeignet, aber nicht vor
  abstract-first Auswahlabschluss).
- `Simulation-only/Diagnostic-only` + `Deferred/Not-suitable-now`:
  **simulation-only/deferred candidate** (kein operativer First-Step).
- `non_canonical_internal_only_region_path`: bleibt strikt intern.

"Nicht zuerst" ist hier ausdrücklich **kein Verwerfungsentscheid**, sondern reine
Sequenzierungsentscheidung unter Scope-Kontrolle.

## 6) Guard-/Scope-/Safety-Grenzen (verbindlich)

Auch für den First-Candidate unverändert hart:

- keine direkte Action-/Retry-/Memory-/Compute-Autorität,
- keine Safety-Override-Semantik,
- keine Promotion deferred/simulation-only/non-canonical Klassen,
- keine Mehrklassen-Expansion im selben Schritt,
- keine Reaktivierung produktiver HH-Integration.

## 7) Minimale Einhängungsrichtung für den nächsten Schritt

Nächste minimale Andockrichtung (ohne Vollimplementierung in diesem Prompt):

- Andocken an BB4/BB19-Linie:
  - Eingänge: bestehende runtime/selection-signals (priority, deferral, blocked/deferred diagnostics,
    reference-informed caveats).
  - Ausgänge: nur advisory selection-attention weighting hints und contract diagnostics.
- Explizit nicht berühren:
  - execution write paths,
  - memory persistence/commit authority,
  - compute-core execution semantics,
  - retry/orchestration/governance surfaces.

## 8) Out-of-scope (unverändert hart)

- kein voller Blue-Brain-Ausbau,
- keine Mehrfachauswahl mehrerer Regionenklassen,
- keine Vollhirn-/Vollsimulationsplattform,
- keine neue Compute-Core-Arbeit,
- keine Planner-/Agenten-/Policy-/Governance-/Orchestration-Plattform,
- keine neue allowed-actions-Erweiterung,
- keine direkte produktive HH-Integration.
