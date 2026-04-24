# Serie BB10 Prompt 1: Neural-Dynamics Candidate Map gegen Runtime/Memory/Selection/Action-Safety

Status: Dieser Schritt liefert eine **repo-basierte Kandidatenkartierung** für neurodynamische Modelle im Anschluss an BB2-BB9. Er führt **keine produktive Neurodynamics-Engine** ein.

## 1) Repo-basierte Ausgangslage

- BB2-BB9 definieren die kanonischen Linien für Runtime, Context/Memory, Selection/Attention, Candidate/Proposal und Action-Safety.
- In den bisherigen BB-Linien sind neurodynamische Spezialmodelle (z. B. Hodgkin-Huxley/Kuramoto) wiederholt als **intentionally deferred** markiert.
- Der Compute-Kern bleibt auf der finalen Exit-Linie (`submit -> compute_canonical -> result/fault/status -> execution_snapshot`) und ist maintenance-only.

Konsequenz für BB10 Prompt 1: Es wird eine **Integration-Candidate-Map** aufgebaut, kein Plattform-Buildout.

## 2) Canonical Neural-Dynamics Candidate Classes

Die BB10-Kandidaten werden in genau folgende Klassen getrennt:

1. `simulation-only candidate`
2. `diagnostic-only dynamics candidate`
3. `runtime-modulating candidate`
4. `selection/attention-modulating candidate`
5. `memory/context-modulating candidate`
6. `action-safety-relevant candidate`
7. `not suitable now`
8. `non-canonical/internal-only dynamics path`

Diese Klassen sind explizit getrennt, um Semantik-Kollaps (z. B. Diagnose == Ausführung) zu verhindern.

## 3) Hodgkin-Huxley Einordnung

**Einordnung jetzt:** primär `simulation-only candidate`, optional später `diagnostic-only dynamics candidate`.

Technische Begründung im aktuellen Repo-Schnitt:

- Hohe Zustandsdichte und Parametrisierung passt nicht direkt zur minimalen BB2-BB9 Runtime-/Selection-/Memory-/Action-Safety-Linie.
- Für produktive Runtime-Modulation wäre zuerst ein stark begrenzter, deterministischer Down-Mapping-Pfad nötig.
- Sinnvoller aktueller Nutzen: offline Kalibrierung, Drift-/Instabilitätsanalyse, Diagnoseableitungen.

Benötigte IO (begrenzt):

- Inputs: Runtime-Phasen/Transitionen, Context/Memory-Referenzen, Selection-/Safety-Diagnostik.
- Outputs: nur Diagnose-/Caveat-Signale oder Offline-Simulationsevidenz.

Nicht erlaubt:

- keine direkte Action-Ausführung,
- kein direkter Memory-Commit,
- keine Compute-Core-Mutation,
- keine direkte Policy-Entscheidung.

## 4) Kuramoto Einordnung

**Einordnung jetzt:** primär `selection/attention-modulating candidate`, sekundär `diagnostic-only dynamics candidate`.

Technische Begründung im aktuellen Repo-Schnitt:

- Kuramoto passt als leichtgewichtiges Synchronisationsmodell gut zu BB4 Selection/Attention.
- Kopplung kann auf Kandidatengruppen bzw. Deferral-/Priority-Kontext erfolgen, ohne Runtime- oder Policy-Klassen umzuschreiben.
- Eignet sich als erster begrenzter Dynamikpfad besser als Hodgkin-Huxley.

Benötigte IO (begrenzt):

- Inputs: Selection-State, Candidate-Set-Metadaten, Context-/Evidence-Qualität, Runtime-/Safety-Caveats.
- Outputs: synchrony/phase signal, candidate weight/caveat, deferral confidence modulation.

Nicht erlaubt:

- keine direkte Candidate-Akzeptanz,
- keine Proposal-Promotion,
- keine Action-Ausführung,
- kein Memory-Commit.

## 5) Allowed Input/Output Surfaces (für realistische Kandidaten)

### Allowed inputs

- runtime state
- context reference
- memory reference
- selection state
- evidence status
- safety state

### Allowed outputs

- modulation signal
- diagnostic signal
- candidate weight/caveat
- synchrony/phase signal
- runtime caveat

### Explicitly disallowed outputs

- direct action execution
- direct memory commit
- direct compute-core mutation
- direct policy decision

## 6) Harte Boundary-Regeln

Neural-dynamics candidates:

- öffnen den Compute-Core nicht,
- committen Memory nicht automatisch,
- führen keine Action aus,
- überschreiben die Safety Boundary nicht,
- dürfen nur modulieren/diagnostizieren bzw. Candidate-States begrenzt beeinflussen.

Internal/expert-only dynamics lanes bleiben non-canonical, bis ein explizites Down-Mapping auf outward references vorliegt.

## 7) Priorisierte nächste Integrationsrichtung

**Priorität 1:** Kuramoto-basierte `selection/attention-modulating` Pilot-Lane (rein modulativ + diagnostisch).

- Hebel: direkte Verbesserung in BB4 Selection/Attention ohne neue Execution- oder Commit-Macht.
- Risiko-/Kostenprofil: niedriger als biophysikalisch detaillierte Modelle.

**Priorität 2:** Diagnostic-only dynamics probes als Runtime/Memory/Action-Safety Caveat-Support.

- Hebel: bessere Caveat-Qualität über BB2/BB8/BB9, ohne Boundary-Verschiebung.

**Nachrangig:** Hodgkin-Huxley bleibt vorerst simulation-only.

- Begründung: hoher State-/Compute-Aufwand und aktuell kein sauberer minimaler produktiver Andockpunkt ohne Scope-Drift.

## 8) Konsistenz-Checkliste für BB10 Prompt 1

- Kandidatenklassen sind unterscheidbar.
- Hodgkin-Huxley/Kuramoto werden nicht als produktiv integriert behauptet.
- Dynamics candidates lösen keine action execution aus.
- Dynamics candidates erzeugen keinen memory commit.
- Dynamics candidates mutieren keinen compute core.
- BB10-Doku bleibt konsistent zu BB2-BB9 sowie Compute-Exit/Maintenance-Linie.
- Internal/expert-only dynamics paths erscheinen nicht als kanonisch.
