# Serie BB6 Prompt 5: Readiness-Sweep und Abschlusslinie für Planning-/Reasoning-Candidates

Status: Serie BB6 ist mit Prompt 5 als **planning/reasoning candidate layer** technisch abgeschlossen.
Die kanonische Linie bleibt bewusst begrenzt: Candidate/Proposal/Diagnostics/Comparison sind
ausdrücklich getrennt, **ohne** Planning-/Reasoning-Engine, **ohne** automatische Action Execution,
**ohne** automatische Compute Invocation und **ohne** Memory Commit. Der Compute-Kern bleibt
auf finaler Exit-Linie maintenance-only.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_PLANNING_REASONING_CANDIDATE_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_ACTION_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_TO_PROPOSAL_TRANSITION_MAP`
  - `CANONICAL_BLUE_BRAIN_NON_EXECUTING_ACTION_PROPOSAL_STATE_MAP`
  - `CANONICAL_BLUE_BRAIN_REASONING_CANDIDATE_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_COMMIT_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie (unverändert):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) BB6-Abschlussmatrix (hart, repo-basiert)

| Bereich | Abschlussklasse | Repo-basierte Feststellung |
|---|---|---|
| Planning/reasoning candidate surface | stable planning/reasoning candidate foundation | Candidate-Klassen und Candidate-Basiszustände sind kanonisch und gegen Runtime/Context/Evidence/Selection/Memory-Boundary rückgebunden. |
| Candidate-to-proposal boundary | stable planning/reasoning candidate foundation | Candidate, proposal und executed-action-if-present sind als getrennte Klassen festgezogen. |
| Non-executing action proposal states | stable planning/reasoning candidate foundation | Proposal-Zustände sind explizit non-executing und tragen `no execution performed`. |
| Candidate diagnostics / insufficiency / caveat feedback | stable planning/reasoning candidate foundation | sufficient/partial/caveated/stale/insufficient/deferred/rejected/proposal-ready + non-canonical/internal-only bleiben getrennt. |
| Candidate comparison semantics | stable planning/reasoning candidate foundation | meaningful/caveated/inconclusive/not-meaningful/blocked + non-canonical/internal-only comparison sind explizit unterschieden. |
| Memory- und commit-feedback als Candidate-/Comparison-Basis | usable with caveats | `future-memory-ready` und `commit unavailable` sind Basis-/Diagnostiksignale, aber keine Persistenzautorität. |
| Proposal-ready -> spätere Action-Interfaces | preparatory only | Proposal-ready bleibt vorbereitend; keine automatische Planung/Entscheidung/Ausführung. |
| Expert/internal/compat planning-like oder action-like Pfade | non-canonical / internal-only | Nur nach explizitem Down-Mapping nutzbar; niemals direkte kanonische Autorität. |
| Actual planning/reasoning engine | intentionally deferred | BB6 liefert keine Engine und keinen reasoning-completed claim. |
| Actual action execution orchestration | intentionally deferred | BB6 erzeugt keine automatische Execution- oder Tool-Invocation-Schicht. |
| Actual memory commit engine | intentionally deferred | Candidate/Proposal/Comparison bleiben strikt ohne Auto-Commit. |
| Neurodynamische Spezialmodelle (Hodgkin-Huxley/Kuramoto) | intentionally deferred | Weiterhin außerhalb dieser Abschlusslinie. |

## 2) Explizite Planning-/Reasoning-Kandidatenlinie (kanonisch)

### 2.1 Candidate-Klassen
Kanonische Candidate-Klassen:
- `runtime-derived planning candidate`
- `context-derived reasoning candidate`
- `evidence/reference-derived reasoning candidate`
- `selection-derived action candidate`
- `memory-candidate-derived reasoning candidate`
- `commit-feedback-derived candidate`
- `insufficient candidate basis`
- `non-canonical/internal-only planning-like path`

### 2.2 Candidate-Quellen und Basiszustände
Kanonische Candidate-Basiszustände:
- `candidate basis available`
- `partial/caveated`
- `stale`
- `insufficient`
- `deferred`
- `candidate proposed but unresolved`
- `evidence observed but no reasoning candidate`
- `blocked`

### 2.3 Proposal-Zustände (non-executing)
Kanonische Proposal-Zustände:
- `proposal created`
- `proposal selected for possible future action`
- `proposal deferred`
- `proposal rejected`
- `proposal blocked`
- `proposal caveated`
- `proposal insufficient basis`
- `no execution performed`

### 2.4 Candidate-Diagnostics-Zustände
Kanonische Diagnostics-Zustände:
- `candidate-basis diagnostic`
- `sufficient candidate diagnostic`
- `partial candidate diagnostic`
- `caveated candidate diagnostic`
- `stale candidate diagnostic`
- `insufficient candidate diagnostic`
- `deferred candidate diagnostic`
- `rejected candidate diagnostic`
- `proposal-ready diagnostic`
- `non-canonical/internal-only diagnostic`

### 2.5 Candidate-Comparison-Zustände
Kanonische Comparison-Zustände:
- `comparable candidates`
- `comparison basis available`
- `comparison meaningful`
- `comparison caveated`
- `comparison inconclusive`
- `comparison not meaningful`
- `comparison blocked`
- `non-canonical/internal-only comparison`

## 3) Final abgesicherte Grenzen (keine Verwischung)

Diese Nicht-Gleichsetzungen sind Abschlussbedingungen von BB6:
- Candidate ≠ Plan.
- Candidate ≠ Reasoning Completed.
- Proposal (auch `selected proposal`) ≠ Action Execution.
- Candidate Diagnostics ≠ Explainability-/Reasoning-Plattform.
- Candidate Comparison ≠ Ranking/Selection/Entscheidungszwang.
- Candidate/Proposal/Comparison ≠ automatische Compute Invocation.
- Candidate/Proposal/Comparison ≠ Memory Commit.
- Candidate/Proposal/Comparison ≠ neurodynamische Modellintegration.

## 4) Compute-Core-Abschlusslinie erneut abgesichert

BB6 öffnet keine neue Compute-Core-Arbeit:
- Compute bleibt finale Compute-Linie mit outward-facing Contracts.
- Compute-Kern bleibt maintenance-only.
- BB6 bleibt auf Candidate-/Proposal-/Diagnostics-/Comparison-Semantik über bestehenden Kernsignalen.

## 5) Nächste Blue-Brain-Richtungen (repo-treu)

1. **Serie BB7: Minimal planning/action interface (non-executing proposal handoff)**
   - Höchster Hebel jetzt: BB6-Proposal-Semantik kann in ein minimales Interface überführt werden,
     ohne Auto-Execution einzuführen.
2. **Serie BB8: Actual Memory Subsystem Minimal Implementation**
   - Liefert reale Persistenz-Rückkopplung für Candidate-/Proposal-Basis, bleibt aber nach BB6
     nachrangig zur unmittelbaren Proposal-Interface-Lücke.
3. **Serie BB9: Neural dynamics integration candidates (inkl. Hodgkin-Huxley/Kuramoto)**
   - Bleibt nachrangig, bis minimale planning/action- und/oder tatsächliche memory-Persistenzanschlüsse
     belastbar realisiert sind.

### Priorisiert als nächster erster Schritt
**Priorität: Serie BB7 zuerst.**

Technische Begründung (knapp):
- BB6 hat genau die non-executing Proposal-Grenze stabilisiert; der nächste direkte Hebel ist ein
  minimaler handoff-fähiger planning/action interface layer auf derselben Grenze.
- BB8 bleibt wichtig, ist aber nicht der unmittelbarste Anschluss zur in BB6 abgeschlossenen
  Candidate-to-Proposal-Linie.
- Hodgkin-Huxley/Kuramoto bleiben weiterhin nicht zuerst, weil BB6 bewusst keine echte Planning-/Action-
  oder Memory-Engine etabliert und neurodynamische Integration ohne diese Anschlussschichten verfrüht wäre.

## 6) Gezielte Konsistenz-Checkliste (BB6-Abschluss)

Die Abschlusslinie gilt nur, solange folgende Bedingungen bestehen bleiben:
- runtime-/context-/evidence-/selection-/memory-derived candidates bleiben unterscheidbar.
- candidate/proposal/executed-action-if-present bleiben unterscheidbar.
- sufficient/partial/caveated/stale/insufficient/deferred/rejected/proposal-ready diagnostics bleiben unterscheidbar.
- meaningful/caveated/inconclusive/not-meaningful/blocked comparisons bleiben unterscheidbar.
- Candidate/Proposal/Comparison erzeugen keine automatische Action Execution.
- Candidate/Proposal/Comparison erzeugen keinen Memory Commit.
- Candidate/Proposal/Comparison erzeugen keine automatische Compute Invocation.
- BB6-Doku bleibt konsistent zu BB2/BB3/BB4/BB5 und zur Compute-Exit-/Maintenance-Linie.
- internal/expert-only Pfade erscheinen nicht als kanonische Planning-/Reasoning-Candidate-Surface.
