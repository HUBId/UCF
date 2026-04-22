# Serie BB4 Prompt 5: Readiness-Sweep und finale Control-/Attention-Grundlinie

Status: BB4 ist als **Control-/Attention-/Selection-Baseline** repo-basiert abgeschlossen.
Die Semantik bleibt explizit auf Runtime/Context/Evidence/Candidate/Trigger/Diagnostics begrenzt und
öffnet weder neuen Compute-Core noch Planning-/Reasoning-/Policy- oder Memory-Commit-Subsysteme.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie (unverändert):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) BB4-Abschlussmatrix (hart, repo-basiert)

| Bereich | Abschlussklasse | Repo-basierte Feststellung |
|---|---|---|
| Control-/Attention-/Selection-Surface | stable control/attention foundation | Auswahlklassen und Dispositionen sind als eigene kanonische Map mit klarer Non-canonical-Grenze kodiert. |
| Trigger-Arbitration + selection-gated invocation | stable control/attention foundation | Triggerquellen, Triggerzustände und Invocation-Gates sind explizit modelliert; Invocation bleibt an `CanonicalComputeEntryPoint::submit` gebunden. |
| Context/Evidence Priority + Candidate Deferral | stable control/attention foundation | Primary/supporting/deferred/stale/insufficient/caveated bleiben getrennt; Candidate-Deferral ist explizit non-commit. |
| Control-/Attention-Diagnostics | stable control/attention foundation | Selected/deferred/ignored/rejected/blocked/insufficient/caveated Diagnoseklassen sind kompakt und runtime-surface-gebunden exportierbar. |
| Caveated Selection-/Trigger-Posture | usable with caveats | Caveated/partial Basis erlaubt nur explizit caveated/deferred Verhalten; keine stille Hochstufung zu primary/sufficient. |
| BB3 future-memory attachment lanes | preparatory only | Candidate-/Attachment-Lanes bleiben proposal/handoff-only, ohne tatsächliche Persistenz-Commit-Autorität. |
| Internal/expert/compat control paths | non-canonical / internal-only | `run_operation_with_entry`, `replay_with_entry`, `build_backend(kind=stub|candle|worker)`, `domains/ai*` bleiben explizit außerhalb der kanonischen BB4-Autorität. |
| Actual memory commit engine | intentionally deferred | Candidate Selection/Deferral bleibt strikt ohne candidate->persisted-memory Auto-Commit. |
| Planning/Reasoning/Policy platform | intentionally deferred | BB4 nutzt deterministische Klassen und keine Ranking-/Policy-/Planner-Plattform. |
| Neurodynamische Spezialintegration (z. B. Hodgkin-Huxley/Kuramoto) | intentionally deferred | Kein Bestandteil der BB4-Control-/Attention-Linie. |

## 2) Explizite BB4-Control-/Attention-Grundlinie

### 2.1 Kanonische Selection-/Attention-Klassen
- `attention target`
- `context selection`
- `evidence/reference selection`
- `memory-candidate selection`
- `compute-trigger selection`
- `non-canonical/internal-only selection path` (explizit ausgeschlossen)

Kanonische Dispositionen bleiben explizit unterscheidbar:
- `selected`,
- `deferred`,
- `ignored_or_irrelevant`,
- `rejected`,
- `blocked`,
- `insufficient`,
- `caveated`.

### 2.2 Kanonische Trigger-Zustände und Trigger-Quellen
Trigger-Arbitration bleibt kanonisch auf:
- Zustände: `trigger candidate`, `selected`, `deferred`, `suppressed`, `blocked`, `insufficient`, `caveated`, `non-canonical/internal-only`.
- Quellen: `context-derived`, `evidence/reference-derived`, `runtime-state-derived`, `memory-candidate-derived`, `feedback-derived`, `manual/internal-only non-canonical`.

Gate bleibt hart:
- Nur kanonisch `selected trigger` mit zulässigem Gate kann Invocation auslösen.
- `deferred/blocked/insufficient/suppressed` lösen **keinen** Compute-Aufruf aus.

### 2.3 Kanonische Priority-/Deferral-Zustände
Context/Evidence Priority bleibt klassifiziert als:
- `primary`, `supporting`, `deferred`, `ignored`, `stale`, `insufficient`, `caveated`, plus non-canonical boundary.

Candidate-Deferral-Lifecycle bleibt klassifiziert als:
- `candidate selected`,
- `candidate deferred`,
- `candidate deferred pending stronger evidence`,
- `candidate deferred pending context update`,
- `candidate rejected`,
- `candidate stale`,
- `candidate insufficient`,
- `candidate not persisted`.

### 2.4 Kanonische Diagnostics
Control-/Attention-Diagnostics bleiben exakt:
- `selected item diagnostic`,
- `deferred item diagnostic`,
- `ignored item diagnostic`,
- `rejected item diagnostic`,
- `blocked selection diagnostic`,
- `insufficient selection diagnostic`,
- `caveated selection diagnostic`,
- `non-canonical/internal-only diagnostic detail`.

Export bleibt auf der bestehenden Runtime-Surface:
- `ComputeStatusEvidenceExportSurface::control_attention_diagnostics`.

## 3) Final abgesicherte Grenzen

Diese Grenzen bleiben explizit kanonisch:
- Selection ist **keine** Planning Engine.
- Priority/Deferral ist **kein** Ranking- oder Policy-System.
- Candidate Selection impliziert **keinen** Memory Commit.
- Trigger-Arbitration ist **keine** neue Scheduler-/Orchestration-Plattform.
- Diagnostics erzeugen **keine** Explainability-/Reasoning-/Audit-Claims.
- Hodgkin-Huxley/Kuramoto sind **nicht** Teil von BB4.

## 4) Compute-Core-Abschlusslinie (erneut gesichert)

BB4 öffnet keine neue Compute-Core-Arbeit:
- Compute-Core bleibt auf finaler Referenzlinie.
- BB4 bleibt auf outward-facing Contracts (`submit`, status/evidence export, snapshot-Bindung).
- Compute-Kern bleibt maintenance-only; BB4 verändert die Core-Semantik nicht.

## 5) Nächste Blue-Brain-Richtungen (1-3, repo-treu)

1. **Serie BB5: actual memory subsystem boundary + minimal memory commit interface**
   - Höchster Hebel nach BB4, weil Candidate/Deferral bereits stabil modelliert sind, aber Commit bewusst fehlt.
2. **Serie BB6: planning/reasoning candidate layer auf BB2/BB3/BB4-Semantik**
   - Erst sinnvoll nach stabiler minimaler Memory-Commit-Grenze, damit Planung nicht nur transient bleibt.
3. **Serie BB7: neural dynamics integration candidates (inkl. Hodgkin-Huxley/Kuramoto)**
   - Nachrangig bis Memory-Commit- oder Planning-Grundlage real existiert.

### Priorisiert als nächster erster Schritt
**Priorität: Serie BB5 zuerst.**

Technische Begründung (knapp):
- BB4 liefert eine belastbare Control-/Attention-/Selection-/Deferral-/Diagnostics-Grundlinie,
  aber weiterhin ohne actual commit boundary.
- BB5 schließt die nächste reale Systemlücke (candidate -> minimal commit contract) mit direkter
  Anschlussfähigkeit an BB3/BB4.
- BB6 ist nachrangig, weil Planning/Reasoning ohne Commit-Grenze leicht in transiente oder implizite
  Semantik abrutscht.
- BB7 (Hodgkin-Huxley/Kuramoto) bleibt nachrangig, solange Memory-/Planning-Grundlage nicht stabil
  als kanonische Runtime-/Context-Anbindung existiert.

## 6) Konsistenz-Checkliste (BB4-Abschluss)

Die Abschlusslinie bleibt nur gültig, solange folgende Bedingungen erfüllt bleiben:
- `selected/deferred/ignored/rejected/blocked/insufficient/caveated` bleiben in Selection,
  Priority/Deferral und Diagnostics getrennt modelliert.
- Trigger-Arbitration bleibt auf kanonischen outward-facing Compute Contracts.
- Deferred Candidate löst weder Compute noch Memory Commit automatisch aus.
- Diagnostics bleiben auf kanonischen Runtime-/Selection-Pfaden und werden nicht zu Explainability-/Audit-Plattform umgedeutet.
- BB4-Doku bleibt konsistent mit BB2/BB3-Readiness und Compute-Exit/Maintenance-Grenze.
- Internal/expert-only Pfade erscheinen nicht als kanonische Control-/Selection-Autorität.
