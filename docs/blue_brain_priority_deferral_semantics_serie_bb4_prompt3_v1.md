# Serie BB4 Prompt 3: Evidence-/Context-priority und Candidate-deferral Lifecycle (Runtime + Selection + Trigger)

Status: BB4 Prompt 3 integriert Priority-/Deferral-Semantik direkt in die bestehende BB3/BB4 Linie,
ohne neue Ranking-, Planning-, Policy-, RL- oder Memory-Commit-Engine einzuführen.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Kanonische Context-/Evidence-Priority-Klassen

Die Priority-Map trennt explizit:
- primary context,
- supporting context,
- deferred context,
- ignored context,
- stale context,
- insufficient context,
- primary evidence/reference,
- supporting evidence/reference,
- caveated evidence/reference,
- non-canonical/internal-only priority path.

Diese Klassen sind deterministische Prioritätsklassen und **keine numerische Ranking- oder Scoring-Engine**.

## 2) Candidate-deferral Lifecycle (ohne Commit)

Die Candidate-Deferral-Lifecycle-Map trennt explizit:
- candidate selected,
- candidate deferred,
- candidate deferred pending stronger evidence,
- candidate deferred pending context update,
- candidate rejected,
- candidate stale,
- candidate insufficient,
- candidate not persisted.

Damit wird Deferral klar von Rejection/Stale/Insufficient getrennt und als eigener reviewbarer Zustand
geführt.

## 3) Bedeutung von Deferral vs Rejected/Ignored/Stale/Insufficient

Deferral bedeutet kanonisch:
- aktuell nicht ausgewählt,
- aber weiterhin potenziell relevant,
- mit explizitem deferral reason,
- mit recheck condition,
- no compute trigger yet,
- no memory commit,
- not rejected.

`deferred context` und `candidate deferred` bleiben also explizite Kontroll-/Lifecycle-Posturen und
werden nicht als versteckter Fehler, Ignore oder Persistenz interpretiert.

## 4) Rückbindung an Reference Quality (BB3)

Priority und Deferral nutzen die BB3-Qualitätsklassen als Gate:
- sufficient,
- partial,
- stale,
- caveated,
- insufficient.

Konsequenz:
- sufficient kann primary selection stützen,
- partial/caveated stützt supporting oder deferred posture,
- stale bleibt stale (nicht als deferred/selected getarnt),
- insufficient blockiert Trigger-Eskalation.

## 5) Rückbindung an BB4 Trigger-Arbitration

Die Priority-/Deferral-Maps bleiben mit BB4 Prompt 2 verbunden:
- primary context selected for trigger,
- deferred candidate does not trigger compute,
- caveated evidence permits caveated trigger,
- insufficient context blocks trigger,
- deferred item may become trigger candidate later.

Dabei bleibt Invocation selection-gated auf `CanonicalComputeEntryPoint::submit`.

## 6) Runtime Feedback / Diagnose-Sicht

Runtime-nahe Feedback-Semantik bleibt explizit:
- what was selected,
- what was deferred,
- why selected/deferred,
- what caveats remain,
- what is ignored/rejected.

Das ist eine schmale technische Sicht und **keine Explainability- oder Monitoring-Plattform**.

## 7) Non-canonical Priority-/Deferral-Pfade

Nicht kanonisch für BB4-Priority-/Deferral-Autorität:
- internal/expert-only hooks,
- legacy/compat lanes (`build_backend(kind=stub|candle|worker)`, `domains/ai*`),
- implizite helper-only Prioritätsannahmen ohne outward mapping.

Diese Pfade bleiben non-canonical/internal-only oder benötigen explizites Down-Mapping auf outward
status/evidence/context references.

## 8) Grenzen

Bewusst nicht Teil dieser Integration:
- keine numerische Ranking- oder Scoring-Engine,
- keine Planning-/Scheduler-/Workflow-Plattform,
- keine Policy-/Governance-/RL-/Reasoning-Plattform,
- keine Memory-Consolidation- oder Commit-Engine,
- keine neue Compute-Core-Arbeit,
- keine neurodynamische Spezialintegration.

## 9) Ergebnis

BB4 Prompt 3 liefert eine belastbare Priority-/Deferral-Semantik:
- Context/Evidence/Candidate-Priorität ist kanonisch klassifiziert,
- deferred bleibt reviewbar und wird nicht mit rejected/ignored/stale/insufficient vermischt,
- Trigger-Arbitration bleibt an dieselbe outward-facing Compute-Linie gebunden,
- Candidate-Deferral bleibt strikt non-commit und readiness-/context-/evidence-gebunden.
