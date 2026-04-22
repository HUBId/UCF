# Serie BB4 Prompt 4: Control-/Attention-feedback in Runtime Diagnostics zurückbinden

Status: BB4 Prompt 4 bindet die Selection-Semantik aus Prompt 1-3 explizit in Runtime Diagnostics und die Blue-Brain state/runtime surface zurück.

## 1) Kanonische Diagnostics-Map

Die kanonische Referenz ist `CANONICAL_BLUE_BRAIN_SELECTION_DIAGNOSTICS_MAP`.

Sie enthält genau diese Klassen:
- selected item diagnostic
- deferred item diagnostic
- ignored item diagnostic
- rejected item diagnostic
- blocked selection diagnostic
- insufficient selection diagnostic
- caveated selection diagnostic
- non-canonical/internal-only diagnostic detail

## 2) Outcomes und Reasons (kompakt, kanonisch)

Runtime Diagnostics nutzen kompakte Gründe statt freier Reasoning-Prosa:
- selected due to sufficient context
- selected due to primary evidence/reference
- deferred due to partial evidence
- blocked due to stale/insufficient basis
- ignored because not relevant to current transition
- rejected due to fault/caveat

Damit bleibt die Darstellung technisch, bounded und stabil.

## 3) Runtime-/State-Surface Rückbindung

Die Diagnostics werden über `ComputeStatusEvidenceExportSurface::control_attention_diagnostics` sichtbar.

Semantische Bindungen:
- selection-gated transition bleibt explizit
- no memory persistence implied bleibt explizit
- caveated/blocked/insufficient Diagnostik kann nächste Trigger-Eignung beeinflussen
- Trigger- und Candidate-Deferral-Signale bleiben auf derselben Selection-Semantik wie Runtime/Context/Evidence

## 4) Trigger-Arbitration + Candidate-Deferral Anschluss

Prompt 4 nutzt weiterhin die bestehenden BB4-Maps:
- `CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP`
- `CANONICAL_BLUE_BRAIN_CANDIDATE_DEFERRAL_LIFECYCLE_MAP`
- `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`

Prompt 4 ergänzt keine neue Scheduler-/Planning-Engine und keine neue Memory-Commit-Engine.

## 5) Explizite Grenzen

Diese Schicht ist ausdrücklich:
- keine Explainability-, Planning-, Policy- oder Audit-Plattform
- keine neue Monitoring-Plattform
- keine Governance-/Reasoning-/Orchestration-Neuplattform
- keine Compute-Core-Neuentwicklung

