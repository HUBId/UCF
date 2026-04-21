# Serie BB2 Prompt 4: Blue-Brain Runtime Diagnostics/Evidence-Feedback Rückbindung

Status: compute result/status/evidence/diagnostic feedback ist jetzt als eigene kanonische
Feedback-Schicht für die Blue-Brain state/runtime surface codiert und gegen non-canonical
internal/expert Pfade abgegrenzt.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Compute-Referenzlinie (verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Diese Prompt-4-Schicht baut **keine** Monitoring-Plattform, keine Reasoning-Engine, keine
Memory-Engine und keine zweite Runtime-/Governance-Wahrheitsquelle. Ziel ist nur die
rückgebundene, runtime-nutzbare Feedback-Semantik auf outward-facing Compute-Exports.

## 1) Repo-basierte Prüfung der bestehenden Pfade

Aktive Grundlage aus BB1 + BB2 Prompt 1-3 bleibt:
- erster realer Integrationskandidat: `runtime_orchestrator_stateful_loop`
- runtime surface: state/inference/status/evidence/internal-control split
- transition/trigger map: pure transition vs compute-trigger vs evidence/status updates
- context/memory boundary: context uptake/memory-adjacent candidate ohne Commit

Vor Prompt 4 bereits sichtbar:
- outward status/evidence exports sind vorhanden
- caveated/degraded/partial/stale status ist outward modelliert
- evidence refs/replay refs sind outward referenzierbar

Noch zu schließen war:
- explizite Feedback-Klassen für Result/Status/Evidence/Diagnostics/Context als
  **runtime feedback map** statt nur verstreuter Surface-/Transition-Beschreibung.

## 2) Kanonische Blue-Brain Feedback-Map

`CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP` führt sechs Klassen explizit:
1. `compute_result_feedback`
2. `status_trust_feedback`
3. `evidence_reference_feedback`
4. `diagnostic_caveat_feedback`
5. `context_uptake_feedback`
6. `non_canonical_internal_expert_feedback`

Pro Lane sind strikt getrennt:
- `canonical_source`
- `runtime_feedback_semantics`
- `transition_binding`
- `memory_boundary`
- `non_canonical_boundary`

## 3) Compute-Result-Feedback in State-Transitions

Result-Feedback ist jetzt explizit als runtime feedback unterscheidbar:
- `blue_brain_feedback_result_integrated_current_runtime_state`
- `blue_brain_feedback_result_rejected_or_blocked`
- `blue_brain_feedback_result_integrated_with_caveat`

Damit ist sauber sichtbar:
- result integrated into current runtime state
- result rejected/blocked due to fault semantics
- result integrated with caveat
- kein Memory-Commit

## 4) Status-/Trust-Feedback auf outward Signalen

`blue_brain_feedback_status_trust_current_to_insufficient` bindet Runtime auf dieselben
outward-facing Statussignale zurück:
- current/trusted
- partial
- stale
- caveated
- degraded
- insufficient/blocked

Keine neue Policy-Engine: die Feedback-Map informiert Runtime-Posture und Folgeübergänge, ersetzt
aber keine bestehende Trigger-/Contract-Linie.

## 5) Evidence-/Reference-Feedback

Explizite Evidence-Feedback-Lanes:
- `blue_brain_feedback_evidence_observed_and_attached`
- `blue_brain_feedback_evidence_caveated_partial_or_insufficient`

Damit ist sichtbar:
- evidence observed
- evidence attached to current runtime context
- evidence caveated/partial
- evidence insufficient for stronger transition
- no automatic memory commit

## 6) Diagnostic Caveats als Runtime Caveats

Prompt 4 trennt Diagnose-Caveats explizit:
- `blue_brain_feedback_diagnostic_only_caveat`
- `blue_brain_feedback_trigger_blocking_or_context_uptake_caveat`

Damit ist unterscheidbar:
- diagnostic only
- runtime-relevant caveat
- trigger-blocking caveat
- context-uptake caveat
- non-canonical/internal diagnostic detail not exported

## 7) Context Uptake vs Evidence vs Memory Persistence

`blue_brain_feedback_context_uptake_transient_memory_adjacent_candidate` verankert:
- observed evidence
- context uptake
- transient runtime context
- memory-adjacent candidate
- actual memory persistence not implemented here

BB2 bleibt bewusst vor BB3: kein Memory-Commit, keine Vorwegnahme einer Memory-Architektur.

## 8) Non-canonical Feedback-Pfade ausgegrenzt

`blue_brain_feedback_non_canonical_internal_expert_only` markiert:
- expert/internal diagnostics
- legacy/compat objects
- raw compute internals

als non-canonical. Nutzung ist nur via Down-Mapping auf outward status/evidence references zulässig.

## 9) Doku-Rückbindung und zweite-Wahrheit-Vermeidung

Diese Doku ist direkt an dieselben code-pinned Maps gebunden und ergänzt BB2 Prompt 1-3, ohne
zweite Wahrheitsquelle.

## 10) Ergebnis

Die Blue-Brain runtime surface kann diagnostics/evidence feedback jetzt kanonisch rückführen:
- Result-/Status-/Evidence-/Diagnostic-/Context-Feedback sind strukturiert unterscheidbar.
- Caveated/degraded/partial/stale und insufficient/blocked sind runtime-seitig nutzbar.
- Evidence feedback bleibt referenzbasiert und führt nicht implizit zu Memory-Persistenz.
- Internal/expert-only Diagnostics bleiben non-canonical.
