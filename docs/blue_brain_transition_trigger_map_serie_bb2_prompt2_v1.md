# Serie BB2 Prompt 2: Blue-Brain State-Transitions und Compute-Trigger Points (repo-basiert, schmal)

Status: Transition-/Trigger-Semantik zwischen Blue-Brain-Runtimezustand und kanonischer Compute-Nutzung ist explizit festgezogen.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP`
  - `CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP`
- `runtime/ucf-runtime/src/orchestrator.rs`
- `runtime/ucf-compute/src/service_surface.rs`

Finale Compute-Referenzlinie (verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Diese BB2-Schärfung bleibt auf derselben outward-facing Compute-Linie. Es gibt keine neue Trigger-Engine, keine Workflow- oder State-Machine-Plattform, keine zweite Execution-Sprache und keine zweite Wahrheitsquelle.

## 1) Repo-treue Ist-Prüfung: Transitionen vs Compute-Invocation

- `runtime_orchestrator_stateful_loop` bleibt der erste reale Blue-Brain-nahe stateful Integrationskandidat.
- Bereits vorhandene BB1/BB2-Prompt-1-Schicht trennt Surface und Phasen, aber Triggerpunkte waren noch nicht als eigene schmale Schicht ausformuliert.
- Semantisch kritische Vermischungszonen wurden explizit adressiert:
  - state/context update vs tatsächlicher compute trigger,
  - inference-required transition vs implizite helper/internal Nebenpfade,
  - status/evidence uptake vs erneute compute invocation.

## 2) Schmale Transition/Trigger-Map (minimal Klassen)

`CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP` führt vier Klassen mit elf kanonischen Lanes:

1. `pure_state_transition`
   - Zustand/Context wird fortgeschrieben, **ohne** Compute-Trigger.
2. `compute_triggering_transition`
   - Kanonische Compute-Auslösung oder explizit blockierter Triggerfall.
3. `evidence_status_update_transition`
   - Ergebnis-/Status-/Evidence-Integration ohne implizite Re-Invocation.
4. `internal_only_or_non_canonical_transition`
   - explizit unterdrückte/sperrige Internal-/Expert-Pfade.

Damit bleiben Zustandsfortschritt und Compute-Trigger semantisch getrennt.

Kanonische lane-IDs (code-pinned):
- `blue_brain_transition_context_available`
- `blue_brain_transition_state_context_refreshed`
- `blue_brain_transition_context_used_for_compute_trigger`
- `blue_brain_transition_compute_trigger_from_context_availability`
- `blue_brain_transition_compute_trigger_from_inference_required`
- `blue_brain_transition_compute_trigger_blocked_insufficient_context`
- `blue_brain_transition_compute_trigger_suppressed_internal_only_path`
- `blue_brain_transition_compute_result_integrated`
- `blue_brain_transition_evidence_observed_without_memory_commit`
- `blue_brain_transition_memory_adjacent_candidate_identified_not_committed`
- `blue_brain_transition_status_evidence_update_without_compute_trigger`

## 3) Kanonische Compute-Trigger-Points

Explizit sichtbar und code-pinned:

- `blue_brain_transition_compute_trigger_from_context_availability`
  - Trigger aus state/context availability (bei verfügbaren handoff/context references).
- `blue_brain_transition_compute_trigger_from_inference_required`
  - Trigger aus inference-required transition.
- `blue_brain_transition_compute_trigger_blocked_insufficient_context`
  - Trigger ist blockiert, wenn Kontext/State nicht reicht.
- `blue_brain_transition_compute_trigger_suppressed_internal_only_path`
  - Trigger bleibt unterdrückt, wenn nur internal/expert Pfad Voraussetzungen erfüllen würde.

Alle legitimen Trigger bleiben an `CanonicalComputeEntryPoint::submit` gebunden.

## 4) Rückbindung an outward-facing Contracts/Handoffs

Jeder kanonische Triggerpunkt verweist auf dieselbe outward Vertragslinie:
- Inference execution: `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
- Status/Evidence: `CanonicalComputeEntryPoint::status + status_evidence_export_surface(...)`

Es gibt keine zweite Ausführungssprache und keine implizite Expert/Internal-Defaultsemantik.

## 5) Status-/Evidence-getriebene Folgeübergänge

Explizit gemacht wurden:
- `blue_brain_transition_compute_result_integrated`
  - compute result integrated transition.
- caveated/degraded/partial Resultate als status/evidence-getriebene Runtime-Postur.
- `blue_brain_transition_status_evidence_update_without_compute_trigger`
  - evidence/status update transition ohne neuen compute trigger.

Damit bleibt Rückfluss aus Status/Evidence ein eigener Übergangspfad.

## 6) Non-canonical Triggerpfade explizit ausgegrenzt

Nicht kanonisch als Triggerautorität:
- `replay_with_entry`, `run_operation_with_entry`
- `build_backend(kind=stub|candle|worker)`
- legacy/compat/internal helper lanes

Wenn solche Pfade technisch involviert sind, müssen sie auf outward status/evidence references zurückgemappt werden, bevor Blue-Brain-facing Nutzung erfolgt.

## 7) Referenzkontinuität an Triggern/Transitionen

Load-bearing Referenzen werden entlang der Trigger-/Transition-Linie fortgeführt:
- request/run identity (wo vorhanden),
- outward status references,
- evidence references,
- active production context.

Keine neue Objektwelt; nur konsistente Weitergabe derselben Kernreferenzen.

## 8) Doku-Rückbindung (eine Wahrheitsquelle)

Diese Doku referenziert ausschließlich die code-pinned Map in `reference_map.rs`.
Keine zweite Wahrheitsquelle.

## 9) Kleine Konsistenzchecks

Ergänzt wurden schmale Checks für:
- minimale Transition/Trigger-Klassen,
- kanonische Triggerbindungen auf submit/status-evidence contract surfaces,
- blockiert/unterdrückt Triggerzustände ohne internal-default escalation,
- Doku-zu-Code-Alignment für BB2 Prompt 2.

## Kurzfazit

BB2 Prompt 2 liefert eine kleine, belastbare Transition-/Trigger-Semantik über der finalen Compute-Linie:
- state transition bleibt getrennt von compute invocation,
- compute trigger points sind explizit und contract-gebunden,
- status/evidence Folgeübergänge bleiben eigenständig,
- non-canonical internal/expert Triggerpfade sind sichtbar ausgegrenzt.
