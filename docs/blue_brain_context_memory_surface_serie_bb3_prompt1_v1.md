# Serie BB3 Prompt 1: Kanonische Blue-Brain Context/Memory Surface auf BB2-Runtime-Grundlinie

Status: Die BB2-Runtime-Grundlinie bleibt unverändert; BB3 Prompt 1 zieht eine code-pinned
Context/Memory-Surface nach, ohne neue Memory-Engine oder neue Storage-Plattform einzuführen.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
- `runtime/ucf-runtime/src/orchestrator.rs`
- `runtime/ucf-compute/src/service_surface.rs`

Finale Compute-Referenzlinie (weiterhin verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Repo-basierter Befund der realen Context-/Memory-nahen Flächen

Aus dem aktuellen Stand sind folgende Flächen real vorhanden und load-bearing:
- transienter Runtime-Kontext:
  - `runtime_orchestrator_stateful_loop`
  - `runtime_handoff_state_from_evidence` / `runtime_handoff_state_from_action_code`
- evidence-backed Kontext:
  - `CanonicalComputeEntryPoint::status_evidence_export_surface`
  - status/evidence posture (`current|partial|stale|caveated|degraded|insufficient`)
- replay/reference-backed Kontext:
  - `replay_preflight` / `replay_with_entry` mit Referenz- und Vergleichsbezug
- memory-adjacent candidate:
  - `blue_brain_transition_memory_adjacent_candidate_identified_not_committed`
- tatsächliche Persistenz:
  - keine kanonische Blue-Brain-Memory-Persistenzlane im aktuellen Repo-Baseline-Stand.

Damit ist explizit getrennt:
- Kontextnutzung,
- Evidence-/Replay-Referenznutzung,
- memory-adjacent Kandidaten,
- und fehlende tatsächliche Memory-Persistenz.

## 2) Kanonische Surface-Map für BB3 Prompt 1

`CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_SURFACE_MAP` führt die kanonischen Klassen ein:
1. `transient_runtime_context`
2. `evidence_backed_context`
3. `replay_reference_backed_context`
4. `memory_adjacent_candidate`
5. `persisted_memory`
6. `non_canonical_internal_only_memory_like_path`

Jede Lane enthält konsistent:
- `source_surface`
- `context_shape`
- `evidence_or_reference_binding`
- `persistence_binding`
- `canonical_guard`

Dadurch bleibt klar sichtbar, welche Referenzen nur Runtime-Kontext stützen und welche (noch)
kein Memory-Commit darstellen.

Zusätzlich codiert die Surface jetzt explizite BB3-Lifecycle-Semantik:
- `blue_brain_transient_runtime_context_available_for_transition`
- `blue_brain_transient_runtime_context_used_for_compute_trigger`
- `blue_brain_compute_result_context_uptake_non_memory`
- `blue_brain_transient_runtime_context_updated_then_discarded`

Damit ist sichtbar:
- context available for current transition,
- context used for compute trigger,
- context updated by result/evidence,
- context discarded or not persisted,
- no memory persistence implied.

## 3) Persisted-Memory-Lane ist explizit als „nicht vorhanden“ codiert

Die Lane `blue_brain_persisted_memory_none_in_current_baseline` ist bewusst eine Null-Lane:
- sie verhindert, dass History/Evidence/Replay semantisch als Memory-Store umgedeutet werden,
- sie macht sichtbar, dass der Persistenzpfad für BB3 Prompt 1 nicht implementiert ist,
- sie schützt den abgeschlossenen Compute-Kern vor implizitem Re-Design.

## 4) Compute-Ergebnisse und Evidence-Feedback sind nicht automatisch Memory

Kanonische Guardrails der Map:
- compute-result uptake (`blue_brain_compute_result_context_uptake_non_memory`) bleibt transient,
- evidence-backed context (`blue_brain_evidence_backed_context_status_export`) bleibt referenzgebunden,
- evidence observed/attached/caveated
  (`blue_brain_evidence_backed_context_attached_or_caveated`) bleibt referenzgebunden und kann
  explizit insufficient sein,
- replay/reference context kann partial/caveated sein
  (`blue_brain_replay_reference_backed_context_caveated_or_partial`) ohne Persistenzwirkung,
- memory-adjacent candidate (`blue_brain_memory_adjacent_candidate_not_committed`) bleibt uncommitted,
- candidate source semantics über context/result/evidence references
  (`blue_brain_memory_adjacent_candidate_derived_sources_uncommitted`) bleiben explizit
  non-committed,
- internal/expert memory-like paths bleiben non-canonical.

Damit gilt für BB3 Prompt 1 explizit:
- compute outputs und evidence feedback dürfen Runtime-Kontext anreichern,
- compute outputs and evidence feedback are treated as context support, not persistence,
- sie dürfen aber ohne separaten Persistenzvertrag nicht als Memory-Commit zählen.

## 5) Non-canonical/internal-only Memory-like Pfade

`blue_brain_internal_expert_memory_like_path_non_canonical` hält die Grenze stabil:
- `run_operation_with_entry` / `replay_with_entry` / `build_backend(kind=stub|candle|worker)` /
  `domains/ai*` bleiben ausgeschlossen von kanonischer Blue-Brain-Memory-Autorität,
- Blue-Brain-facing Nutzung ist nur nach Down-Mapping auf outward status/evidence references zulässig.

## 6) Ergebnis für Serie BB3-Fortsetzung

BB3 Prompt 1 liefert eine belastbare Surface-Semantik, auf der später aufgebaut werden kann:
- klare Trennung zwischen transient runtime context, evidence-backed context,
  replay/reference-backed context, memory-adjacent candidate und persisted memory,
- keine automatische Gleichsetzung von Evidence/Compute mit Memory,
- keine Öffnung des finalisierten Compute-Kerns,
- klare Startlinie für spätere, explizit spezifizierte Memory-Implementierung.

Rückbindung an BB2-Transitions bleibt explizit:
- transition uses transient context,
- transition updates context,
- transition produces memory-adjacent candidate,
- transition observes evidence/reference,
- transition does not imply persistence unless explicitly supported.

Nicht enthalten in diesem Schritt:
- kein Memory-Engine-Bau,
- keine Vector-DB/Knowledge-Graph-Plattform,
- keine neurodynamische Spezialmodell-Integration.
