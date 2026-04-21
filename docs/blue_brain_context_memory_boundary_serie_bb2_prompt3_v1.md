# Serie BB2 Prompt 3: Blue-Brain Context/Memory-Adjacent Boundary über finaler Compute-Linie

Status: repo-basierte Abgrenzung zwischen reiner Compute-Nutzung, context-bearing Nutzung,
memory-adjacent Kandidaten und Evidence/Replay-Referenznutzung ist explizit codiert und getestet.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP`
- `runtime/ucf-runtime/src/orchestrator.rs`
- `runtime/ucf-compute/src/service_surface.rs`

Finale Compute-Referenzlinie (verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Diese Serie baut **keine Memory-Architektur**, keine Vektor-DB, kein Langzeitgedächtnis und keine zweite
Runtime-Plattform. Sie schärft nur die semantische Boundary für BB3.

## 1) Reale context-/memory-adjacent Flächen (repo-basiert)

Explizit getrennt:
- reine Compute-Consumer:
  - `ops_compute_probe` (kanonischer Submit + outward status/evidence consumption, ohne
    runtime-owning context/memory).
- context-bearing Blue-Brain Surface:
  - `runtime_orchestrator_stateful_loop` (state/context-tragend, aber ohne persistentes Memory-Contract).
- memory-adjacent Surface:
  - `runtime_handoff_state_from_evidence` + Trigger-Marker
    `blue_brain_transition_memory_adjacent_candidate_identified_not_committed`.
- evidence/reference consumer:
  - `status_evidence_export_surface` für `bundle_refs`, `trace refs`, `history/replay refs`.
- non-canonical context path:
  - `run_operation_with_entry`, `replay_with_entry`, `build_backend(kind=stub|candle|worker)`, `domains/ai*`.

## 2) Kanonische Blue-Brain context/memory boundary map

`CANONICAL_BLUE_BRAIN_CONTEXT_MEMORY_BOUNDARY_MAP` unterscheidet minimal und vollständig:
1. `pure_compute_consumer`
2. `context_bearing_blue_brain_surface`
3. `memory_adjacent_blue_brain_surface`
4. `evidence_reference_consumer`
5. `internal_only_or_non_canonical_context_path`

Die Map trennt pro Lane explizit:
- `compute_invocation_reference`
- `context_reference`
- `evidence_or_replay_reference`
- `memory_posture`
- `boundary_guard`

Damit werden compute invocation references, state/context references und evidence/replay references
nicht semantisch vermischt.

## 3) Compute invocation vs Context-/Memory-Referenzen

Kanonisch sichtbar:
- Compute-Aufruf (`CanonicalComputeEntryPoint::submit`) ist expliziter Triggerpfad.
- Context-Referenz ist state/runtime-bezogen und kein Memory-Commit.
- Evidence-/Replay-Referenz ist Beobachtungs-/Beleg-Referenz und kein Memory-Eintrag.
- `memory_adjacent` bedeutet in BB2 nur: Kandidat identifiziert, **nicht committed**.

## 4) Context-bearing Übergänge in der runtime surface

`CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP` ergänzt die Übergangssemantik um:
- `blue_brain_transition_context_available`
- `blue_brain_transition_context_used_for_compute_trigger`
- `blue_brain_transition_compute_result_integrated`
- `blue_brain_transition_evidence_observed_without_memory_commit`
- `blue_brain_transition_memory_adjacent_candidate_identified_not_committed`

Zusätzlich bleiben bestehende Trigger-/Block-/Suppress-Lanes bestehen, damit canonical submit/status
stabil bleibt.

## 5) Evidence-/Replay-Bezug vs Memory-Bezug

Explizit in Boundary-Lanes:
- evidence reference
- replay/reference basis
- context uptake
- no memory persistence implied

Replay- und Evidence-Surfaces bleiben referenzbasiert und outward-facing; sie sind kein Memory-Ersatz.

## 6) Internal-/Expert-only Context-Pfade

Internal-/Expert-Lanes sind explizit als non-canonical markiert:
- kein Blue-Brain-default Trigger
- kein kanonischer Context/Memory authority path
- nur via Down-Mapping auf outward status/evidence references nutzbar

## 7) Doku-Rückbindung

Diese Doku ist auf dieselben code-pinned Maps gebunden (`reference_map.rs`), ohne zweite
Wahrheitsquelle.

## 8) Zielbild für BB3-Vorbereitung (ohne Vorwegnahme)

Erreicht ist eine tragfähige Trennung:
- Compute
- Context
- Evidence/Replay References
- Memory-Adjacent Candidate (ohne Commit)

Nicht enthalten und bewusst deferred:
- eigene Memory-Engine
- Persistenz-/Storage-Subsystem
- Cognitive-State-Plattform
