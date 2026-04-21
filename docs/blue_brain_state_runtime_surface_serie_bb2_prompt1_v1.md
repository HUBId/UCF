# Serie BB2 Prompt 1: Kanonische Blue-Brain State/Runtime Surface (repo-basiert, schmal)

Status: minimale, belastbare Blue-Brain state/runtime surface über der finalen Compute-Linie.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_SURFACE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_PHASE_MAP`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP`
- `runtime/ucf-runtime/src/orchestrator.rs`
- `runtime/ucf-compute/src/service_surface.rs`

Finale Compute-Referenzlinie (verbindlich):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

Diese BB2-Surface bleibt strikt auf derselben finalen Compute-Linie. Es gibt **keine zweite Compute-Semantik**, keine zweite Wahrheitsquelle und keine Workflow-Engine.

## 1) Repo-treue Prüfung der realen Blue-Brain-seitigen Flächen

### Realer Kernkandidat aus BB1
- `runtime_orchestrator_stateful_loop` bleibt der erste reale Blue-Brain Integrationskandidat.
- Er ist stateful/orchestration-nah und compute-kopplungsrelevant, aber weiterhin caveated, solange Residualpfade (env/compat intake) nicht vollständig auf canonical submit/status-evidence gebunden sind.

### Stateful/orchestration-nahe/model-consuming Flächen
- `runtime/ucf-runtime/src/orchestrator.rs` trägt state/context und steuert Compute-Anfragen.
- `runtime/ucf-ops/src/lib.rs::run_compute_probe` bleibt ein outward-aligned Referenzkonsument, aber kein Blue-Brain-Core-Loop.

### Blue-Brain-facing Contracts/Handoffs aus BB1
- Inference: `CanonicalComputeEntryPoint::submit`.
- Status/Health/Trust: `CanonicalComputeEntryPoint::status + status_evidence_export_surface(status)`.
- Evidence/Replay-Referenzen: `status_evidence_export_surface (evidence refs)`.
- State-adjacent bleibt referenzbasiert (`context_digest`, `runtime_handoff_state_from_evidence`, `runtime_handoff_state_from_action_code`).

### Wo Semantik heute verschwimmen könnte
- state vs inference: orchestrator-nahe Vorbereitung vs tatsächliche Compute-Invocation.
- status/diagnostics vs evidence/replay: beide laufen über dieselbe outward surface, müssen aber unterschiedlich konsumiert werden.
- internal/expert control vs outward runtime: `replay_with_entry`, `run_operation_with_entry`, `build_backend(...)`, `domains/ai*` sind nicht Teil der kanonischen Blue-Brain-Runtime-Surface.

## 2) Schmale kanonische Blue-Brain Runtime-Surface-Map

Die Surface bleibt auf fünf minimalen Klassen:

1. `blue_brain_state_bearing_surface`
   - State/context-tragende Orchestrierungsfläche.
2. `blue_brain_inference_bearing_surface`
   - Kanonische Inference-Anforderung über `submit`.
3. `blue_brain_status_health_trust_surface`
   - Runtime-relevante Status-/Health-/Trust-Signale (`current|partial|stale|caveated|degraded`).
4. `blue_brain_evidence_replay_facing_surface`
   - Evidence-/Trace-/History-Referenzaufnahme (`sufficient|partial|caveated|insufficient`).
5. `blue_brain_internal_only_runtime_control_surface`
   - explizit non-canonical internal/expert lane.

Damit laufen state, inference, status, evidence und internal control nicht mehr implizit ineinander.

## 3) Minimale kanonische Blue-Brain Laufphasen

Die Phase-Sicht bleibt bewusst klein:

1. `blue_brain_phase_state_context_available`
   - state/context verfügbar, referenzierbar.
2. `blue_brain_phase_compute_invocation_requested`
   - Compute-Aufruf angefordert via canonical submit.
3. `blue_brain_phase_compute_result_integrated`
   - Result/Fault/Status in Runtimezustand integriert.
4. `blue_brain_phase_status_evidence_observed`
   - status/evidence aus outward surface beobachtet und aufgenommen.
5. `blue_brain_phase_caveated_degraded_partial_runtime_state`
   - caveated/degraded/partial Laufzeitpostur explizit markiert (ohne versteckte Expert-Eskalation).

Keine Workflow-Engine, keine spekulative Kognitionspipeline.

## 4) Rückbindung an die finale Compute-Linie

- Outward-facing Compute contracts bleiben: `submit`, `status`, `status_evidence_export_surface`, `integration_hook_view`.
- Blue-Brain state wird referenziert/kontextualisiert, aber compute-intern nicht nachmodelliert.
- Runtime-Surface stoppt an der outward contract boundary; compute-interne Runtime-Details leaken nicht als Blue-Brain-Standardpayload.

## 5) Status-/Trust-/Evidence-Rückfluss

Blue-Brain Runtime nimmt nicht nur Invocation vor, sondern führt Rücksignale zurück:
- current/caveated/degraded/partial/stale Status- und Trust-Signale.
- Evidence-/Trace-/History-Referenzaufnahme inkl. sufficient/partial/caveated/insufficient.
- bei partial/stale/insufficient: explizite caveated/degraded Runtime-Postur statt impliziter Normalisierung.

## 6) Interne/Expert-only Control-Surfaces explizit ausgegrenzt

Explizit non-canonical für Blue-Brain-Runtime:
- `service_surface::{replay_with_entry, run_operation_with_entry}`
- `backends::build_backend(kind=stub|candle|worker)`
- `domains/ai*` compatibility lanes

Diese Pfade dürfen nicht als kanonische Blue-Brain-Runtime präsentiert werden.

## 7) Doku-Rückbindung (eine Wahrheitsquelle)

Diese BB2-Doku ist direkt auf die code-pinned Maps in `reference_map.rs` gebunden.
Keine zweite Wahrheitsquelle.

## 8) Kleine Konsistenzchecks

Ergänzt wurden schmale Checks für:
- Präsenz und Vollständigkeit der fünf Runtime-Surface-Klassen.
- Präsenz und Vollständigkeit der fünf minimalen Laufphasen.
- Pinning auf finale Compute-Linie (`submit -> compute_canonical -> result/fault/status`).
- explizite Ausgrenzung von internal/expert-only runtime control als non-canonical.

## Kurzfazit

BB2 Prompt 1 etabliert eine schmale, repo-treue Blue-Brain state/runtime surface oberhalb des abgeschlossenen Compute-Kerns. Sie trennt state/inference/status/evidence/internal control sauber, bleibt auf outward-facing Compute-Verträgen und schafft eine belastbare Basis für die nächsten BB2-Schritte ohne Overengineering.
