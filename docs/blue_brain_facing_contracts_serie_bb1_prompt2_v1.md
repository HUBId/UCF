# Serie BB1 Prompt 2: Blue-Brain-facing Contracts (state/inference/status/evidence, schmal und repo-treu)

Status: Konsolidierung der Blue-Brain-facing Vertragslinie auf dem abgeschlossenen Compute-Kern.
Keine neue Architekturwelt, keine zweite Execution-Sprache, kein Compute-Core-Ausbau.

Primäre Rückbindung (keine zweite Wahrheitsquelle):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_MAP`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_FINAL_REFERENCE_LINE`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`
- `docs/blue_brain_integration_map_serie_bb1_prompt1_v1.md`

Festes Compute-Fundament bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- dieselben outward-facing top-level semantics
- klar getrennte internal/expert lanes

## 1) Outward-facing Bestand gegen Blue-Brain-Bedarf

Die relevanten outward-facing Compute-Verträge sind bereits vorhanden und bleiben kanonisch:
- execution contract: `CanonicalComputeEntryPoint::submit`
- status/diagnostics contract: `status + status_evidence_export_surface (status)`
- evidence/reference contract: `status_evidence_export_surface (evidence refs)`
- integration-safe hooks: `integration_hook_view` (read-only/caveated outward boundary)

Repo-treue Einordnung aus Prompt 1 bleibt aktiv:
- real core candidate: `runtime_orchestrator_stateful_loop`
- adjacent outward anchor: `ops_compute_probe`
- indirect/compat: `replay_diff_backend_recompute`, `domains_ai_compat_lane`
- internal-only: `bench_compute_subcommand`, `runtime_hooks_and_frame_helpers`

## 2) Schmale Blue-Brain-facing Contract-Map (minimal classes)

Es gelten genau fünf Klassen:

1. `blue_brain_inference_facing_execution_contract`
2. `blue_brain_state_facing_context_reference_contract`
3. `blue_brain_status_health_trust_contract`
4. `blue_brain_evidence_reference_contract`
5. `blue_brain_expert_internal_only_non_contract`

Keine neue API-Familie; nur präzisere Klassifizierung derselben bestehenden Compute-Outward-Line.

## 3) Inference-facing Contract (kanonisch, ohne zweite Execution-Welt / no second execution world)

Kanonischer Pfad:
- `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
- result/fault/status-Semantik bleibt identisch zur finalen Compute-Linie.

Erlaubte Semantik:
- Inferenz-/Ausführungsnutzung auf `submit -> compute_canonical -> result/fault/status`.

Ausgeschlossen:
- `build_backend(kind=stub|candle|worker)` als Blue-Brain-facing Standard.
- replay/expert operation semantics als impliziter Inference-Standard.

## 4) State-facing Contract (nur state-adjacent, repo-basiert)

Kanonischer Fokus:
- request context (`context_digest`) + runtime-handoff state references
  (`runtime_handoff_state_from_evidence`, `runtime_handoff_state_from_action_code`).

Erlaubte Semantik:
- state-adjacent reference/context linkage für Integrationsanschluss.

Ausgeschlossen:
- spekulative Cognitive-State-Gesamtarchitektur.
- Leaken compute-interner Scheduler-/Runtime-Strukturen als outward Vertrag.

## 5) Status-/Trust-/Diagnostics-facing Contract

Kanonischer Pfad:
- `CanonicalComputeEntryPoint::status + status_evidence_export_surface (status)`

Verbindliche Top-level Signale:
- `current / partial / stale / caveated / degraded`
- trust/service status auf derselben outward Oberfläche.

Ausgeschlossen:
- interne Diagnosegraphen als verpflichtender Blue-Brain-facing Surface.

## 6) Evidence-/Reference-facing Contract

Kanonischer Pfad:
- `CanonicalComputeEntryPoint::status_evidence_export_surface (evidence refs)`

Erlaubte Semantik:
- evidence bundle references
- trace/evidence references (outward-facing relevant)
- caveated/partial evidence posture

Ausgeschlossen:
- rohe interne diagnostics/trace Objekte als Blue-Brain-facing Standardpayload.

## 7) Expert-/Internal-only Pfade explizit ausgeschlossen

Nicht Blue-Brain-facing Standardverträge:
- `replay_with_entry`, `run_operation_with_entry`
- `build_backend(kind=stub|candle|worker)`
- `domains/ai*` compatibility lane

Diese Pfade bleiben technisch vorhanden, aber als expert/internal-only markiert.

## 8) Doku-Rückbindung

Diese Prompt-2-Konsolidierung erweitert Prompt 1, ersetzt ihn nicht:
- gleiche Kandidatenlogik,
- gleiche finale Compute-Linie,
- gleiche outward Anchors.

Die Doku bleibt an Codekonstanten in `reference_map.rs` gebunden; keine zweite Wahrheitsquelle.

## 9) Kleine Konsistenzchecks (minimal)

Ergänzt wurden nur kleine Drift-/Boundary-Checks:
- alle fünf Contract-Klassen vorhanden,
- inference bleibt auf canonical submit + result/fault/status core,
- status/evidence bleiben auf `status_evidence_export_surface`,
- expert/internal-only lane bleibt explizit non-contract für Blue-Brain-facing Standard.

## Ergebnis (Prompt 2)

Damit ist die Blue-Brain-facing Grundschicht jetzt schmal und belastbar festgezogen:
- inference-, state-, status-/trust- und evidence-facing Nutzung sind getrennt und benannt,
- compute-interne/expert-only Semantik ist explizit ausgegrenzt,
- spätere Blue-Brain-Subsysteme können auf derselben finalen Contract-Linie aufbauen.
