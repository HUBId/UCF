# Serie BB1 Prompt 3: Erste kanonische Blue-Brain-to-Compute Handoff-Semantik (schmal, repo-treu)

Status: erste echte Übergabepunkte zwischen Blue-Brain-nahen Flächen und abgeschlossenem Compute-Kern sind explizit konsolidiert.
Keine neue Workflow-Engine, keine zweite Execution-Welt, keine Governance-/Produkt-API-Ebene.

Primäre Rückbindung (keine zweite Wahrheitsquelle):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP`
  - `CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_MAP`
  - `CANONICAL_FINAL_REFERENCE_LINE`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface}`
- `runtime/ucf-compute/src/contracts.rs`
  - `runtime_handoff_state_from_evidence`
  - `runtime_handoff_state_from_action_code`

Feste Compute-Linie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- gleiche outward-facing Status-/Evidence-Sprache
- expert/internal lanes bleiben technisch vorhanden, aber non-canonical

## 1) Reale Blue-Brain-to-Compute Übergabepunkte (repo-präzise)

Faktisch bestehende oder naheliegende Handoffs:
- inference-facing: `submit`-Pfad über `CanonicalComputeEntryPoint`
- status/diagnostics/trust-facing: `status` + `status_evidence_export_surface (status)`
- evidence/reference-facing: `status_evidence_export_surface (evidence refs)`
- state-adjacent: `context_digest` + handoff-state mapping
- non-canonical: `replay_with_entry`, `run_operation_with_entry`, `build_backend(...)`, compat lanes

Damit werden implizite Mischungen (interne Diagnoseobjekte, expert-only runtime ops, compat helper paths) als non-canonical Grenze sichtbar gehalten.

## 2) Schmale Blue-Brain Handoff-Map (genau 5 Klassen)

Es gelten genau diese minimalen Klassen:
1. `blue_brain_to_compute_inference_handoff`
2. `blue_brain_to_compute_status_diagnostics_handoff`
3. `blue_brain_to_compute_evidence_reference_handoff`
4. `blue_brain_to_compute_state_adjacent_reference_handoff`
5. `blue_brain_non_canonical_expert_internal_handoff`

Keine Workflow-Engine; nur explizite Handoff-Klassifikation auf der bestehenden Compute-Vertragslinie.

## 3) Inference handoff semantics (kanonisch)

Outbound von Blue Brain:
- nur canonical submit request envelope (`ComputeSubmitRequest{ExecuteInline}`)

Return von Compute:
- canonical result/fault/status semantics
- execution snapshot auf derselben outward Linie

Explizit ausgeschlossen:
- zweite Execution-Welt
- implizite Expert-/Replay-Semantik als Inference-Standard

## 4) Status-/Trust-/Diagnostics handoff semantics

Kanonischer Rückkanal:
- `status + status_evidence_export_surface (status)`

Verbindliche Top-level-Signale:
- `current / partial / stale / caveated / degraded`
- trust/service state auf derselben outward Oberfläche

Explizit ausgeschlossen:
- compute-interne Diagnosewelt als Pflicht-Handoff-Payload

## 5) Evidence-/Reference handoff semantics

Kanonischer Rückkanal:
- `status_evidence_export_surface (evidence refs)`

Verbindliche Nutzlast:
- evidence bundle references
- trace/evidence references (outward-facing relevant)
- partial/caveated evidence semantics

Explizit ausgeschlossen:
- rohe interne Diagnose-/Trace-Objekte als Blue-Brain-facing Standard

## 6) State-adjacent handoff semantics (nur minimal)

Kanonischer Kontextbezug:
- request `context_digest`
- `runtime_handoff_state_from_evidence`
- `runtime_handoff_state_from_action_code`

Verbindliche Rückgabe:
- state-adjacent handoff refs mit `complete|partial|caveated|blocked`

Explizit ausgeschlossen:
- spekulative Cognitive-State-Plattform
- Leckage compute-interner Runtime-Strukturen

## 7) Non-canonical Handoffs klar ausgegrenzt

Bleiben non-canonical für Blue-Brain-facing Standardhandoff:
- expert-only paths: `run_operation_with_entry`, `replay_with_entry`
- internal backend plumbing: `build_backend(kind=stub|candle|worker)`
- legacy/compat adapters: `domains/ai*`

Falls diese Pfade genutzt werden, müssen Outputs vor Blue-Brain-facing Nutzung auf die outward canonical status/evidence references zurückgemappt werden.

## 8) Referenzobjekte an den Handoffs (keine neue Objektwelt)

Kernreferenzen bleiben durchgängig:
- request/run identity (`ComputeJobHandle`-gebundene Referenzspur)
- snapshot/evidence references
- outward-facing status references
- active production context nur soweit load-bearing

## 9) Doku-Rückbindung

Prompt 3 erweitert Prompt 1/2, ersetzt sie nicht:
- gleiche Blue-Brain-Kandidatenbasis
- gleiche Blue-Brain-facing Vertragsbasis
- jetzt zusätzlich explizite Handoff-Semantiklinie

## 10) Kleine Konsistenzchecks (minimal)

Nur kleine Drift-Guards:
- alle 5 Handoff-Klassen vorhanden
- inference/status/evidence/state-adjacent bleiben getrennt
- inference/status/evidence bleiben auf derselben finalen Compute-Linie
- expert/internal-only bleibt explizit non-canonical

## Ergebnis (Prompt 3)

Damit ist die erste kanonische Blue-Brain-to-Compute Handoff-Schicht belastbar und schmal:
- Übergabepunkte explizit statt implizit,
- inference/status/evidence/state-adjacent sauber getrennt,
- non-canonical expert/internal Pfade klar ausgegrenzt,
- keine Workflow-Engine und keine zweite Wahrheitsquelle.
