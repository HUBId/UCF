# Serie BB1 Prompt 4: Erster echter Blue-Brain integration candidate (repo-basiert festgezogen)

Status: erster echter Blue-Brain-Integrationskandidat ist jetzt explizit ausgewählt und an dieselbe finale Compute-Linie rückgebunden.
Keine neue Core-Architektur, keine neue Integrationssprache, keine Governance-/Produkt-Ausbauarbeit.

Primäre Rückbindung (keine zweite Wahrheitsquelle):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_MAP`
  - `CANONICAL_BLUE_BRAIN_FACING_CONTRACT_MAP`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_HANDOFF_MAP`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface}`
- `runtime/ucf-compute/src/contracts.rs`
  - `runtime_handoff_state_from_evidence`
  - `runtime_handoff_state_from_action_code`

Finale Compute-Linie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- dieselbe status/evidence outward export surface
- expert/internal lanes bleiben non-canonical

## 1) Präziser Kandidatenvergleich (repo-treu)

Gegeneinander geprüft auf inference-/status-/evidence-/state-adjacent Anschluss:
- `runtime_orchestrator_stateful_loop`
  - real Blue-Brain-core-naher stateful consumer
  - nutzt die kanonische Compute-Linie als Ziel, hat aber noch schmale mixed intake Restkante
- `ops_compute_probe`
  - technisch sauberer outward consumer und Integrationsanker
  - aber bewusst **adjacent**, nicht Blue-Brain-Core-Orchestrierung
- `replay_diff_backend_recompute`
  - gemischte diagnostics/comparison surface
  - nicht als primäre outward Blue-Brain-Integrationsbasis geeignet
- `domains_ai_compat_lane`, `bench_compute_subcommand`, `runtime_hooks_and_frame_helpers`
  - legacy/internal/helper-nah
  - explizit kein echter Blue-Brain-Integrationskandidat jetzt

Auswahlentscheidung:
- **`runtime_orchestrator_stateful_loop`** ist als
  `selected_first_real_blue_brain_integration_candidate` festgezogen.
- Einstufung: **plausible with caveats** (nicht künstlich als vollständig integration-ready dargestellt).

## 2) Schmale Candidate-Map (genau 4 Klassen)

Es gelten genau diese minimalen Klassen:
1. `integration-ready candidate`
2. `plausible with caveats`
3. `mixed/transitional candidate`
4. `not a real Blue-Brain integration candidate now`

Keine Portfolio-Matrix. Keine zweite Architekturkarte.

## 3) Explizite Rückbindung des ausgewählten Kandidaten

Für `runtime_orchestrator_stateful_loop` gilt bindend:
- inference-facing:
  - `blue_brain_to_compute_inference_handoff`
  - `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
- status/trust/diagnostics-facing:
  - `blue_brain_to_compute_status_diagnostics_handoff`
  - `status + status_evidence_export_surface(status)`
- evidence/reference-facing:
  - `blue_brain_to_compute_evidence_reference_handoff`
  - `status_evidence_export_surface(evidence refs)` + runtime evidence-chain linkage
- state-adjacent:
  - `blue_brain_to_compute_state_adjacent_reference_handoff`
  - `context_digest + runtime_handoff_state_from_evidence/runtime_handoff_state_from_action_code`

Explizit **nicht** Teil des Kandidatenclaims:
- `replay_with_entry`
- `run_operation_with_entry`
- `build_backend(kind=stub|candle|worker)`
- `domains/ai*` compat lanes

## 4) Minimale Härtung der Restkanten

Prompt-4-Härtung bleibt absichtlich klein:
- ein dediziertes, code-gepinntes Candidate-Map-Objekt (`CANONICAL_BLUE_BRAIN_INTEGRATION_CANDIDATE_MAP`)
- explizite Exclusion-Semantik für mixed/legacy/internal-only Pfade
- tests, die sicherstellen, dass der ausgewählte reale Kandidat auf der kanonischen Handoff-/Contract-Linie bleibt

Keine neue Ausbauwelle.

## 5) Caveats (explizit und ehrlich)

Stabil:
- die canonical outward Compute-Linie und ihre inference/status/evidence semantics
- die Trennung canonical vs non-canonical expert/internal lanes

Constrained but acceptable:
- `runtime_orchestrator_stateful_loop` ist realer erster Kandidat, aber noch mit begrenzter mixed-intake Restkante

Bewusst nicht Teil dieses ersten Falls:
- Hodgkin-Huxley / Kuramoto
- neue kognitive Gesamtarchitektur
- Governance-/Produkt-/Auth-/Tenant-Ausbau
- Legacy-/compat-Pfade als Integrationsautorität

## 6) Doku-Linie

Prompt 4 erweitert Prompt 1-3 auf derselben Linie:
- gleiche Kandidatenbasis
- gleiche Contract-Sprache
- gleiche Handoff-Sprache
- jetzt plus expliziter erster Kandidat mit klaren Ausschlusskanten

Keine zweite Wahrheitsquelle.

## 7) Zielgerichtete Checks

Nur schmale Drift-Guards:
- 4 Candidate-Klassen sind vorhanden und stabil
- `runtime_orchestrator_stateful_loop` bleibt explizit ausgewählter realer Kandidat
- inference/status/evidence/state-adjacent Bindungen zeigen auf die kanonische Compute-Linie
- mixed/legacy/internal-only Flächen erscheinen nicht als primäre Kandidatenbasis

## Ergebnis (Prompt 4)

Der erste echte Blue-Brain-Integrationsfall ist jetzt technisch ehrlich festgezogen:
- realer Kandidat ausgewählt,
- kanonisch rückgebunden,
- legacy/internal-only Grenzen explizit,
- Ausbaupfad bleibt schmal und repo-basiert statt spekulativ.
