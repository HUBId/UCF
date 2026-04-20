# Serie BB1 Prompt 1: Blue-Brain Integration Map (repo-basiert, schmal)

Status: technisches Mapping des realen Blue-Brain-Systemkerns gegen den abgeschlossenen Compute-Kern.
Keine neue Großarchitektur, keine zweite Integrationssprache, kein Compute-Core-Ausbau.

Primäre Rückbindung (keine zweite Wahrheitsquelle):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_FINAL_REFERENCE_LINE`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
  - `CANONICAL_BLUE_BRAIN_INTEGRATION_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`
- `docs/real_compute_exit_dossier_serie_l_v1.md`

Festes Integrationsfundament bleibt:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- `compute_execution_contract`
- `compute_status_diagnostics_contract`
- `compute_evidence_reference_contract`
- `integration_hook_view` (read-only/caveated Boundary)

## 1) Reale Blue-Brain-nahe Systemflächen (repo-treu)

### A) Real Blue-Brain core candidate
- `runtime_orchestrator_stateful_loop`
  - Oberfläche: `runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::{try_new_from_env,step_once}`
  - Warum real Blue-Brain-nah:
    - stateful orchestration + model-consuming control loop.
    - faktischer Verbraucher von Compute-Signalen für Laufzeitentscheidungen.
  - Reife heute:
    - hoher technischer Hebel, aber Mixed-Intake (noch nicht rein über outward contracts).

### B) Blue-Brain-adjacent compute consumer (nicht Core)
- `ops_compute_probe`
  - Oberfläche: `runtime/ucf-ops/src/lib.rs::run_compute_probe`
  - Warum relevant:
    - sauberer outward Consumer über canonical submit/status/evidence.
  - Warum nicht Blue-Brain-Core:
    - Probe/Diagnosefläche, kein stateful kognitiver Orchestrierungskern.

### C) Indirect or compatibility-touching surfaces
- `replay_diff_backend_recompute` (`runtime/ucf-replay/src/lib.rs::replay_records`)
  - technisch nützlich für Vergleich/Drift, aber kein primärer outward Service-Consumer.
- `domains_ai_compat_lane` (`domains/ai*`, `domains/ai-backends`)
  - legacy/compat boundary, nicht canonical Blue-Brain-Integrationsbasis.

### D) Internal-only / nicht sinnvoll für Blue-Brain-Integration
- `bench_compute_subcommand` (`runtime/ucf-bench/src/main.rs::run_compute`)
  - Benchmark-Harness, kein outward Integrationsvertrag.
- `runtime_hooks_and_frame_helpers` (`runtime/ucf-runtime/src/hooks.rs`, `domains/ucf-frames/src/v1/*`)
  - helper/schema proximity, aber keine eigenständige outward compute-consumer Autorität.

## 2) Schmale Klassen (minimal)

Die BB1-Klassifikation bleibt absichtlich auf vier Klassen begrenzt:

1. `real_blue_brain_core_candidate`
2. `blue_brain_adjacent_compute_consumer`
3. `indirect_or_compatibility_touching_surface`
4. `internal_only_or_not_meaningful_for_blue_brain_integration`

## 3) Kanonische Integrationsgrenzen (Compute abgeschlossen, maintenance-only)

Alle BB1-Einordnungen sind auf dieselben outward-facing Compute-Verträge gebunden:

- Execution: `compute_execution_contract` (`CanonicalComputeEntryPoint::submit`)
- Status/Diagnostics: `compute_status_diagnostics_contract` (`status_evidence_export_surface` status)
- Evidence/Reference: `compute_evidence_reference_contract` (`status_evidence_export_surface` evidence)
- Integration hooks: `integration_hook_view` als read-only/caveated Grenze

Explizit ausgeschlossen:
- legacy/compat lanes als primäre Blue-Brain-Integrationsautorität,
- expert/internal mutating hooks als outward Integrationsvertrag.

## 4) Implizite/unsaubere Kopplungen explizit markiert

Heute caveated bzw. unsauber getrennt:

1. `runtime_orchestrator_stateful_loop`
   - Mixed-Intake über env/backend-Pfade + summary-basierte Kopplung.
   - Zielbild: progressive Bindung auf `submit + status_evidence_export_surface`, ohne zweiten Vertrag.
2. `replay_diff_backend_recompute`
   - indirekte compute-Nähe, aber semantisch Vergleichspfad, kein outward runtime contract.
3. `domains_ai_compat_lane`
   - historisch/kompatibilitätsgetriebene Kopplung statt canonical outward line.

## 5) Die 1–3 wichtigsten echten Blue-Brain-Integrationskandidaten

Nur reale Kandidaten mit technischem Hebel:

1. **`runtime_orchestrator_stateful_loop`** (Top-Kandidat)
   - echter stateful orchestration Kern, faktischer Compute-Consumer.
   - größter Hebel für reale Blue-Brain-Integration.
2. **`ops_compute_probe`** (Referenzanker, adjacent)
   - kein Core-Kandidat, aber zentral zur Stabilisierung derselben outward Contracts.
3. **`replay_diff_backend_recompute`** (nur caveated Kandidat)
   - diagnostischer Hebel, aber bewusst kein primärer Blue-Brain-Outward-Consumer.

Warum andere Flächen aktuell nicht geeignet sind:
- `domains_ai_compat_lane`: legacy/compat getrieben.
- `bench_compute_subcommand`: internal/dev-only.
- `runtime_hooks_and_frame_helpers`: helper-nah, aber kein eigenständiger Integrationsvertrag.

## 6) Doku-Rückbindung

BB1 ergänzt die bestehende Exit-/Integrationslinie, ersetzt sie nicht:
- Serie L Exit-Doku bleibt authoritative für die abgeschlossene Compute-Linie.
- Diese BB1-Map konkretisiert nur den Blue-Brain-Integrationsanschluss auf derselben Vertragsbasis.
- Keine zweite Wahrheitsquelle.

## 7) Kleine Konsistenzchecks (minimal)

Die Minimalchecks bleiben:

1. Blue-Brain-Klassen müssen vollständig und exklusiv unterscheidbar bleiben.
2. `real_blue_brain_core_candidate` darf nicht ohne outward contract basis markiert werden.
3. legacy/internal Flächen dürfen nicht als outward-aligned Blue-Brain-Integration erscheinen.
4. Doku muss auf dieselbe finale Compute-Linie gebunden bleiben.

Technisch gespiegelt über:
- `runtime/ucf-compute/src/reference_map.rs` (`CANONICAL_BLUE_BRAIN_INTEGRATION_MAP` + Tests)

## 8) Ergebnis-Snapshot (ehrlich, schmal)

- Reale Blue-Brain-Kernkandidat-Fläche: **`runtime_orchestrator_stateful_loop`**
- Compute-adjacent Referenz-Consumer: **`ops_compute_probe`**
- Caveated/indirekt: **`replay_diff_backend_recompute`**
- Legacy/internal-only: **`domains_ai_compat_lane`**, **`bench_compute_subcommand`**, **`runtime_hooks_and_frame_helpers`**

Damit ist der Blue-Brain-Integrationsfundament-Start klar auf derselben finalen Compute-Vertragsfläche fixiert, ohne neue Architekturspur.
