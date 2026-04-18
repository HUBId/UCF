# Serie N: Broader UCF System Integration Map v1 (repo-basiert, schmal)

Status: repo-basierte Einordnung breiterer UCF-Systemflächen gegen die **finale Compute-Linie** ohne neue Ausbauwelle.

Diese Datei ergänzt Serie M um die breitere Systemsicht und bleibt bewusst auf derselben
Referenzsprache aus Serie K/M:

- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- `status_evidence_export_surface` als outward Status-/Evidence-Export
- `integration_hook_view` als read-only/caveated Hook-Grenze
- `compatibility backends + internal/legacy worker/domain lanes are extension/internal only`

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_FINAL_REFERENCE_LINE`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`
- `runtime/ucf-ops/src/lib.rs` (`run_compute_probe`)
- `runtime/ucf-runtime/src/orchestrator.rs` (`RuntimeOrchestrator::try_new_from_env`)
- `runtime/ucf-replay/src/lib.rs` (`replay_records`)
- `runtime/ucf-bench/src/main.rs` (`run_compute`)

## 1) Minimale Klassen (Serie-N-Breitsicht)

- `real_compute_facing_candidate`
- `indirect_or_compatibility_touching_surface`
- `internal_only_relation`
- `no_meaningful_compute_integration_candidate`

## 2) Repo-basierte Systemflächen (breiter, aber schmal klassifiziert)

| surface | primäre Repo-Fläche | Klasse | kurze Einordnung gegen finale Compute-Linie |
|---|---|---|---|
| `ops_compute_probe` | `runtime/ucf-ops/src/lib.rs::run_compute_probe` | `real_compute_facing_candidate` | Nutzt `CanonicalComputeEntryPoint::submit` plus `status_evidence_export_surface` und mappt auf canonical consumer semantics. |
| `runtime_orchestrator_env_bootstrap` | `runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::try_new_from_env` | `real_compute_facing_candidate` | Lasttragender Runtime-Anschluss, aber heute noch env/`build_backend`-intake statt durchgehend canonical submit/export line. |
| `replay_diff_backend_recompute` | `runtime/ucf-replay/src/lib.rs::replay_records` | `indirect_or_compatibility_touching_surface` | Recompute-/Diff-Lane über `build_backend` für Replay-Vergleich; technisch relevant, aber kein outward Runtime-Servicevertrag. |
| `bench_compute_subcommand` | `runtime/ucf-bench/src/main.rs::run_compute` | `internal_only_relation` | Internes Benchmark-Harness (`build_backend -> backend.compute`) ohne outward Status-/Evidence-Vertrag. |
| `domains_ai_compat_lane` | `domains/ai*`, `domains/ai-backends`, `domains/ai-host-abi` | `indirect_or_compatibility_touching_surface` | Historische/kompatible Adaptergrenze; nicht an canonical status/evidence export hooks gebunden. |
| `runtime hooks / frame helpers` | z. B. `runtime/ucf-runtime/src/hooks.rs` | `no_meaningful_compute_integration_candidate` | Konsumiert persistierte Summary-/Frame-Daten, aber kein stabiler outward Compute-Integrationsvertrag. |

## 3) Explizite Legacy-/Scheinintegration-Markierung

Folgende Flächen sind **nicht** als direkte nächste outward Integrationskandidaten zu lesen:

- `domains_ai_compat_lane`: compatibility seam, bleibt außerhalb canonical outward contracts.
- `replay_diff_backend_recompute`: technisch valide Replay-Lane, aber als Vergleichspfad statt produktiver
  domain-facing contract.
- `bench_compute_subcommand`: intern/dev/test-only Harness.
- runtime-interne Helperpfade mit `compute_summary`-Lesen ohne `status_evidence_export_surface`-Bindung.

## 4) Kleine Priorisierung (nur echte Hebel, keine Wunschliste)

1. `runtime_orchestrator_env_bootstrap` (**höchster Hebel**)
   - weil load-bearing Runtime-Eintritt und derzeit wichtigster verbleibender Mixed-Intake-Pfad.
2. `ops_compute_probe` (**Stabilitäts-/Driftanker beibehalten**)
   - weil bereits sauber aligned und als Referenzconsumer für Serie-N-Folgeschritte dient.
3. `replay_diff_backend_recompute` (**nur klar abgegrenzt halten**)
   - nicht als outward candidate aufblasen, sondern bewusst als indirect/compat surface markieren.

## 5) Rückbindung an bestehende finale Integrationsdoku

- Serie M bleibt erste Post-Core-Integrationsreihe (`docs/compute_consumer_integration_map_serie_m_v1.md`).
- Diese Serie-N-Datei erweitert die Sicht nur um breitere Repo-Systemflächen und nutzt dieselben
  finalen Anchors aus Serie K/M.
- Keine zweite Integrationssprache, keine neue Gesamtarchitekturkarte.

## 6) Direkte nächste Schritte aus dieser Kartierung (Serie N)

1. Orchestrator-Intake schrittweise auf canonical submit/status-evidence line ausrichten (ohne Big-Bang).
2. Für orchestrator-nahe Konsumenten explizit kennzeichnen, wo nur summary/internal helper data gelesen wird.
3. Replay-Lane dokumentarisch als indirect/compat boundary fixieren (kein outward-contract uplift).
4. `domains/ai*`-Kompatibilitätspfade explizit als legacy/compat maintainen, nicht als primären Integrationspfad.
5. Bei jedem neuen Consumer dieselben vier Klassen erzwingen, bevor Integrationspriorität vergeben wird.
