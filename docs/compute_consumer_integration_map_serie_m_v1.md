# Serie M: Compute Consumer Integration Map v1

Status: repo-basierte Abschlusssicht für die **erste** Post-Core-Integrationsreihe.

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
  - `DomainFacingConsumerAlignment`
  - `DomainFacingCompletionStatus`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface}`
  - `ComputeStatusEvidenceExportSurface::canonical_consumer_view()`

## 1) Alignment-Klassen (minimal)

- `aligned_canonical_outward`
- `legacy_compat_path`
- `needs_final_integration_adjustment`
- `internal_dev_test_only`

## 2) Completion-Status-Map (schmal, outward-orientiert)

- `aligned_to_final_compute_line`
- `mostly_aligned_with_caveats`
- `mixed_transitional`
- `internal_only_not_true_outward_consumer`

## 3) Consumer-seitige Consumption-Pattern (schmal)

- status pattern:
  - `canonical_status_consumer`
  - `mixed_legacy_consumption_pattern`
  - `internal_dev_test_only`
- evidence pattern:
  - `canonical_evidence_reference_consumer`
  - `mixed_legacy_consumption_pattern`
  - `internal_dev_test_only`

## 4) Reale Consumer-Map (ausgewählt, repo-treu)

1. `runtime_orchestrator_env_bootstrap`
   - repo surface: `runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::try_new_from_env`
   - execution: `build_backend(cfg from env)`
   - status/evidence: `compute summary -> runtime orchestration state` + `compute_summary.compute_chain_digest + runtime evidence chain`
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `needs_final_integration_adjustment`
   - completion status: `mostly_aligned_with_caveats`
   - caveat: load-bearing Runtime-Consumer; unterstützt noch compat backend kinds und ist deshalb noch nicht voll auf `submit/status_evidence_export_surface`.

2. `ops_compute_probe`
   - repo surface: `runtime/ucf-ops/src/lib.rs::run_compute_probe`
   - execution: `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
   - status/evidence: `CanonicalComputeEntryPoint::status + status_evidence_export_surface`
   - status pattern: `canonical_status_consumer`
   - evidence pattern: `canonical_evidence_reference_consumer`
   - alignment: `aligned_canonical_outward`
   - completion status: `aligned_to_final_compute_line`
   - caveat: bewusst constrained; konsumiert top-level Status-/Evidence-Refs statt tiefer Internals.

3. `replay_diff_backend_recompute`
   - repo surface: `runtime/ucf-replay/src/lib.rs::replay_records`
   - execution: `build_backend(cfg from replay spec) -> backend.compute(...)`
   - status/evidence: replay/diff policy comparison + persisted evidence refs
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `legacy_compat_path`
   - completion status: `mixed_transitional`
   - caveat: bewusst compat-/replay-orientierte Recompute-Lane, kein outward Runtime-Servicevertrag.

4. `bench_compute_subcommand`
   - repo surface: `runtime/ucf-bench/src/main.rs::run_compute`
   - execution: `build_backend(cfg) -> backend.compute(...)`
   - status/evidence: Benchmarkmetriken
   - status pattern: `internal_dev_test_only`
   - evidence pattern: `internal_dev_test_only`
   - alignment: `internal_dev_test_only`
   - completion status: `internal_only_not_true_outward_consumer`
   - caveat: internes Harness, kein normaler Domänenanschluss.

5. `domains_ai_compat_lane`
   - repo surface: `domains/ai* + domains/ai-backends compatibility crates`
   - execution/status/evidence: legacy host ABI adapter lane
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `legacy_compat_path`
   - completion status: `internal_only_not_true_outward_consumer`
   - caveat: explizite Compatibility-Grenze außerhalb outward-facing Compute-Verträge.

## 5) Serie-M-Abschlussmatrix (erste Post-Core-Reihe)

| status | surfaces | repo-basierte Aussage |
|---|---|---|
| **aligned post-core integration** | `ops_compute_probe` | real auf finaler Compute-Linie: submit + status/evidence export + canonical consumer semantics. |
| **integration-usable with caveats** | `runtime_orchestrator_env_bootstrap` | technisch nutzbar, aber weiterhin Env/compat-backend Pfad statt durchgehend canonical submit/export line. |
| **mixed / transitional** | `replay_diff_backend_recompute` | gezielt replay-/compat-Lane; nicht als outward canonical runtime contract geführt. |
| **intentionally deferred** | `bench_compute_subcommand`, `domains_ai_compat_lane` | bench bleibt internal harness, domains compat bleibt legacy seam; beide bewusst außerhalb dieser Post-Core-Reihe. |

## 6) Erste Post-Core-Integrationslinie (explizit)

Diese erste Reihe endet mit folgender harten Linie:

- **Sauber auf finaler Compute-Linie:** `ops_compute_probe`.
- **Bewusst caveated/transitional:**
  - `runtime_orchestrator_env_bootstrap` (load-bearing, aber noch mixed legacy contract intake),
  - `replay_diff_backend_recompute` (compat/replay lane).
- **Bewusst deferred außerhalb der Reihe:**
  - `bench_compute_subcommand`,
  - `domains_ai_compat_lane`.

Ab hier ist weitere Arbeit **breitere Systemintegration** auf UCF-Ebene (Consumer-Rollout/Harmonisierung), nicht mehr Compute-Core-Abschlussarbeit.

## 7) Nächste Richtungen nach Serie M (nur technische Hebel)

1. **Serie N (priorisiert): broader UCF system integration review**
   - Hebel: `runtime_orchestrator_env_bootstrap` ist load-bearing und aktuell der wichtigste caveated Consumer.
2. **Serie P (nachrangig): targeted domain rollout auf stabilisierter Compute-Integration**
   - Hebel: weitere domain-facing Consumers nacheinander auf canonical submit + canonical status/evidence semantics bringen.
3. **Serie O (nachrangig): maintenance-only follow-up lane**
   - Hebel: nur Drift-Vermeidung/Locking; kein großer Integrationssprung.

### Priorisierung (genau eine zuerst)

**Zuerst: Serie N.**

Kurzbegründung:
- höchster unmittelbarer Hebel, weil der runtime orchestrator die lasttragende Integrationskante bleibt,
- Serie P wird wirksamer, wenn dieser zentrale Consumer vorher weiter canonicalisiert ist,
- Serie O ist sinnvoll, aber ohne zusätzliche Integrationswirkung und daher nachrangig.

## 8) Deliberate limits

- Keine neue Integrationsplattform.
- Keine neue Consumer-spezifische Vertragswelt.
- Keine Produkt-API-/Auth-/Billing-/Governance-Ausweitung.
- Canonical outward-facing Verträge bleiben: Execution + Status/Diagnostics + Evidence/References.
- Consumer-seitige Caveat-Semantik bleibt auf derselben Top-Level-Sprache:
  `current/trusted`, `caveated|partial`, `degraded|unavailable`, `insufficient evidence references`.
