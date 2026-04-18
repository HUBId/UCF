# Serie M: Compute Consumer Integration Map v1

Status: schmale, repo-treue Consumer-Map für reale compute-konsumierende Flächen nach dem finalen Compute-Core-Abschluss.

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
  - `DomainFacingConsumerAlignment`

## Alignment-Klassen (minimal)

- `aligned_canonical_outward`
- `legacy_compat_path`
- `needs_final_integration_adjustment`
- `internal_dev_test_only`

## Consumer-seitige Consumption-Pattern (schmal)

- status pattern:
  - `canonical_status_consumer`
  - `mixed_legacy_consumption_pattern`
  - `internal_dev_test_only`
- evidence pattern:
  - `canonical_evidence_reference_consumer`
  - `mixed_legacy_consumption_pattern`
  - `internal_dev_test_only`

## Reale Consumer-Map (ausgewählt)

1. `runtime_orchestrator_env_bootstrap`
   - repo surface: `runtime/ucf-runtime/src/orchestrator.rs::RuntimeOrchestrator::try_new_from_env`
   - execution: `build_backend(cfg from env)`
   - status/evidence: runtime-seitige Zusammenführung aus Compute-Summary/Evidence-Chain
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `needs_final_integration_adjustment`
   - caveat: load-bearing Runtime-Consumer mit weiter bestehender Compat-Backend-Unterstützung.

2. `ops_compute_probe`
   - repo surface: `runtime/ucf-ops/src/lib.rs::run_compute_probe`
   - execution: `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
   - status/evidence: `status + status_evidence_export_surface`
   - status pattern: `canonical_status_consumer`
   - evidence pattern: `canonical_evidence_reference_consumer`
   - alignment: `aligned_canonical_outward`
   - caveat: bewusst constrained; konsumiert top-level Status-/Evidence-Refs statt tiefer Internals.

3. `replay_diff_backend_recompute`
   - repo surface: `runtime/ucf-replay/src/lib.rs::replay_records`
   - execution: `build_backend(cfg from replay spec) -> backend.compute(...)`
   - status/evidence: replay- und diff-orientierte Referenzsicht
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `legacy_compat_path`
   - caveat: bewusst compat-orientierte Recompute-Lane, nicht als outward Runtime-Servicevertrag führen.

4. `bench_compute_subcommand`
   - repo surface: `runtime/ucf-bench/src/main.rs::run_compute`
   - execution: `build_backend(cfg) -> backend.compute(...)`
   - status/evidence: Benchmarkmetriken
   - status pattern: `internal_dev_test_only`
   - evidence pattern: `internal_dev_test_only`
   - alignment: `internal_dev_test_only`
   - caveat: internes Harness, kein normaler Domänenanschluss.

5. `domains_ai_compat_lane`
   - repo surface: `domains/ai* + domains/ai-backends compatibility crates`
   - execution/status/evidence: Legacy Host-ABI Adapter-Lane
   - status pattern: `mixed_legacy_consumption_pattern`
   - evidence pattern: `mixed_legacy_consumption_pattern`
   - alignment: `legacy_compat_path`
   - caveat: explizite Compatibility-Grenze außerhalb outward-facing Compute-Verträge.

## Deliberate limits

- Keine neue Integrationsplattform.
- Keine neue Consumer-spezifische Vertragswelt.
- Keine Produkt-API-/Auth-/Billing-/Governance-Ausweitung.
- Canonical outward-facing Verträge bleiben: Execution + Status/Diagnostics + Evidence/References.
- Consumer-seitige Caveat-Semantik bleibt auf derselben Top-Level-Sprache:
  `current/trusted`, `caveated|partial`, `degraded|unavailable`,
  `insufficient evidence references`.
