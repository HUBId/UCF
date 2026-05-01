# Repo Map (Operational Modules)

Zweck: schneller, repo-treuer Einstieg in die **aktuell operativen** Linien ohne zweite Wahrheitsquelle.

## Canonical operational map

- **BlueBrain operational sweep map (BB20 P1):** `docs/blue_brain_bb20_production_readiness_sweep_serie_bb20_prompt1_v1.md`
- **BB23 freeze/maintenance baseline:** `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- **BB23 allowed-change envelope:** `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- **Region-1 maintenance reference surface (BB25):**
  - `docs/blue_brain_first_region_finalization_serie_bb24_prompt10_v1.md`
  - `docs/blue_brain_first_region_stabilization_serie_bb25_prompt1_v1.md`
  - `docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md`
  - `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
  - `docs/blue_brain_post_bb25_roadmap_decision_serie_bb25_prompt4_v1.md`
  - `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
  - `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
- **Region-2 selection line (BB26 P1):**
  - `docs/blue_brain_second_region_selection_serie_bb26_prompt1_v1.md`
- **Region-2 integration line (BB26 P2):**
  - `docs/blue_brain_second_region_integration_serie_bb26_prompt2_v1.md`
- **Region-2 runtime/selection/reference contract line (BB26 P3):**
  - `docs/blue_brain_second_region_runtime_selection_reference_contract_serie_bb26_prompt3_v1.md`
- **Region-2 first bounded inter-region relation line (BB26 P4):**
  - `docs/blue_brain_first_inter_region_relation_line_serie_bb26_prompt4_v1.md`
- **Region-2 diagnostics/caveat/deferred semantics line (BB26 P5):**
  - `docs/blue_brain_second_region_diagnostics_caveat_deferred_semantics_serie_bb26_prompt5_v1.md`
- **Two-region maintenance stabilization/reference line (BB27):**
  - `docs/blue_brain_two_region_maintenance_stabilization_serie_bb27_prompt1_v1.md`
  - `docs/blue_brain_two_region_docs_tests_reference_cleanup_serie_bb27_prompt2_v1.md`
  - `docs/blue_brain_bb27_final_two_region_stabilization_sweep_serie_bb27_prompt3_v1.md`
- **Cross-line state semantics (BB20 P2):** `docs/blue_brain_bb20_cross_line_terminology_state_harmonization_serie_bb20_prompt2_v1.md`
- **Final readiness sweep + next-priority lock (BB20 P4):** `docs/blue_brain_bb20_final_readiness_sweep_next_priority_lock_serie_bb20_prompt4_v1.md`
- **Docs operational index:** `docs/README.md`

## Runtime / compute core

- **Canonical compute contracts + invariants:**
  - `runtime/ucf-compute/src/contracts.rs`
  - `runtime/ucf-compute/src/reference_map.rs`
- **Final compute reference line:** `docs/final_reference_line_serie_j_v1.md`

## Policy / scope authority

- `policies/packs/base_v1/`
- `policies/packs/overlays/{test,dev,prod}/`
- `policies/manifest.toml`
- `docs/supported_scope_execution_v13.md`

## Boundary markers

Maintenance-Interpretation folgt der BB23 Allowed-Change-Map: maintenance-safe, hardening-safe, doc/reference cleanup sind zulässig; Reaktivierung deferred/non-canonical oder Capability-Ausweitung benötigt expliziten Re-Scope.

- `advisory-only` / `bounded` Linien bleiben ohne direkte Ausführungsautorität.
- `candidate-only`, `test-only`, `deferred`, `non-canonical` markieren nicht-operative oder nicht-hochgestufte Pfade.
- `no-direct-*` Guard Rails bleiben verpflichtend.
- Im aktuellen post-BB27-Betrieb sind Region 1 und Region 2 die einzigen aktiven Regionenexpansionen; die Zwei-Regionen-Basis bleibt bounded/advisory/reference-constrained, und eine dritte Regionenklasse bleibt explizit re-scope-pflichtig (nicht automatisch aktiv).

Primäre Referenzen:
- `docs/blue_brain_execution_guard_rails_production_facing_serie_bb18_prompt3_v1.md`
- `docs/blue_brain_bb16_readiness_sweep_bounded_dynamics_execution_line_serie_bb16_prompt4_v1.md`
- `docs/blue_brain_runtime_selection_contract_hardening_serie_bb19_prompt1_v1.md`
