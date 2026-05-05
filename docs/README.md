# UCF Operational Documentation Index (Blue-Brain Authority Chain)

Dieses README ist die **kanonische Einstiegsfläche** für operative Doku-Pfade im aktuellen Repo-Stand.

## 0) Authority chain (historical vs current)

Kanonische Status-Map:
- `docs/blue_brain_authority_chain_status_map.md`

- **Current operational authority (maßgeblich):**
  - `docs/blue_brain_bb29_post_maintenance_default_decision_map_serie_bb29_prompt5_v1.md`
  - `docs/blue_brain_bb29_final_maintenance_handoff_map_serie_bb29_prompt6_v1.md`
- **Historical snapshots (nicht aktuelle operative Autorität):**
  - `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
  - `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
  - `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
  - `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`

Regel: Bei Konflikten zwischen historischen BB25/BB27-Aussagen und der BB29-Endlage gilt **immer** die BB29-Current-Authority-Linie.

Autoritätsklassen im Repo-Kontext:
- current operational authority
- historical snapshot
- supporting reference
- non-canonical / outdated pointer


## 1) Canonical operational entrypoints

- **Operational readiness map (BB20 P1):** `docs/blue_brain_bb20_production_readiness_sweep_serie_bb20_prompt1_v1.md`
- **Terminology/state semantics harmonization (BB20 P2):** `docs/blue_brain_bb20_cross_line_terminology_state_harmonization_serie_bb20_prompt2_v1.md`
- **BB22 cross-line stabilization/cleanup:**
  - `docs/blue_brain_bb22_narrow_cross_line_stabilization_pass_serie_bb22_prompt1_v1.md`
  - `docs/blue_brain_bb22_cross_line_guard_signal_consistency_serie_bb22_prompt2_v1.md`
  - `docs/blue_brain_bb22_remaining_internal_only_transition_doc_cleanup_serie_bb22_prompt3_v1.md`
  - `docs/blue_brain_bb22_readiness_sweep_final_cross_line_stabilization_serie_bb22_prompt4_v1.md`
- **Final BB20 readiness sweep + next-priority lock (BB20 P4):** `docs/blue_brain_bb20_final_readiness_sweep_next_priority_lock_serie_bb20_prompt4_v1.md`
- **BB23 freeze/maintenance baseline:** `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- **BB23 maintenance guard rails / allowed-change envelope:** `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- **BB23 final freeze/readiness statement (maintenance transition):** `docs/blue_brain_bb23_final_freeze_readiness_statement_serie_bb23_prompt3_v1.md`
- **Region-1 maintenance reference surface (BB25):**
  - _Authority class: historical snapshot line (nicht aktuelle operative Autorität)._
  - `docs/blue_brain_first_region_finalization_serie_bb24_prompt10_v1.md`
  - `docs/blue_brain_first_region_stabilization_serie_bb25_prompt1_v1.md`
  - `docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md`
  - `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
  - `docs/blue_brain_post_bb25_roadmap_decision_serie_bb25_prompt4_v1.md`
  - `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
  - `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
  - `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
  - `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`
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
- **Region-2 tests/guards cleanup line (BB26 P6):**
  - `docs/blue_brain_second_region_tests_guards_cleanup_serie_bb26_prompt6_v1.md`
- **Two-region guard/contract consistency line (BB26 P7):**
  - `docs/blue_brain_two_region_guard_contract_consistency_serie_bb26_prompt7_v1.md`
- **BB26 readiness sweep / second-region expansion boundary (BB26 P8):**
  - `docs/blue_brain_bb26_readiness_sweep_second_region_expansion_boundary_serie_bb26_prompt8_v1.md`
- **Two-region maintenance stabilization/reference line (BB27):
  - _Authority class: supporting reference (historical transition line)._
  - `docs/blue_brain_two_region_maintenance_stabilization_serie_bb27_prompt1_v1.md`
  - `docs/blue_brain_two_region_docs_tests_reference_cleanup_serie_bb27_prompt2_v1.md`
  - `docs/blue_brain_bb27_final_two_region_stabilization_sweep_serie_bb27_prompt3_v1.md`
  - `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
  - `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`
- **Third-region selection line (BB28):
  - `docs/blue_brain_third_region_selection_serie_bb28_prompt1_v1.md`
  - `docs/blue_brain_third_region_integration_serie_bb28_prompt2_v1.md`
  - `docs/blue_brain_third_region_runtime_selection_reference_contract_serie_bb28_prompt3_v1.md`
  - `docs/blue_brain_third_region_relation_line_serie_bb28_prompt4_v1.md`
  - `docs/blue_brain_third_region_diagnostics_caveat_deferred_semantics_serie_bb28_prompt5_v1.md`
  - `docs/blue_brain_third_region_tests_guards_cleanup_serie_bb28_prompt6_v1.md`
  - `docs/blue_brain_three_region_guard_contract_consistency_serie_bb28_prompt7_v1.md`
  - `docs/blue_brain_bb28_readiness_sweep_third_region_expansion_boundary_serie_bb28_prompt8_v1.md`
- **Anatomical region decision/selection line (BB30):**
  - `docs/blue_brain_anatomical_region_decision_line_serie_bb30_prompt1_v1.md`
  - `docs/blue_brain_first_anatomical_region_selection_serie_bb30_prompt2_v1.md`
  - `docs/blue_brain_first_anatomical_region_integration_serie_bb30_prompt3_v1.md`
  - `docs/blue_brain_first_anatomical_region_model_decision_serie_bb30_prompt4_v1.md`
  - `docs/blue_brain_first_anatomical_region_diagnostics_contract_semantics_serie_bb30_prompt5_v1.md`
  - `docs/blue_brain_bb30_readiness_sweep_first_anatomical_expansion_boundary_serie_bb30_prompt6_v1.md`
- **First anatomical maintenance stabilization/reference line (BB31):**
  - `docs/blue_brain_first_anatomical_stabilization_line_serie_bb31_prompt1_v1.md`
  - `docs/blue_brain_first_anatomical_docs_tests_index_cleanup_serie_bb31_prompt2_v1.md`
  - `docs/blue_brain_bb31_final_first_anatomical_stabilization_sweep_serie_bb31_prompt3_v1.md`
- **Three-region maintenance stabilization/reference line (BB29):
  - `docs/blue_brain_three_region_maintenance_stabilization_line_serie_bb29_prompt1_v1.md`
  - `docs/blue_brain_three_region_docs_tests_index_cleanup_serie_bb29_prompt2_v1.md`
  - `docs/blue_brain_bb29_final_three_region_stabilization_sweep_serie_bb29_prompt3_v1.md`
  - `docs/blue_brain_bb29_post_decision_lock_serie_bb29_prompt4_v1.md`
  - `docs/blue_brain_bb29_post_maintenance_default_decision_map_serie_bb29_prompt5_v1.md`
- **BlueBrain/Runtime/Selection hardening line (BB19):**
  - `docs/blue_brain_runtime_selection_contract_hardening_serie_bb19_prompt1_v1.md`
  - `docs/blue_brain_runtime_selection_diagnostics_hardening_serie_bb19_prompt2_v1.md`
- **Execution guard rails (BB18):** `docs/blue_brain_execution_guard_rails_production_facing_serie_bb18_prompt3_v1.md`
- **Real compute final reference line:**
  - `docs/final_reference_line_serie_j_v1.md`
  - `docs/final_production_readiness_evidence_pack_serie_j_v1.md`

## 2) Status classes and where to verify them

- **frozen stable baseline:** BB19/BB21/BB22, verbindlich über BB23 Prompt 1–3.
- **maintenance-only stable:** BB3/BB8/BB13/BB14/BB17/BB18 ohne Capability-Ausweitung.
- **advisory-only / bounded:** BB10/BB11/BB16 bleiben advisory-only.
- **usable-with-caveats (frozen semantics):** BB6/BB7/BB9/BB15 candidate/caveat slices ohne Promotion.
- **deferred / test-only / non-canonical:** nicht Teil der operativen Baseline, keine implizite Reaktivierung.

Referenzpunkte für die technische Scope-Grenze:
- `docs/supported_scope_execution_v13.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- `docs/blue_brain_bb23_final_freeze_readiness_statement_serie_bb23_prompt3_v1.md`

## 2.1) Post-BB29 maintenance default (three-region bounded handoff)

- Aktive Regionenbasis ist **Region 1 + Region 2 + Region 3 (bounded advisory/reference lane)**.
- Standardmodus bleibt **Maintenance/Bugfix/Cleanup** mit klarer no-direct-* Guard-Linie.
- **Region 4 ist nicht aktiv** und benötigt einen späteren expliziten Re-Scope.
- Die BB28/BB29-Linie schließt als kontrollierte, maintenance-gehärtete Drei-Regionen-Basis ohne Plattformausweitung.
- Canonical stabilization map: `docs/blue_brain_three_region_maintenance_stabilization_line_serie_bb29_prompt1_v1.md`
- Canonical post-decision map: `docs/blue_brain_bb29_post_maintenance_default_decision_map_serie_bb29_prompt5_v1.md`


## 2.2) Audit-Baseline (Blue-Brain / Drei-Regionen, BB29) — 2026-05-04

Für einen reproduzierbaren Audit-Baseline-Pass wurden die kanonischen AGENTS-Checks frisch ausgeführt; die Reports liegen unter:

- `out/blue_brain_audit_baseline_2026-05-04/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-04/gate_report.json`

Ausgeführt wurden:

- `cargo test --workspace`
- `cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-04/docs_lint_report.json`
- `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-04/gate_report.json`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`

Diese Baseline dokumentiert den wartungsstabilen Stand ohne Scope-Erweiterung (keine Region-4-/Plattform-/Planner-Neulogik).

## 3) Guard rails and boundaries (must stay visible)

- Keine implizite Scope-Erweiterung außerhalb der unterstützten Scope-Artefakte.
- `no-direct-*` Grenzen (kein direkter Action/Compute/Retry/Policy/Memory-Commit Pfad aus advisory-only Linien).
- Non-canonical/internal-only Pfade haben **keine** direkte operative Autorität.

Primäre Guard-/Boundary-Dokumente:
- `docs/blue_brain_execution_guard_rails_production_facing_serie_bb18_prompt3_v1.md`
- `docs/blue_brain_bb16_readiness_sweep_bounded_dynamics_execution_line_serie_bb16_prompt4_v1.md`
- `docs/supported_scope_execution_v13.md`

## 4) Series closure docs (operationally relevant)

Nur operativ relevante Abschlusslinien (BB8–BB19) als Referenzfläche:
- `docs/blue_brain_memory_diagnostics_runtime_feedback_serie_bb8_prompt3_v1.md`
- `docs/blue_brain_kuramoto_minimal_modulation_path_serie_bb10_prompt2_v1.md`
- `docs/blue_brain_kuramoto_input_groups_parametrization_hardening_serie_bb12_prompt3_v1.md`
- `docs/blue_brain_combined_retrieval_diagnostics_stale_invalidated_failed_feedback_serie_bb15_prompt2_v1.md`
- `docs/blue_brain_bb16_readiness_sweep_bounded_dynamics_execution_line_serie_bb16_prompt4_v1.md`
- `docs/blue_brain_bb18_readiness_sweep_production_hardening_closure_serie_bb18_prompt4_v1.md`
- `docs/blue_brain_runtime_selection_contract_hardening_serie_bb19_prompt1_v1.md`
- `docs/blue_brain_runtime_selection_diagnostics_hardening_serie_bb19_prompt2_v1.md`

## 5) Deprecated index intent (cleanup)

Die frühere `Chip-2` Architektur-/Modul-Indexstruktur in dieser Datei war für den aktuellen operativen UCF-Stand nicht mehr maßgeblich und wurde als primärer Einstieg entfernt.

Wenn einzelne ältere Dateien weiterhin gebraucht werden, müssen sie gegen die oben genannten kanonischen operativen Referenzpfade validiert werden.

- **Hippocampus-first role consolidation (BR1):**
  - `docs/blue_brain_hippocampus_region_role_map_serie_br1_prompt1_v1.md`
  - `docs/blue_brain_hippocampus_minimal_bounded_integration_serie_br1_prompt2_v1.md`
  - `docs/blue_brain_hippocampus_surface_diagnostics_contracts_hardening_serie_br1_prompt3_v1.md`
  - `docs/blue_brain_br1_hippocampus_readiness_sweep_expansion_boundary_serie_br1_prompt4_v1.md`

- **Amygdala-next role consolidation (BR2):**
  - `docs/blue_brain_amygdala_region_role_map_serie_br2_prompt1_v1.md`
  - `docs/blue_brain_amygdala_surface_diagnostics_contracts_hardening_serie_br2_prompt3_v1.md`
