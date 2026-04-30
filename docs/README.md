# UCF Operational Documentation Index (BB22 Cleanup)

Dieses README ist die **kanonische Einstiegsfläche** für operative Doku-Pfade im aktuellen Repo-Stand.

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
  - `docs/blue_brain_first_region_finalization_serie_bb24_prompt10_v1.md`
  - `docs/blue_brain_first_region_stabilization_serie_bb25_prompt1_v1.md`
  - `docs/blue_brain_region1_maintenance_reference_surface_serie_bb25_prompt2_v1.md`
  - `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
  - `docs/blue_brain_post_bb25_roadmap_decision_serie_bb25_prompt4_v1.md`
  - `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
  - `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
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
