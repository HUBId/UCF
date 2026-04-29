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
- **BlueBrain/Runtime/Selection hardening line (BB19):**
  - `docs/blue_brain_runtime_selection_contract_hardening_serie_bb19_prompt1_v1.md`
  - `docs/blue_brain_runtime_selection_diagnostics_hardening_serie_bb19_prompt2_v1.md`
- **Execution guard rails (BB18):** `docs/blue_brain_execution_guard_rails_production_facing_serie_bb18_prompt3_v1.md`
- **Real compute final reference line:**
  - `docs/final_reference_line_serie_j_v1.md`
  - `docs/final_production_readiness_evidence_pack_serie_j_v1.md`

## 2) Status classes and where to verify them

- **stable / production-usable:** anhand der BB18/BB19 Linien + Serie-J Final-Referenzlinie.
- **advisory-only / bounded:** primär in BB16/BB19 Vertrags- und Diagnostics-Dokus.
- **candidate-only / test-only / deferred / non-canonical:** explizit in den jeweiligen Serien-Dokus markiert; kein direkter Produktionsanspruch.

Referenzpunkt für die technische Scope-Grenze:
- `docs/supported_scope_execution_v13.md`

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
