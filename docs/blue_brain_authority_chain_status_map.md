# Blue-Brain Authority Chain Status Map (Canonical)

Zweck: Diese Datei definiert die **kanonische historical-vs-current-authority map** für Blue-Brain-Dokumente im aktuellen Repo-Stand.  
Sie ist bewusst klein gehalten und ersetzt keine Fachdokumente, sondern ordnet deren Autoritätsstatus.

## Authority classes

Nur diese Klassen sind für die operative Einordnung zu verwenden:

1. **current operational authority**
   - Maßgebliche operative Endlage; bei Konflikten immer vorrangig.
2. **historical snapshot**
   - Historischer Entscheidungs-/Handoff-Stand; dokumentarisch wichtig, aber nicht operativ vorrangig.
3. **supporting current reference**
   - Stützende aktuelle Referenz für Guard Rails, Audit, Scope, Discoverability oder Nachvollziehbarkeit; keine eigenständige konkurrierende Endlage.
4. **stale discoverability pointer**
   - Älterer oder verkürzter Verweis, der bei isolierter Lektüre wie aktuelle Autorität wirken kann, aber über diese Map relativiert werden muss.
5. **non-canonical/internal-only shadow surface**
   - DBM-/Microcircuit-/Neuro-/adjacent-domain Oberfläche außerhalb der aktuellen operativen Blue-Brain-Autorität; keine implizite Region, Relation, Modellplattform oder Consumer-Autorität.

Die kompakte maintenance-facing Discoverability-Map ist `docs/blue_brain_maintenance_discoverability_map_v1.md`; das Shadow-Surface-Inventar ist `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md`; die kanonische sechs-Regionen-Inventar-/Rollenkarte ist `docs/blue_brain_canonical_region_inventory_map_v1.md`; die Guard-/Semantic-Drift-Map ist `docs/blue_brain_guard_semantic_drift_map_v1.md`; die Discoverability-Findings-Map ist `docs/blue_brain_discoverability_findings_map_v1.md`; die aktuelle Maintenance-Action-Map ist `docs/blue_brain_current_maintenance_action_map_v1.md`; die Discoverability-Cleanup-Abschlussnotiz ist `docs/blue_brain_discoverability_cleanup_pass_v1.md`. Diese Dateien sind Supporting References, nicht zweite Wahrheitsquellen.

## Canonical classification (post-BR6/IR1/MD2/MD3/SC1 current line)

### Current operational authority (maßgeblich)

- `docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md`
- `docs/blue_brain_ir1_readiness_sweep_inter_region_closure_serie_ir1_prompt4_v1.md`
- `docs/blue_brain_md2_model_deepening_docs_tests_reference_cleanup_v1.md`
- `docs/blue_brain_md3_second_deepening_rescope_line_v1.md`
- `docs/blue_brain_md3_second_model_deepening_implementation_line_v1.md`
- `docs/blue_brain_md3_second_model_deepening_hardening_line_v1.md`
- `docs/blue_brain_md3_readiness_sweep_system_closure_v1.md`
- `docs/blue_brain_post_md3_maintenance_decision_pass_v1.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`
- `docs/blue_brain_sc1_prompt4_final_system_consolidation_sweep_v1.md`

Operative Endlage: **Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus** gelten als bounded, advisory/reference/diagnostic integrierte anatomische Regionen. IR1 bleibt die führende bounded inter-region architecture; die kompakte Relation-Closure-Referenz ist `docs/blue_brain_canonical_inter_region_relation_map_v1.md`; die kompakte systemweite Modellgrenzen-Referenz ist `docs/blue_brain_canonical_model_boundary_map_v1.md`; MD1/MD2 bleibt genau eine maintenance-gehärtete erste Modellvertiefung (`Amygdala ↔ Thalamus`). MD3 Prompt 1 öffnet auf Re-Scope-Ebene genau einen zweiten Kandidaten (`Amygdala ↔ Basal Ganglia`); MD3 Prompt 2 implementiert genau diesen Kandidaten minimal als relation-level bounded Kuramoto-like advisory/diagnostic line; MD3 Prompt 3 härtet die Grenzen; MD3 Prompt 4 schließt den Stand als maintenance-ready. Der Post-MD3-Maintenance-/Decision-Pass hält den Befund fest: keine aktive spätere Re-Scope-Option bleibt repo-basiert offen. Plattformbildung oder direkte Action-/Execution-/Retry-/Memory-/Compute-/Safety-Autorität bleiben ausgeschlossen. Der finale Default bleibt Maintenance/Bugfix/Cleanup/Report-Refresh; weiterer Regionsausbau, dritte Modellvertiefung und Plattformbildung sind nicht aktiv.

### Historical snapshot (nicht aktuelle operative Autorität)

- `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
- `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
- `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
- `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`
- `docs/blue_brain_bb29_post_maintenance_default_decision_map_serie_bb29_prompt5_v1.md`
- `docs/blue_brain_bb29_final_maintenance_handoff_map_serie_bb29_prompt6_v1.md`

Historische Aussagen bleiben erhalten (BB25/BB27/BB29-Zeitpunkte), werden aber durch die BR1-BR6/IR1/MD2-Endlage operativ übersteuert, sofern sie weniger als sechs integrierte anatomische Regionen oder ältere Expansion-Locks beschreiben.

### Supporting current reference (nicht konkurrenzierende Endlage)

- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- `docs/blue_brain_bb23_final_freeze_readiness_statement_serie_bb23_prompt3_v1.md`
- `docs/blue_brain_bb28_readiness_sweep_third_region_expansion_boundary_serie_bb28_prompt8_v1.md`
- `docs/blue_brain_bb29_final_three_region_stabilization_sweep_serie_bb29_prompt3_v1.md`
- `docs/blue_brain_bb29_post_decision_lock_serie_bb29_prompt4_v1.md`
- `docs/blue_brain_anatomical_region_canonical_map_serie_bb32_prompt1_v1.md`
- `docs/blue_brain_br1_hippocampus_readiness_sweep_expansion_boundary_serie_br1_prompt4_v1.md`
- `docs/blue_brain_br2_amygdala_readiness_sweep_expansion_boundary_serie_br2_prompt4_v1.md`
- `docs/blue_brain_br3_thalamus_readiness_sweep_expansion_boundary_serie_br3_prompt4_v1.md`
- `docs/blue_brain_br4_basal_ganglia_readiness_sweep_expansion_boundary_serie_br4_prompt4_v1.md`
- `docs/blue_brain_br5_cerebellum_readiness_sweep_expansion_boundary_serie_br5_prompt4_v1.md`
- `docs/blue_brain_md1_readiness_sweep_model_deepening_closure_v1.md`
- `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md`
- `docs/blue_brain_sc1_prompt3_cross_line_terminology_guard_checklist_consolidation_v1.md`
- `docs/blue_brain_audit_baseline_map_v1.md`
- `docs/blue_brain_maintenance_discoverability_map_v1.md`
- `docs/blue_brain_discoverability_findings_map_v1.md`
- `docs/blue_brain_discoverability_cleanup_pass_v1.md`
- `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md`
- `docs/blue_brain_canonical_region_inventory_map_v1.md`
- `docs/blue_brain_canonical_inter_region_relation_map_v1.md`
- `docs/blue_brain_canonical_model_boundary_map_v1.md`
- `docs/blue_brain_maintenance_consolidation_pass_v1.md`
- `docs/blue_brain_guard_semantic_drift_map_v1.md`
- `docs/blue_brain_maintenance_consolidation_refresh_2026_05_10.md`
- `docs/blue_brain_current_maintenance_action_map_v1.md`
- `docs/blue_brain_structural_closure_map_v1.md`
- `docs/blue_brain_hh_readiness_decision_map_v1.md`
- `docs/blue_brain_first_hh_candidate_map_v1.md`
- `docs/blue_brain_hh_prerequisite_map_v1.md`
- `docs/blue_brain_hh_readiness_closure_map_v1.md`

### Historical/supporting implementation-stage references (durch spätere Authority relativiert)

- `docs/blue_brain_inter_region_architecture_serie_ir1_prompt1_v1.md`
- `docs/blue_brain_first_inter_region_implementation_serie_ir1_prompt2_v1.md`
- `docs/blue_brain_inter_region_diagnostics_contracts_serie_ir1_prompt3_v1.md`
- `docs/blue_brain_first_inter_region_relation_line_serie_bb26_prompt4_v1.md`
- `docs/blue_brain_two_region_guard_contract_consistency_serie_bb26_prompt7_v1.md`

Diese Dateien bleiben wichtig für Implementierungs- und Guard-Trail-Nachvollziehbarkeit. Wenn sie engere Zwischenstände, two-region Sprache oder frühe relation activation wording enthalten, gilt für aktuelle operative Aussagen immer die spätere BR6/IR1-Prompt-4/MD2/MD3/SC1-Linie.

### Stale discoverability pointer / non-canonical shadow surface

- Jede Lesart, die BB25-, BB27- oder BB29-Dokumente als heute gleichrangige operative Autorität behandelt.
- Jede Lesart, die Prefrontal Cortex, Anterior Cingulate Cortex, Insula oder zusätzliche DBM-/Microcircuit-/Neuro-Shadow-Crates als aktuell operativ integrierte Regionen behandelt.
- Implizite Fortsetzungs-/Expansionspointer ohne expliziten Re-Scope.
- Jede Modellvertiefungslesart, die aus MD1/MD2/MD3 eine globale Modellplattform, weitere Kandidaten oder direkte Autorität ableitet; MD3 priorisiert, implementiert, härtet und schließt ausschließlich `Amygdala ↔ Basal Ganglia` als bounded Kuramoto-like second deepening; der Post-MD3-Pass lässt `BLUE_BRAIN_POST_MD3_POSSIBLE_FUTURE_RE_SCOPE_CANDIDATE` leer.
- Jede Lesart verteilter Terminologie, die `advisory-only`, `caveated`, `deferred`, `blocked`, `insufficient`, `diagnostic-only`, `reference-only`, `current model mode` oder `non-canonical/internal-only` mit direkter Action-/Execution-/Retry-/Memory-/Compute-/Safety-Autorität verwechselt.
- Jede Lesart, die die Präsenz von `crates/dbm_*`, `crates/microcircuit_*`, `crates/biophys_*` oder angrenzenden Brain/DigitalBrain/Neuromod/SNN/FEP-Domains als operative Blue-Brain-Autorität auslegt; maßgeblich ist stattdessen `docs/blue_brain_non_canonical_shadow_surface_inventory_v1.md`.

## Conflict rule (single truth)

Wenn historische BB25/BB27/BB29-Aussagen und die post-BR6/MD3-Endlage unterschiedlich sind, gilt **ausschließlich** die BR6/IR1/MD2/MD3/System-Audit/SC1-Prompt-4-Current-Authority-Linie als operativ verbindlich. MD3 Prompt 4 ist die finale MD3-Readiness-Map nach der zweiten Vertiefung; SC1 Prompt 4 bleibt die systemweite Maintenance-Entscheidung; die Structural-Closure-Map v1 bündelt diesen Stand ohne neue Autorität und erklärt nur, dass ein separater HH-Readiness-Block vertretbar ist, nicht HH-Implementierung. Die HH-Readiness-Decision-Map v1 konkretisiert nur Voraussetzungen, Grenzen, Nicht-Ziele und spätere Re-Scope-Kriterien; die First-HH-Candidate-Map v1 isoliert genau `Basal Ganglia ↔ Cerebellum` als ersten späteren HH-Kandidaten auf Kandidatenebene. Beide sind keine HH-Implementierung. Keine der Dateien ist eine neue Funktions- oder Plattformquelle.
