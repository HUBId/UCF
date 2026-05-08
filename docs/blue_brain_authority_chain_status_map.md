# Blue-Brain Authority Chain Status Map (Canonical)

Zweck: Diese Datei definiert die **kanonische historical-vs-current-authority map** für Blue-Brain-Dokumente im aktuellen Repo-Stand.  
Sie ist bewusst klein gehalten und ersetzt keine Fachdokumente, sondern ordnet deren Autoritätsstatus.

## Authority classes

Nur diese Klassen sind für die operative Einordnung zu verwenden:

1. **current operational authority**
   - Maßgebliche operative Endlage; bei Konflikten immer vorrangig.
2. **historical snapshot**
   - Historischer Entscheidungs-/Handoff-Stand; dokumentarisch wichtig, aber nicht operativ vorrangig.
3. **supporting reference**
   - Stützende Referenz für Guard Rails, Scope oder Nachvollziehbarkeit; keine eigenständige konkurrierende Endlage.
4. **non-canonical / outdated pointer**
   - Verweis, der nicht als aktuelle Autorität gelesen werden darf (z. B. veraltete oder historisch überholte Pointer).

## Canonical classification (post-BR6 current line)

### Current operational authority (maßgeblich)

- `docs/blue_brain_br6_hypothalamus_readiness_sweep_expansion_boundary_serie_br6_prompt4_v1.md`
- `docs/blue_brain_ir1_readiness_sweep_inter_region_closure_serie_ir1_prompt4_v1.md`
- `docs/blue_brain_md2_model_deepening_docs_tests_reference_cleanup_v1.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`

Operative Endlage: **Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus** gelten als bounded, advisory/reference/diagnostic integrierte anatomische Regionen. IR1 bleibt die führende bounded inter-region architecture; MD1/MD2 bleibt genau eine maintenance-gehärtete Modellvertiefung (`Amygdala ↔ Thalamus`) und öffnet keinen zweiten Kandidaten. Der nächste Default ist Konsolidierung/Maintenance, nicht weiterer Regionsausbau.

### Historical snapshot (nicht aktuelle operative Autorität)

- `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
- `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`
- `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
- `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`
- `docs/blue_brain_bb29_post_maintenance_default_decision_map_serie_bb29_prompt5_v1.md`
- `docs/blue_brain_bb29_final_maintenance_handoff_map_serie_bb29_prompt6_v1.md`

Historische Aussagen bleiben erhalten (BB25/BB27/BB29-Zeitpunkte), werden aber durch die BR1-BR6/IR1/MD2-Endlage operativ übersteuert, sofern sie weniger als sechs integrierte anatomische Regionen oder ältere Expansion-Locks beschreiben.

### Supporting reference (nicht konkurrenzierende Endlage)

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

### Non-canonical / outdated pointer

- Jede Lesart, die BB25-, BB27- oder BB29-Dokumente als heute gleichrangige operative Autorität behandelt.
- Jede Lesart, die Prefrontal Cortex, Anterior Cingulate Cortex oder Insula als aktuell operativ integrierte Regionen behandelt.
- Implizite Fortsetzungs-/Expansionspointer ohne expliziten Re-Scope.
- Jede Modellvertiefungslesart, die aus MD1/MD2 eine globale Modellplattform oder einen zweiten aktiven Vertiefungskandidaten ableitet.

## Conflict rule (single truth)

Wenn historische BB25/BB27/BB29-Aussagen und die post-BR6-Endlage unterschiedlich sind, gilt **ausschließlich** die BR6/IR1/MD2/System-Audit-Current-Authority-Linie als operativ verbindlich.
