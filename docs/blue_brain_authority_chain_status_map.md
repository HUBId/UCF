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

## Canonical classification (BB25 vs BB27)

### Current operational authority (maßgeblich)

- `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`
- `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`

Operative Endlage: **Region 1 und Region 2 aktiv**, Maintenance/Bugfix/Cleanup als Default, Region 3 inaktiv bis explizitem Re-Scope.

### Historical snapshot (nicht aktuelle operative Autorität)

- `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`
- `docs/blue_brain_final_maintenance_handoff_serie_bb25_prompt6_v1.md`

Historische Aussage bleibt erhalten (BB25-Zeitpunkt), aber wird durch BB27-Endlage operativ übersteuert.

### Supporting reference (nicht konkurrenzierende Endlage)

- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- `docs/blue_brain_bb23_final_freeze_readiness_statement_serie_bb23_prompt3_v1.md`
- `docs/blue_brain_bb26_readiness_sweep_second_region_expansion_boundary_serie_bb26_prompt8_v1.md`
- `docs/blue_brain_bb27_final_two_region_stabilization_sweep_serie_bb27_prompt3_v1.md`

### Non-canonical / outdated pointer

- Jede Lesart, die BB25-Dokumente als heute gleichrangige operative Autorität behandelt.
- Implizite Fortsetzungs-/Expansionspointer ohne expliziten Re-Scope.

## Conflict rule (single truth)

Wenn historische BB25-Aussagen und BB27-Endlage unterschiedlich sind, gilt **ausschließlich** die BB27-Current-Authority-Linie als operativ verbindlich.
