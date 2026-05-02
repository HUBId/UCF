# Serie BB25 Prompt 5: Post-BB25 Maintenance-Default Decision Map

> ⚠️ **Authority notice (historical snapshot):** Dieses Dokument beschreibt den BB25-Stand als historischen Entscheidungs-Snapshot.  
> **Nicht** als aktuelle operative Autorität lesen.  
> Maßgebliche aktuelle Endlage:  
> - `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md`  
> - `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`  
> Kanonische Klassifikation: `docs/blue_brain_authority_chain_status_map.md`

Status: **historischer Snapshot (BB25-Stand), nicht aktuelle operative Autorität**. Dokumentiert den damaligen Post-BB25-Entscheidungsstand mit Region 1 aktiv und Region 2 nicht aktiv. Aktuelle operative Autorität liegt bei den BB27-Abschlussdokumenten.

Diese Datei konsolidiert ausschließlich die technische Default-Entscheidung nach BB25. Sie führt **keine** neue Serie, **keine** neue Regionenimplementierung und **keine** neue Plattformarbeit ein.

## 1) Repo-basierte Abschlusslage (BB25)

- Region 1 ist als erste Regionenklasse stabilisiert und maintenance-hardened abgeschlossen (BB24 Prompt 10 + BB25 Prompt 1–3).
- Die BB23-Freeze-/Maintenance-Baseline bleibt bindend als Allowed-Change-Rahmen.
- Die Post-BB25-Entscheidung aus Prompt 4 bleibt gültig: kein automatischer Region-2-Start, kein automatischer Serienfortsatz.

## 2) Canonical post-BB25 decision map

| Decision state | Status | Wirkung |
| --- | --- | --- |
| `maintenance_default_active` | **active** | Standardarbeit: Bugfix/Cleanup/Maintenance innerhalb bestehender Guard Rails und ohne Capability-Ausweitung. |
| `region1_active_stabilized_baseline` | **active** | Region 1 ist die einzige aktive Regionenexpansion und dient als referenzierte, stabilisierte Baseline. |
| `region2_not_active_explicit_rescope_required` | **not active** | Region 2 ist nicht verworfen, aber aktuell geschlossen; Öffnung nur über spätere explizite Re-Scope-/Priorisierungsentscheidung. |
| `deferred_non_canonical_out_of_scope_continuation` | **not allowed in default lane** | Reaktivierung deferred/non-canonical Pfade oder Scope-/Plattformausbau bleibt außerhalb des Maintenance-Defaults. |

## 3) Maintenance-default (verbindlich)

Nach BB25 gilt technisch verbindlich:
- Default-Modus ist **Maintenance/Bugfix/Cleanup**,
- es folgt **keine** automatische neue Regionenserie,
- es gibt **keine** implizite funktionale Öffnung zusätzlicher Regionenklassen.

Maintenance-default bleibt an den BB23 Allowed-Change-Envelope gebunden und darf dessen Scope-Grenzen nicht aufweichen.

## 4) Region-2 Re-Scope (bewusst offen, nicht aktiv)

Region 2 bleibt explizit in folgendem Zustand:
- **nicht aktiv**, **nicht automatisch als nächster Schritt**,
- **nicht verworfen**,
- nur bei späterer, expliziter Re-Scope-Entscheidung mit klarer technischer Begründung.

Damit wird Roadmap-Flexibilität erhalten, ohne den aktuellen Maintenance-Default zu verlassen.

## 5) Unveränderte Guard-/Out-of-scope-Grenzen

Unverändert außerhalb des aktiven Post-BB25-Defaults:
- Mehrfachregionen-Expansion,
- direkte Hodgkin-Huxley-Produktivintegration,
- neue `allowed-actions`-Erweiterung,
- Planner-/Agentenplattform,
- Policy-/Governance-Plattform,
- Retry-/Queue-/Orchestration-Plattform,
- Retrieval-/Consolidation-/Reasoning-Plattform,
- neue Compute-Core-Arbeit jenseits maintenance-only.

## 6) Canonical references (single-truth alignment)

Diese Decision-Map ist auf folgende bestehende Referenzen gebunden:
- `docs/blue_brain_bb23_freeze_maintenance_baseline_serie_bb23_prompt1_v1.md`
- `docs/blue_brain_bb23_maintenance_guard_rails_allowed_change_envelope_serie_bb23_prompt2_v1.md`
- `docs/blue_brain_first_region_finalization_serie_bb24_prompt10_v1.md`
- `docs/blue_brain_region1_final_stabilization_sweep_serie_bb25_prompt3_v1.md`
- `docs/blue_brain_post_bb25_roadmap_decision_serie_bb25_prompt4_v1.md`

Sie ergänzt diese Linien als Post-BB25-Default-Explizitheit und erzeugt keine zweite operative Wahrheit.


## 7) Authority classification (historical vs current)

- **Dokumenttyp:** historical snapshot (BB25-Zeitpunkt).
- **Verbindlichkeit heute:** nur historisch erklärend; **nicht** maßgebliche operative Endlage.
- **Current authority:** `docs/blue_brain_bb27_post_maintenance_default_decision_map_serie_bb27_prompt5_v1.md` und `docs/blue_brain_bb27_final_maintenance_handoff_map_serie_bb27_prompt6_v1.md`.
