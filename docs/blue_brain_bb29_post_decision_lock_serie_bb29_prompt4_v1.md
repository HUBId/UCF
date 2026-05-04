# Serie BB29 Prompt 4: Entscheidung nach BB29 festziehen

Status: **Roadmap-Entscheidung nach der stabilisierten Drei-Regionen-Basis ist explizit fixiert**.  
Diese Notiz führt **keine** neue Funktionsarbeit ein und startet **keine** neue Serie.

## 1) Repo-basierte Lageprüfung nach BB29

Die bestehende Referenzfläche zeigt konsistent:

- Region 1, Region 2 und Region 3 sind kontrolliert geöffnet und maintenance-hardened.
- Bounded Relationen 1↔2, 1↔3 und 2↔3 sind vorhanden und begrenzt.
- Guard-/Contract-/Diagnostics-Semantik ist auf der Drei-Regionen-Basis bereits final konsolidiert.
- no-direct-\* und BB23-Freeze-/Maintenance-Grenzen bleiben unverändert verpflichtend.

Damit ist der operative Zustand nach BB29 technisch stabil genug, um als Default im Maintenance-Modus weitergeführt zu werden.

## 2) Region-4-Hebelprüfung (ehrlich, nicht künstlich)

Aktuell ist **kein klar lokalisierbarer technischer Hebel** im Repo erkennbar, der zwingend eine Region-4-Öffnung verlangt.

- Es liegt kein nachweisbar ungelöstes Problem vor, das innerhalb der maintenance-hardened Drei-Regionen-Basis nicht bearbeitbar wäre.
- Ein Region-4-Schritt wäre derzeit primär ein Re-Scope-Wunsch und kein aus der Referenzfläche erzwungener Integrationsbedarf.

## 3) Entscheidung (genau eine)

**Entscheidung:** Nach BB29 ist `Maintenance/Bugfix/Cleanup ohne neue Serie` der explizite Default.

Abgrenzung:

- Kein automatischer Übergang in neue Serienlogik.
- Kein impliziter Start einer Region-4-Linie.
- Ein späterer Region-4-Re-Scope bleibt nur als **separater, expliziter, technisch begründeter** Entscheidungsakt möglich.

## 4) Trennlinie Maintenance vs. späteres Re-Scope

### Maintenance jetzt (zulässig)

- Bugfixes, Hardening, Cleanup, Doku-/Referenzkonsistenz.
- Guard-/Contract-Konsistenzpflege innerhalb der Drei-Regionen-Grenzen.

### Nicht Teil von Maintenance jetzt (nicht automatisch zulässig)

- neue Regionenklasse (Region 4),
- Vollsimulation oder Plattform-Ausweitung,
- Planner-/Agentenlogik, Policy-Governance-Ausweitung, Retry-Orchestrierung,
- neue Compute-Core-Serienlogik.

## 5) Referenzanker

- `docs/blue_brain_three_region_maintenance_stabilization_line_serie_bb29_prompt1_v1.md`
- `docs/blue_brain_three_region_docs_tests_index_cleanup_serie_bb29_prompt2_v1.md`
- `docs/blue_brain_bb29_final_three_region_stabilization_sweep_serie_bb29_prompt3_v1.md`
- `docs/roadmap/REPO_MAP.md`
