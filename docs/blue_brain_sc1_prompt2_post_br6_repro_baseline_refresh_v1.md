# Serie SC1 Prompt 2: Post-BR6 repro/test baseline refresh consolidation

> Maintenance-discoverability note: This file is supporting current evidence for the 2026-05-08 audit baseline. It is not a second authority source; read it with `docs/blue_brain_audit_baseline_map_v1.md` and `docs/blue_brain_authority_chain_status_map.md`.

Status: gezielte Umsetzung der höchsten priorisierten Konsolidierungsmaßnahme aus `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`. Diese Datei ist eine **Consolidation-Action-Evidence-Line**; sie baut keine neue Region, keine neue Relation, keinen neuen Modellvertiefungskandidaten, keine Planner-/Agentenlogik, keine Policy-Governance und keine Compute-Core-Arbeit.

## 1) Präzise Verortung der priorisierten Maßnahme

SC1 Prompt 1 priorisierte als Maßnahme mit höchstem Hebel den **Repro-/Test-Baseline-Refresh for post-BR6**. Die Kernursache war nicht ein Codepfad, sondern eine Evidenzlücke: die versionierten Blue-Brain-Baselines unter `out/blue_brain_audit_baseline_2026-05-02/` und `out/blue_brain_audit_baseline_2026-05-04/` waren historische BB29-/pre-BR6-Belege, während die aktuelle Authority-Linie sechs integrierte anatomische Regionen, IR1, MD2, MD3 und SC1-Maintenance-Closure umfasst.

Direkt betroffene Knoten:

- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md` als Audit- und Priorisierungsursprung,
- `docs/README.md` als operative Doku-Einstiegsfläche,
- `docs/blue_brain_authority_chain_status_map.md` als Authority-Klassifikation,
- `out/blue_brain_audit_baseline_2026-05-08/` als frische post-BR6 Evidenzfläche,
- `out/docs_lint_report.json` und `out/gate_report.json` als kanonische Root-Reports.

Folgeeffekte und kleinster sinnvoller Hebel:

- Regionen: Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus bleiben bounded; der Refresh ändert keinen Regionsstatus.
- Relationen: IR1 bleibt Relation Map, nicht Orchestrator; der Refresh ändert keine Relationklasse.
- Modellgrenzen: MD2 bleibt genau die erste maintenance-gehärtete Modellvertiefung (`Amygdala ↔ Thalamus`); MD3 bleibt genau die zweite bounded Modellvertiefung (`Amygdala ↔ Basal Ganglia`); der Refresh öffnet keinen weiteren Kandidaten.
- Guard Rails: no-direct-* Grenzen werden nur verifiziert und dokumentarisch nachgeschärft, nicht erweitert.
- Kleinste Hebeländerung: frische Baseline-Artefakte plus ein eindeutiger Konsolidierungsverweis in README/Authority/Audit, damit historische Baselines nicht mehr als aktuelle Evidenz missverstanden werden.

## 2) Canonical consolidation action map

| Consolidation state | Canonical target in this action | Allowed reading | Forbidden reading |
| --- | --- | --- | --- |
| `primary consolidation target` | post-BR6 Repro-/Test-Baseline-Refresh | Frische Evidenz, dass Docs-Lint und Readiness-Gate auf der sechs Regionen + IR1 + MD2 + MD3 + SC1 Authority-Linie reproduzierbar laufen. | Neue Runtime-, Selection-, Region-, Relation-, Model- oder Compute-Funktion. |
| `supporting affected surface` | `docs/README.md`, `docs/blue_brain_authority_chain_status_map.md`, SC1-Audit-Doku | Einstiegspunkte und Authority-Klassifikation zeigen die 2026-05-08-Baseline als aktuelle Evidenzfläche. | Zweite operative Wahrheit neben BR6/IR1/MD2/MD3/SC1. |
| `guard-sensitive area` | no direct action/execution/retry/memory/compute/safety override; no implicit region/model expansion | Guards bleiben unverändert bounded und werden im Action-Doc sichtbar gehalten. | Direct trigger, direct commit, direct invocation, safety override oder Planner-/Agenten-/Policy-Autorität. |
| `doc/test evidence area` | `out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`, `out/blue_brain_audit_baseline_2026-05-08/gate_report.json`, Root-Reports | Reproduzierbare Reports für die aktuelle Doku-/Readiness-Linie. | Behavior-Beweis für neue Features oder externe Betriebsfreigabe. |
| `non-canonical residual path` | ältere Baselines 2026-05-02 und 2026-05-04 | Historische Vergleichs- und Audit-Spur. | Aktuelle post-BR6 Endlage oder konkurrierende Authority. |

Diese Action Map ist absichtlich klein: sie klassifiziert nur den Konsolidierungszustand dieser Maßnahme und ersetzt weder die Regionendokumente noch IR1/MD2/MD3/SC1.

## 3) Umgesetzte Konsolidierungsmaßnahme

Die priorisierte Maßnahme wurde als post-BR6 Baseline-Refresh umgesetzt:

- neue versionierte Baseline-Fläche: `out/blue_brain_audit_baseline_2026-05-08/`,
- frischer Docs-Lint-Report für die aktuelle Doku-Oberfläche,
- frischer Readiness-Gate-Report für das `test`-Profil,
- README-Korrektur von historischer BB29-Baseline zu aktueller post-BR6-Konsolidierungsevidenz,
- Authority-Map-Einordnung dieser Datei als supporting current reference, nicht als konkurrierende Current Authority,
- SC1-Audit-Restpunkt von offener Repro-Lücke auf erledigte erste Konsolidierungsmaßnahme reduziert.

## 4) Regions-, Relations- und Modellgrenzen

Diese Maßnahme ist rein evidenz- und dokumentationsbezogen:

- Keine neue anatomische Region wird eingeführt.
- Keine bestehende Region erhält neue Runtime-/Selection-/Reference-Autorität.
- Keine IR1-Relation wird aktiviert, promoted oder in Orchestrierung umgedeutet.
- Keine `current model mode`-Lesart wird operativ verstärkt.
- MD2 bleibt genau die erste maintenance-gehärtete Modellvertiefung (`Amygdala ↔ Thalamus`); MD3 bleibt genau die zweite bounded Modellvertiefung (`Amygdala ↔ Basal Ganglia`). Beide bleiben diagnostisch/contract-bounded.
- Real Compute Stack und Compute-Core bleiben maintenance-only und werden nicht geändert.

## 5) No-direct-* und Scope-Grenzen

Für diese Action gilt explizit:

- kein direct action trigger,
- kein direct execution trigger,
- kein direct retry trigger,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override,
- keine implizite Policy-/Governance-Autorität,
- keine implizite neue Region,
- kein impliziter neuer Modellvertiefungskandidat,
- keine globale Neurodynamik- oder Modellplattform.

Docs- und Report-Artefakte dürfen nur als Evidence/Readiness-Spur gelesen werden. Sie erzeugen keine operative Autorität für Verbraucher außerhalb der bestehenden BR6/IR1/MD2/SC1-Linie.

## 6) Doku-/Test-/Code-Kohärenz

Die Kohärenz wird über genau eine Evidenzlinie hergestellt:

- `docs/README.md` zeigt die frische Baseline als aktuelle post-BR6/IR1/MD2/MD3/SC1 Konsolidierungsevidenz.
- `docs/blue_brain_authority_chain_status_map.md` klassifiziert diese Datei als supporting current reference, damit keine zweite operative Wahrheit entsteht.
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md` verweist auf diese Umsetzung als erledigte Maßnahme 1.
- `out/blue_brain_audit_baseline_2026-05-08/` enthält die versionierte Report-Fläche.
- Root-Reports unter `out/docs_lint_report.json` und `out/gate_report.json` bleiben die kanonischen aktuellen Check-Ausgaben.

Es wurden keine Code-, Schema-, Policy- oder Fixture-Semantiken geändert.

## 7) Non-canonical Residuen

Die älteren Baselines bleiben erhalten, werden aber herabgestuft:

- `out/blue_brain_audit_baseline_2026-05-02/`: historische pre-/early-Blue-Brain Auditspur.
- `out/blue_brain_audit_baseline_2026-05-04/`: historische BB29-/Drei-Regionen-Baseline.

Beide dürfen weiterhin zur Reproduktion historischer Zustände und zum Vergleich verwendet werden, aber nicht als aktuelle Evidenz für die post-BR6 sechs-Regionen-Endlage.

## 8) Gezielte Checks

Für diese Maßnahme sind die maßgeblichen Checks:

```bash
cargo test --workspace
cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-08/gate_report.json
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Die Reports belegen nur Doku-/Readiness-/Build-Kohärenz; sie sind keine Feature-Freigabe.

## 9) Abschlussnotiz

Geänderte Dateien/Flächen in diesem Pass:

- `docs/blue_brain_sc1_prompt2_post_br6_repro_baseline_refresh_v1.md`
- `docs/README.md`
- `docs/blue_brain_authority_chain_status_map.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`
- `out/blue_brain_audit_baseline_2026-05-08/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-08/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-08/gate_report.json`
- `out/docs_lint_report.json`
- `out/gate_report.json`

Umgesetzte Maßnahme: die wichtigste SC1-Restschwäche, eine fehlende frische post-BR6/IR1/MD2/MD3 Repro-/Test-Baseline, ist geschlossen oder deutlich reduziert. Historische BB29-Baselines bleiben als Residuen erhalten, sind aber nicht mehr die sichtbare aktuelle Evidenzfläche.

Nach SC1 Prompt 3 verbleibende sinnvolle Konsolidierungsmaßnahme:

1. Relation cleanup/hardening review für selection-mediated und execution-interface-mediated IR1-Wortlaut, insbesondere Basal Ganglia/Cerebellum/Hypothalamus-Kanten.

Hinweis: Die zuvor hier genannte Cross-line terminology/guard checklist consolidation wurde durch `docs/blue_brain_sc1_prompt3_cross_line_terminology_guard_checklist_consolidation_v1.md` umgesetzt.
