# Serie SC1 Prompt 3: Cross-line terminology and guard checklist consolidation

> Maintenance-discoverability note: This file is the supporting current terminology/guard checklist. It constrains readings of current and historical docs, but does not create new operational authority or expansion scope.

Status: gezielte Umsetzung der zweitpriorisierten Konsolidierungsmaßnahme aus `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`. Diese Datei ist eine **Second-Consolidation-Action-Evidence-Line**; sie baut keine neue Region, keine neue Relation, keinen neuen Modellvertiefungskandidaten, keine Planner-/Agentenlogik, keine Policy-Governance und keine Compute-Core-Arbeit.

## 1) Präzise Verortung der zweitpriorisierten Maßnahme

SC1 Prompt 1 priorisierte nach dem Repro-/Test-Baseline-Refresh die **Cross-line terminology/guard checklist consolidation**. Die Kernursache war kein fehlender Runtime-Pfad, sondern verteilte Semantik: `advisory-only`, `caveated`, `deferred`, `blocked`, `insufficient`, `diagnostic-only`, `reference-only`, `current model mode` und `non-canonical/internal-only` waren in vielen Regionen-, Relations-, Modell- und Guard-Dokumenten konsistent, aber nicht als kompakte Consumer-Checkliste verdichtet.

Direkt betroffene Flächen:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` als konzentrierte Codefläche für Region-, Relation-, Modellvertiefungs- und Guard-Zustände.
- `runtime/ucf-compute/src/lib.rs` als re-exportierte öffentliche Referenzfläche für die kompakte Checklist-Semantik.
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md` als Audit- und Priorisierungsursprung.
- `docs/blue_brain_authority_chain_status_map.md` und `docs/README.md` als Einstiegspunkte, damit die Checkliste nicht neben der Current-Authority-Linie verborgen bleibt.

Kleinste sinnvolle Änderung mit hohem Hebel: eine kleine kanonische Action Map plus eine kompakte, testgestützte Guard-Checkliste für die vorhandenen Begriffe. Dadurch wird kein bestehender Consumer umgebaut; bestehende bounded/no-direct-Grenzen werden nur expliziter prüfbar.

## 2) Kanonische second consolidation action map

| Action-map state | Betroffene Fläche | Erlaubte Lesart | Explizit verboten |
| --- | --- | --- | --- |
| `secondary consolidation target` | cross-line Terminologie und Guard-Checkliste | Verdichtung bestehender Begriffsbedeutungen für Maintenance-Leser und Tests. | Neue Feature-Semantik, neue Region, neue Relation, neue Modellplattform. |
| `supporting affected surface` | Region-/Relation-/MD2-/Guard-Dokumente und `blue_brain_region_first_integration.rs` | Stützende Konsolidierung der bereits existierenden Semantik. | Zweite operative Wahrheit neben BR6/IR1/MD2/MD3/SC1. |
| `guard-sensitive area` | no-direct action/execution/retry/memory/compute/safety override Grenzen | Explizite Verbotsmatrix für Consumer-Reads. | Direct trigger, direct commit, direct invocation, safety override oder Planner-/Agenten-/Policy-Autorität. |
| `doc/test evidence area` | diese Datei, README, Authority-Map und Unit-Test in `ucf-compute` | Nachweis, dass alle Begriffe eine bounded Consumer-Lesart und eine no-direct-Verbotslinie haben. | Behavior-Beweis für neue Funktionen oder externe Betriebsfreigabe. |
| `non-canonical residual path` | ältere Formulierungen in historischen BB-/BR-/IR-/MD-Dokumenten | Historische Spur, sofern durch die Current-Authority-Linie und diese Checkliste eingeordnet. | Aktuelle Endlage oder konkurrierende Consumer-Semantik. |

## 3) Konsolidierte Terminologie-/Guard-Checkliste

| Begriff | Erlaubter Consumer-Read | Verbotene Autorität |
| --- | --- | --- |
| `advisory-only` | bounded positive read only | kein direct action trigger, direct execution trigger, direct retry trigger, direct memory commit, direct compute invocation, safety override, direct selection authority oder Promotion zu starker Autorität |
| `caveated` | bounded read with visible caveat only | keine Promotion zu strong support; keine direct action/execution/retry/memory/compute/safety authority |
| `deferred` | not-active-yet status read only | keine stille Aktivierung, keine Retry-Orchestrierung, keine direct action/execution/memory/compute/safety authority |
| `blocked` | fail-closed unavailable/forbidden-path read only | keine Fallback-Aktivierung, kein Override, keine direct action/execution/retry/memory/compute/safety authority |
| `insufficient` | weak-evidence diagnostic read only | kein Support-Signal, keine Promotion, keine direct action/execution/retry/memory/compute/safety authority |
| `diagnostic-only` | observable diagnostic state read only | keine advisory promotion, keine direct action/execution/retry/memory/compute/selection/safety authority |
| `reference-only` | read-only context/reference access only | keine Mutation, kein direct memory commit, keine direct action/execution/retry/compute/safety authority |
| `current model mode` | descriptive model-mode read only | keine Contract-Autorität, keine Modellplattform, kein weiterer Vertiefungskandidat, keine direct action/execution/retry/memory/compute/safety authority |
| `non-canonical/internal-only` | internal/test/residual traceability read only when explicitly caveated | keine consumer-operative Semantik, keine direct action/execution/retry/memory/compute/safety-, Region-, Relation- oder Modellautorität |

Die gleiche Minimalmatrix ist in `CANONICAL_BLUE_BRAIN_CROSS_LINE_TERMINOLOGY_GUARD_CHECKLIST` abgebildet. Sie ist eine Konsolidierungs- und Testreferenz, kein neuer Runtime-Entscheider.

## 4) Regions-, Relations- und Modellgrenzen gegen Nebenwirkungen

Diese Maßnahme ändert keine aktuelle Region-Surface:

- Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus bleiben bounded, advisory/reference/diagnostic integrierte Regionen.
- IR1-Relationen bleiben bounded reads; selection-mediated und execution-interface-mediated wording wird durch die Checkliste nicht zu Orchestrierung.
- MD2 bleibt genau die maintenance-gehärtete erste `Amygdala ↔ Thalamus`-Vertiefung; MD3 bleibt genau die bounded zweite `Amygdala ↔ Basal Ganglia`-Vertiefung; `current model mode` bleibt deskriptiv.
- Real Compute Stack, Runtime, Policy und Compute-Core bleiben maintenance-only und unverändert in ihrer operativen Semantik.

## 5) No-direct-* und Scope-Grenzen

Für alle Checklist-Begriffe gilt gemeinsam:

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

## 6) Doku-/Test-/Code-Kohärenz

Die Kohärenz wird über genau eine kompakte Referenz hergestellt:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` enthält die Action Map und Checklist-Struktur.
- Der Unit-Test `sc1_second_consolidation_action_and_guard_checklist_are_canonical` prüft Vollständigkeit, read-only-Lesart und no-direct-Verbote für alle Checklist-Begriffe.
- `runtime/ucf-compute/src/lib.rs` re-exportiert die Checklist-Fläche für bestehende Referenznutzer.
- `docs/README.md` und `docs/blue_brain_authority_chain_status_map.md` verweisen auf diese Datei als supporting reference, nicht als konkurrierende Current Authority.
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md` wird von offener zweitpriorisierter Maßnahme auf umgesetzt/gestützt aktualisiert.
- `docs/blue_brain_maintenance_verification_findings_map_v1.md` ergänzt die Check-/Evidence-Lesereihenfolge als supporting verification reference; sie ersetzt keine Current Authority.

## 7) Non-canonical Residuen

Ältere Dokumente dürfen weiterhin einzelne Begriffe lokal erklären. Sobald sie von der kompakten Checkliste abweichen oder mehr Autorität suggerieren, gilt diese SC1-Prompt-3-Checkliste als stützende Maintenance-Einordnung unterhalb der BR6/IR1/MD2/MD3/System-Audit/SC1-Current-Authority-Linie. Dadurch bleibt keine zweite operative Wirklichkeit für Consumer-Reads stehen.

## 8) Gezielte Checks

Für diese Maßnahme sind die maßgeblichen Checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
```

Die Checks belegen Doku-/Code-/Test-Kohärenz und Guard-Stabilität; sie sind keine Feature-Freigabe.

## 9) Abschlussnotiz

Geänderte Dateien/Flächen in diesem Pass:

- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
- `runtime/ucf-compute/src/lib.rs`
- `docs/blue_brain_sc1_prompt3_cross_line_terminology_guard_checklist_consolidation_v1.md`
- `docs/README.md`
- `docs/blue_brain_authority_chain_status_map.md`
- `docs/blue_brain_system_audit_consolidation_serie_sc1_prompt1_v1.md`

Umgesetzte Maßnahme: Die zweitwichtigste SC1-Restschwäche, verteilte Cross-line-Terminologie mit Guard-Drift-Risiko, ist durch eine kleine kanonische Action Map, eine kompakte Guard-Checkliste und einen Unit-Test deutlich reduziert.

Verbleibende sinnvolle Konsolidierungsmaßnahme: Relation cleanup/hardening review für selection-mediated und execution-interface-mediated IR1-Wortlaut, insbesondere Basal Ganglia/Cerebellum/Hypothalamus-Kanten.
