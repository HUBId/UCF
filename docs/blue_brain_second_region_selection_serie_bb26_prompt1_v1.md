# Serie BB26 Prompt 1: Second-Region-Selection Line (genau eine zweite Regionenklasse)

Status: **kanonische Second-Region-Selection-Linie für BB26 Prompt 1**.

Diese Entscheidung öffnet **keine** zweite Region direkt in Runtime. Sie priorisiert exakt eine nächste reale Regionenklasse für eine spätere, minimal kontrollierte Expansion und hält Region 1 als einzige aktive Expansion.

## 1) Kandidatenprüfung: Hebel, Nähe, Risiko, Scope

| Regionenklasse | Funktionaler UCF-Hebel (jetzt) | Integrationsnähe zu vorhandenen Linien | Cross-line-/Autoritätsrisiko | Modellschwere | Test-/Doku-Tragfähigkeit | Scope-Ausweitungsgefahr | BB26-Einordnung |
|---|---|---|---|---|---|---|---|
| Memory/Context-related | hoch: bessere Referenzqualität, Caveat-/Invalidation-Semantik, stabilere runtime feedback loops | sehr hoch (BB8 + BB17 + BB21) | niedrig-mittel (beherrschbar via read/reference-only guard rails) | niedrig-mittel (abstract-first) | hoch (bestehende map-/contract-Testflächen) | mittel (nur bei impliziter Persistenzautorität) | **Second-expansion candidate** |
| Action-readiness/Execution-interface-related | mittel-hoch: bessere Eligibility-Interpretation | hoch (BB19 + BB21) | mittel-hoch (Nähe zu Execution/Retry-Autorität) | niedrig | mittel-hoch | mittel-hoch | **Viable but not second** |
| Bounded dynamics modulation-related | mittel: advisory modulation | mittel-hoch (BB12) | mittel (Fehlkopplung kann Scope drücken) | mittel | mittel | mittel | **Later-phase candidate** |
| Simulation-only/Diagnostic-only (HH-nah) | niedrig operativ, diagnostisch relevant | mittel (BB10 diagnostics) | niedrig operativ, aber Integrationsdruck hoch | hoch | mittel | niedrig operativ, hoher Aufwand | **Simulation-only/deferred candidate** |
| Deferred/Not-suitable-now Klassen | aktuell niedrig | niedrig | hoch bei vorzeitiger Öffnung | variabel | niedrig-mittel | hoch | **Simulation-only/deferred candidate** |
| non_canonical_internal_only_region_path | keine kanonische operative Rolle | n/a | hoch bei Promotion | n/a | n/a | hoch | **Non-canonical/internal-only path** |

## 2) Kanonische second-region selection map

Verbindliche Zustände für BB26-P1:

1. `second_expansion_candidate`
2. `viable_but_not_second`
3. `later_phase_candidate`
4. `simulation_only_deferred_candidate`
5. `non_canonical_internal_only_path`

## 3) Explizite Auswahlkriterien (technisch)

- **Funktionaler UCF-Hebel:** verbessert direkt Reference-/Context-Qualität in operativen Entscheidungen.
- **Integrationsnähe:** nutzt bestehende Runtime/Selection/Context/Reference-Verträge statt neuer Core-Pfade.
- **Bounded advisory-only Kompatibilität:** keine direkte Ausführungsautorität, nur bounded Hinweise/Diagnostik.
- **Geringe Autoritätsgefahr:** keine Action-/Retry-/Memory-Commit-/Compute-Autorität.
- **Überschaubare Tiefe:** abstract-first anschließbar; dynamics-first bleibt optional später.
- **Komplement zu Region 1:** Region 1 bleibt attention/selection-zentriert; Region 2 ergänzt reference/context Robustheit statt Dublette.

## 4) Priorisierte zweite Regionenklasse

**Gewählte zweite reale Regionenklasse:** `Memory/Context-related`.

Warum jetzt Region 2:

- maximaler Hebel auf Referenzvalidität und Caveat-geregelte Konsistenz in bereits produktionsnahen BB8/BB17/BB21 Linien,
- stärkste Wiederverwendung bestehender hardened surfaces ohne neue Compute-/Planner-/Governance-Welten,
- erlaubt eine schmale, kontrollierte Erweiterung als **abstract-first** bevor dynamics-lastige Vertiefung sinnvoll wäre.

Warum schwerere/HH-nahe Klassen nicht zuerst:

- HH-nahe Klassen bleiben zu modellschwer und operativ nicht hebelstärkster nächster Schritt,
- dynamics-first vor context/reference Ergänzung würde Integrationsrisiko bei geringerem unmittelbaren UCF-Nutzen erhöhen.

## 5) Einordnung der übrigen Kandidaten

- `Action-readiness/Execution-interface-related` → **viable but not second**.
- `Bounded dynamics modulation-related` → **later-phase candidate**.
- `Simulation-only/Diagnostic-only` + `Deferred/Not-suitable-now` → **simulation-only/deferred candidate**.
- `non_canonical_internal_only_region_path` → **non-canonical/internal-only path**.

"Nicht jetzt" ist eine Sequenzierungsentscheidung, keine Verwerfung.

## 6) Guard-/Scope-/Safety-Absicherung

Für die zweite Region bleiben unverändert hart:

- keine direkte Action-/Retry-/Memory-/Compute-Autorität,
- keine Safety-Override-Semantik,
- keine implizite Reaktivierung deferred/non-canonical Pfade,
- keine Öffnung einer dritten Regionenklasse im selben Schritt.

## 7) Minimale Einhängungsrichtung (nächster Schritt, ohne Vollimplementierung)

Geplante minimale Andockpunkte für Region 2:

- **Anschluss an BB8/BB17/BB21:**
  - Inputs: canonical reference validity, context evidence priority, caveat/invalidated diagnostics.
  - Outputs: advisory-only context/reference weighting hints + contract diagnostics.
- **Explizit nicht berühren:**
  - execution write/retry paths,
  - memory persistence/commit authority,
  - compute-core execution semantics,
  - policy/governance/planner/orchestration lanes.

Komplementarität zu Region 1:

- Region 1: attention/selection-focused advisory lane.
- Region 2: context/reference quality and caveat lane.
- Ergebnis: ergänzend statt redundant.
