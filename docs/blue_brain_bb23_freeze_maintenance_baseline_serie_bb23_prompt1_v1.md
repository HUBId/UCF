# BlueBrain Serie BB23 — Prompt 1: Freeze-/Maintenance-Baseline (v1)

Status: **finalisierte operative BlueBrain-Linien sind eingefroren**. BB23 etabliert einen expliziten Erhaltungsrahmen ohne neue Funktionsserie.

## 1) Kanonische Freeze-/Maintenance-Map

| Linie | Statusklasse | BB23-Einordnung |
|---|---|---|
| Runtime/Selection Contract (BB19), Execution/Reference Interaction (BB21), Cross-line Stabilization (BB22) | **frozen stable baseline** | Operative Kernlinie; nur schmale Maintenance-Änderungen zulässig. |
| Context/Memory/Reference (BB3/BB8/BB17), Minimal Execution + Integrity/Terminal Boundaries (BB13/BB14/BB18) | **maintenance-only stable line** | Stabil nutzbar, Semantik eingefroren; keine Capability-Ausweitung. |
| Bounded dynamics/minimale Delta-Linie (BB10/BB11/BB16) | **advisory-only frozen line** | Explizit advisory-only; keine operative Aufwertung und keine Dynamics-Ausweitung. |
| Candidate/reasoning/planning/placeholder-nahe Linien (BB6/BB7/BB9/BB15 candidate slices) | **usable with caveats but frozen semantics** | Nur im bestehenden engen Scope nutzbar; keine Promotion zur neuen Plattform. |
| Internal/expert/dev/test/compat/deferred Pfade | **deferred/non-canonical/not part of baseline** | Bleiben nicht-operativ, nicht-promotable ohne neue explizite Serie außerhalb BB23. |

## 2) Maintenance-only Boundary (zulässig vs. out-of-scope)

Zulässige Maintenance-Pässe:
- deterministische Bugfixes ohne Semantik-Ausweitung,
- enge Guard-/Fail-Closed-Härtung bestehender Pfade,
- Terminologie-/Doku-/Readiness-Map-Konsistenz,
- testseitige Drift-Korrekturen ohne neue Capability.

Out-of-scope (für BB23 und Folge-Maintenance-Pässe unverändert geblockt):
- neue allowed-actions-Erweiterungen,
- neue Planner-/Agentenlogik oder Plattformbildung,
- Retry-/Queue-/Orchestration-Plattform,
- Retrieval-/Consolidation-/Reasoning-Plattformausbau,
- Neurodynamik-Ausweitung,
- neue Compute-Core-Arbeit jenseits maintenance-only,
- implizite Reaktivierung von deferred/non-canonical Pfaden.

## 3) Guard Rails im Freeze-Modus

Die bestehenden Guard-Linien bleiben bindend und unverändert:
- no-direct-* / no-auto-* / fail-closed Exclusion-Linien,
- canonical vs non-canonical Trennung,
- terminal state separation (`completed/failed/cancelled/blocked/unavailable/unsupported/non-canonical`),
- bounded advisory-only dynamics,
- maintenance-only Compute-Exit-Boundary.

BB23 ergänzt keine neue Governance-Plattform, sondern fixiert die Auslegung: fehlende Evidenz oder Scope-Unklarheit wird als **nicht maintenance-only** behandelt.

## 4) Drift- und Scope-Aufweichungsschutz

Für Änderungen nach BB23 gilt:
1. Jede Änderung muss einer bestehenden stabilen Linie zuordenbar sein.
2. Jede Änderung muss die bestehende Statusklasse erhalten (frozen/maintenance/advisory/non-canonical).
3. Jede Änderung, die neue Autorität/Capability erzeugt, ist kein Maintenance-Pass.
4. Deferred/non-canonical/test-only Pfade dürfen nicht stillschweigend wie operative Pfade dokumentiert oder getestet werden.

## 5) Maintenance-nahe Prüfspur (targeted)

Für schmale Maintenance-Pässe bleibt mindestens erforderlich:
1. `cargo test --workspace`
2. `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
3. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
4. targeted crate/module checks für direkt berührte Flächen (`cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`).

Die volle Matrix bleibt nur dann Pflicht, wenn tatsächlich breit wirksame Schnittstellen oder Autoritätsketten geändert werden.

## 6) Serienentscheidung nach BB23

Repo-basierte Entscheidung nach BB23:
- **Keine weitere Feature-Serie als Default.**
- Standardmodus ist **Maintenance/Bugfix/Cleanup ohne neue Serienstruktur**.
- Eine neue Serie ist nur dann technisch ehrlich, wenn eine explizite neue Scope-Entscheidung außerhalb dieser Freeze-Baseline getroffen und separat belegt wird.
