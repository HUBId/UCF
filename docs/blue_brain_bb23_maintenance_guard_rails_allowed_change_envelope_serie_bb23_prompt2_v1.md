# BlueBrain Serie BB23 — Prompt 2: Maintenance Guard Rails / Allowed-Change Envelope (v1)

Status: BB23 präzisiert die Freeze-/Maintenance-Baseline aus Prompt 1 in einen **kanonischen, technischen Allowed-Change-Rahmen**. Ziel ist klare Trennung zwischen maintenance-safe Änderungen und Scope-Ausweitung.

## 1) Kanonische Allowed-Change-Map

| Klasse | Bewertung | Technische Bedeutung im BB23-Freeze-Modus |
|---|---|---|
| `allowed maintenance change` | **zulässig** | Deterministische, semantik-erhaltende Pflege: kleine Refactors, typo-/naming-fixes, Kommentar-/Dokuklarstellung, ohne neue Capability/Autorität. |
| `allowed bugfix/hardening change` | **zulässig** | Fail-closed-, no-direct-*-, Guard- oder Integritäts-Härtung bestehender Pfade; behebt Defekt ohne neue operative Linie. |
| `allowed doc/reference cleanup` | **zulässig** | Readiness-/Abschluss-/Referenz-Doku angleichen, solange Statusklassen (`frozen`, `maintenance-only`, `advisory-only`, `candidate-only`, `non-canonical`) nicht aufgeweicht werden. |
| `change requiring explicit re-scope` | **nicht maintenance-default** | Änderung erzeugt neue Autorität, neue Übergänge oder neue operative Interpretation; nur mit expliziter neuer Scope-Entscheidung außerhalb BB23. |
| `deferred/non-canonical reactivation attempt` | **nicht zulässig in maintenance** | Reaktivierung/Promotion von deferred/test-only/non-canonical Pfaden in operative/canonical Pfade ohne neue Serie. |
| `out-of-scope expansion` | **unzulässig** | Plattform- oder Capability-Ausbau jenseits des Freeze-Rahmens; klar außerhalb BB23-Maintenance. |

## 2) Positivliste: Was bleibt als Maintenance erlaubt

Erlaubt sind ausschließlich schmale Änderungen, die bestehende Semantik erhalten:
- deterministische Bugfixes an bereits kanonischen Linien,
- Guard-/Assertion-/Fail-Closed-Härtung bestehender no-direct-* Grenzen,
- Testergänzungen zur Drift-Erkennung (ohne neue Capability-Freischaltung),
- Doku- und Referenzkonsolidierung auf bestehende BB19/BB21/BB22/BB23-Klassifikation,
- kleine Cleanup-/Lesbarkeitsänderungen ohne Runtime-/Selection-/Execution-Autoritätsgewinn.

## 3) Negativliste: Änderungen mit Re-Scope-Pflicht

Die folgenden Klassen sind **nicht** maintenance-only und benötigen expliziten Re-Scope:
- Erweiterung `allowed-actions` oder neue ausführbare Action-/Tool-Klassen,
- neue Planner-/Agentenlogik oder neue Entscheidungs-/Policy-Autorität,
- Retry-/Queue-/Orchestration-Plattformaufbau,
- Retrieval-/Consolidation-/Reasoning-Plattformausbau,
- Neurodynamik-Ausweitung über bounded advisory-only hinaus,
- Compute-Core-Neuentwicklung jenseits maintenance-only,
- neue automatische Memory-Commit-/Persistenz-Autorität,
- jedes Upgrading von `candidate-only`, `advisory-only`, `reference-only` zu execution-authoritative Verhalten.

## 4) Deferred / Test-only / Non-canonical Reaktivierungsgrenze

Für BB23 gilt fail-closed:
- `deferred` bleibt deferred,
- `test-only` bleibt test-only,
- `non-canonical/internal-only` bleibt non-canonical/internal-only,
- Reaktivierung in kanonische operative Pfade ist **kein** Maintenance-Pass,
- Reaktivierung erfordert explizite neue Serien-/Scope-Entscheidung außerhalb BB23.

## 5) Guard-Rail-Bindung (single source of technical truth)

Die bestehenden Guard Rails bleiben unverändert maßgeblich:
- `no-direct-*` und `no-auto-*` Grenzen,
- `advisory-only` bleibt advisory-only,
- `candidate-only` bleibt candidate-only,
- `reference-only`/weak signal bleibt begrenzte Referenzlinie,
- canonical vs non-canonical Trennung,
- terminal-state separation (`completed/failed/cancelled/blocked/unavailable/unsupported/non-canonical`).

Jede Änderung, die diese Bindungen aufweicht oder eine zweite Guard-Wirklichkeit einführt, ist nicht maintenance-only.

## 6) Wartungsmodus-Checks (targeted)

Für BB23-maintenance Pässe auf betroffenen Flächen mindestens:
1. Format/Lint/Compile-Hygiene (`cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`).
2. Relevante Tests für berührte Guard-/Scope-Flächen.
3. Docs-/Readiness-Konsistenz:
   - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
   - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`

Die Vollmatrix bleibt nur dann erforderlich, wenn breit wirksame Schnittstellen/Autoritätsketten verändert werden.
