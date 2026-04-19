# Serie N: Broader UCF System Integration Map v1 (repo-basiert, schmal, hart priorisiert)

Status: repo-basierte, technisch harte Priorisierung breiterer UCF-Systemflächen gegen die **finale Compute-Linie**. Fokus bleibt Review/Priorisierung und eine reviewbare Anschlusslinie; kein vorgezogener breiter Ausbau.

Diese Datei bleibt auf derselben Referenzsprache aus Serie K/M/L:

- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- `status_evidence_export_surface` als outward Status-/Evidence-Export
- `integration_hook_view` als read-only/caveated Hook-Grenze
- `compatibility backends + internal/legacy worker/domain lanes are extension/internal only`

Code source of truth (keine zweite Integrationswelt):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_FINAL_REFERENCE_LINE`
  - `CANONICAL_COMPUTE_INTEGRATION_CONTRACT_VIEW`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`
- `runtime/ucf-ops/src/lib.rs` (`run_compute_probe`)
- `runtime/ucf-runtime/src/orchestrator.rs` (`RuntimeOrchestrator::try_new_from_env`)
- `runtime/ucf-replay/src/lib.rs` (`replay_records`)
- `runtime/ucf-bench/src/main.rs` (`run_compute`)

## 1) Linien nebeneinander: finaler Kern, erste Post-Core-Linie, breitere Kandidaten

Die Einordnung bleibt ausdrücklich dreigeteilt:

| Linie | Inhalt | Status |
|---|---|---|
| finale Compute-Linie | `submit -> compute_canonical -> result/fault/status -> execution_snapshot` + canonical status/evidence export | abgeschlossen / Referenzkern |
| erste Post-Core-Integrationslinie | `ops_compute_probe` über `submit` + `status_evidence_export_surface` + `canonical_consumer_view` | aligned post-core integration (aus Serie M) |
| spätere breitere Systemintegration | weitere Serie-N-Kandidaten (`runtime_orchestrator_env_bootstrap`, `replay_diff_backend_recompute`, etc.) | nur reviewbar, nicht vorweg implementiert |

Damit ist klar: Serie N dokumentiert Anschlussfähigkeit, baut aber keine zweite Integrationswelle.

## 2) Prompt-1 Kandidaten hart gegengeprüft (ohne neue hypothetische Kandidaten)

Bewertungsachsen (rein technisch):
1. **Semantikpassung** zu outward execution/status/evidence semantics.
2. **Integrationsschnitt**: schmale Anbindung vs. Kern wieder aufreißen.
3. **Landepfad**: finale Compute-Linie vs. internal/legacy Pfade.

| candidate | outward execution/status/evidence fit | schmale Integration möglich? | Landepfad | harte Einordnung |
|---|---|---|---|---|
| `ops_compute_probe` | hoch: nutzt `submit` + `status_evidence_export_surface` | ja | finale Compute-Linie | technisch sauberster Referenzkandidat |
| `runtime_orchestrator_env_bootstrap` | mittel: load-bearing, aber Mixed-Intake (`build_backend` + summary) | ja, wenn Intake schrittweise auf canonical submit/export verschoben wird | heute gemischt, Ziel finale Linie | hoher Hebel, aber caveated |
| `replay_diff_backend_recompute` | niedrig-mittel: Replay-/Diff-Nutzen, kein outward service contract | begrenzt; uplifting würde semantisch überdehnen | compat/internal lane | nur plausibel unter klaren Grenzen |
| `bench_compute_subcommand` | niedrig: benchmark harness ohne outward semantics | nicht sinnvoll als Integrationsarbeit | internal/dev-test only | kein breiter Integrationskandidat |
| `domains_ai_compat_lane` | niedrig: historische ABI/compat Signale statt canonical export semantics | nein, würde legacy Richtung stärken | legacy/compat boundary | für breitere Integration jetzt nicht sinnvoll |
| `runtime hooks / frame helpers` | sehr niedrig: summary/frame Leser ohne stabile outward contract-Bindung | nein | runtime-internal helper paths | kein meaningful candidate |

## 3) Schmale Anschlusslinien-Sicht (`integration_follow_on_view`)

Nur vier reviewbare Klassen, ohne Roadmap- oder Governance-Overlay:

- `already_aligned`
- `first_post_core_aligned`
- `broader_review_candidate`
- `not_pursued_now`

Abbildung (schmal, technisch):

| candidate/lane | integration_follow_on_view | Kurzbegründung |
|---|---|---|
| `final_compute_reference_line` | `already_aligned` | Finaler Referenzkern ist abgeschlossen und bleibt unverändert maßgeblich. |
| `ops_compute_probe` | `first_post_core_aligned` | Erste Post-Core-Integration ist bereits auf canonical submit/status/evidence gebunden. |
| `runtime_orchestrator_env_bootstrap` | `broader_review_candidate` | Breiter Runtime-Hebel, aber heute Mixed-Intake; nur als reviewbarer Anschlusskandidat geführt. |
| `replay_diff_backend_recompute` | `broader_review_candidate` | Vergleichspfad mit Nutzen, aber nicht als outward Service-Contract. |
| `domains_ai_compat_lane` | `not_pursued_now` | Legacy-/Compat-getrieben, nicht über canonical outward semantics motiviert. |
| `bench_compute_subcommand` | `not_pursued_now` | Interner Benchmark-Pfad ohne outward execution/status/evidence-Ziel. |
| `runtime hooks / frame helpers` | `not_pursued_now` | Hilfssignale ohne stabile outward Vertragsbindung. |

## 4) Schmale `candidate_priority_view` (minimal, nicht als Portfolio-Matrix)

Nur minimale Prioritätsklassen:
- `high_leverage_aligned_candidate`
- `plausible_but_caveated_candidate`
- `low_value_or_legacy_driven_candidate`
- `not_worth_broader_integration_now`

Abbildung der bestehenden Serie-N-Kandidaten:

| candidate | candidate_priority_view | Kurzbegründung |
|---|---|---|
| `ops_compute_probe` | `high_leverage_aligned_candidate` | Bereits auf finaler submit/status/evidence Linie; outward direkt nutzbar ohne Kernumbau. |
| `runtime_orchestrator_env_bootstrap` | `plausible_but_caveated_candidate` | Load-bearing Runtime-Hebel, aber derzeit Mixed-Intake; nur mit schmaler Canonicalisierung sinnvoll. |
| `replay_diff_backend_recompute` | `plausible_but_caveated_candidate` | Technisch nützlich als Vergleichspfad, aber kein outward contract uplift; strikt boundary-gebunden halten. |
| `domains_ai_compat_lane` | `low_value_or_legacy_driven_candidate` | Attraktivität kommt primär aus Legacy-Kopplung, nicht aus Semantikpassung zur finalen Linie. |
| `bench_compute_subcommand` | `not_worth_broader_integration_now` | Benchmark-Harness; kein outward execution/status/evidence Mehrwert. |
| `runtime hooks / frame helpers` | `not_worth_broader_integration_now` | Nähe ist rein daten-/helper-basiert, nicht contract-basiert. |

## 5) Priorisierungskriterium explizit (gegen Nähe-Bias)

Priorisierung erfolgt **nicht** über historische Nähe, sondern nur wenn zusammen erfüllt:

1. Anschluss an finale Compute-Linie (`submit -> ... -> status/evidence export`).
2. Semantikpassung zu outward execution/status/evidence statt interner Hilfssignale.
3. Outward-facing Nutzbarkeit für reale UCF-Systemflächen.
4. Kein neuer Kernumbau erforderlich (schmale Integration genügt).

Fehlt einer dieser Punkte deutlich, wird ein Kandidat zurückgestuft.

## 6) Priorisierte breitere Kandidaten: minimale reviewbare Anschlussaussagen

Nur minimale Aussagen, keine Vorab-Implementierung:

1. `runtime_orchestrator_env_bootstrap` (`broader_review_candidate`)
   - warum anschlussfähig: load-bearing Runtime-Intake mit realer Systemhebelwirkung.
   - genügende outward-facing Contracts: canonical `submit` + `status_evidence_export_surface` (status/evidence), ohne neue Sonderverträge.
   - warum jetzt nicht implementiert: Mixed-Intake braucht separaten Review-Entscheid; Serie N hält nur den Anschlusskorridor fest.

2. `replay_diff_backend_recompute` (`broader_review_candidate`)
   - warum anschlussfähig: technisch wertvoll für Drift-/Vergleichssicht.
   - genügende outward-facing Contracts: kein neuer outward Runtime-Contract; nur Referenz auf bestehende status/evidence Linien zur Vergleichbarkeit.
   - warum jetzt nicht implementiert: uplift zu produktivem Outward-Consumer würde die boundary semantisch überdehnen.

## 7) Versteckte Legacy-Abhängigkeit explizit zurückgestuft

Bewusst zurückgestuft/ausgeschlossen für breitere Integration jetzt:

- `domains_ai_compat_lane` → `low_value_or_legacy_driven_candidate`
  - Haupthebel wäre Zugriff über legacy/compat Adapter statt canonical outward semantics.
- `replay_diff_backend_recompute` bleibt caveated
  - Nur als Replay-/Vergleichspfad sinnvoll; kein disguised outward Runtime-Contract.
- `bench_compute_subcommand`, `runtime hooks / frame helpers` → `not_worth_broader_integration_now`
  - rein interne/dev-nahe Pfade ohne semantisch sauberen Anschluss an finale Compute-Linie.

## 8) Keine implizite "schon gebaut"-Sprache

- `broader_review_candidate` bedeutet ausschließlich reviewbarer Anschlusskandidat.
- Es wird kein Kandidat als bereits integriert markiert, wenn er nicht `already_aligned` oder `first_post_core_aligned` ist.
- Serie N dokumentiert bewusst Anschlusslinien statt Ausbauarbeit.

## 9) Nur 1-3 echte nächste breite Integrationsrichtungen

Keine Wunschliste; nur technisch sinnvolle nächste Richtungen:

1. **`runtime_orchestrator_env_bootstrap` schmal canonicalisieren** (`plausible_but_caveated_candidate` → Ziel: aligned)
   - Warum: größter breiter Runtime-Hebel bei gleichzeitig möglicher schrittweiser Anpassung ohne Kernaufbruch.
2. **`ops_compute_probe` als Referenzanker stabil halten** (`high_leverage_aligned_candidate`)
   - Warum: bereits sauberer outward Consumer; dient als Drift-/Semantikanker für weitere Integration.
3. **`replay_diff_backend_recompute` boundary-klar halten statt upliften** (`plausible_but_caveated_candidate`)
   - Warum: nützlich als technische Vergleichsfläche, aber bewusst kein outward Integrationsziel.

## 10) Doku-Rückbindung (keine zweite Wahrheitsquelle)

- Serie N priorisiert nur die bereits in `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP` sichtbaren Kandidaten.
- Serien K/M/L bleiben semantische Leitplanken; diese Datei ergänzt nur die **harte Priorisierungsschicht**.
- Keine zusätzliche Governance-/Portfolio-/Roadmap-Struktur.

## 11) Kleine Konsistenzchecks (nur nötig für final-line Anschlussfähigkeit)

Für priorisierte Kandidaten gilt als Mindestcheck:
- `already_aligned` und `first_post_core_aligned` müssen auf canonical submit/status/evidence Bezug halten.
- `broader_review_candidate` darf nicht als bereits integriert/formal aligned zur finalen Linie formuliert sein.
- `not_pursued_now` bleibt explizit außerhalb aktueller Integrationsarbeit.
- Die bestehende `candidate_priority_view`-Klassifikation bleibt rein ergänzend und darf keine implizite Implementierungszusage erzeugen.

Diese Konsistenz ist über die bestehende Code-/Doku-Kopplung in `runtime/ucf-compute/src/reference_map.rs` Tests abgesichert.

## 12) Serie-N-Abschlussmatrix (harte Abschlussprüfung, repo-basiert)

Die Abschlussmatrix zieht nur die bereits im Repo belegten Flächen zusammen; keine neuen Kandidaten:

| Fläche | Abschlussstatus | Repo-basierte Kurzbegründung |
|---|---|---|
| `runtime_orchestrator_env_bootstrap` | `genuine next integration candidate` | In `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP` als `needs_final_integration_adjustment` + `mostly_aligned_with_caveats` geführt; hoher Hebel bei schmaler Canonicalisierung möglich. |
| `replay_diff_backend_recompute` | `plausible but deferred` | Als `legacy_compat_path` + `mixed_transitional` klassifiziert; technisch nützlich, aber kein outward service contract uplift. |
| `domains_ai_compat_lane` | `reviewed and not pursued now` | Explizite Legacy-/Compat-Grenze (`legacy_compat_path`, `internal_only_not_true_outward_consumer`), kein sinnvoller Anschluss an outward canonical status/evidence. |
| `bench_compute_subcommand` | `not meaningful as compute-facing integration` | Internal/dev-test benchmark harness (`internal_dev_test_only`), kein outward execution/status/evidence Ziel. |
| `runtime hooks / frame helpers` | `not meaningful as compute-facing integration` | Hilfspfad ohne stabile outward Vertragsbindung (`integration_hook_view` bleibt read-only/caveated boundary). |
| `ops_compute_probe` | bereits integriert (Referenzanker, kein nächster Kandidat) | Einziger Consumer mit `aligned_to_final_compute_line`; bleibt Baseline, nicht neuer Ausbaukandidat. |

## 13) Explizite breitere Systemintegrations-Review-Linie nach Abschluss

Nach Serie N gilt explizit und abschließend:

- **Realer breiter Anschlusskandidat:** nur `runtime_orchestrator_env_bootstrap`.
- **Bewusst reviewt, jetzt nicht verfolgt:** `replay_diff_backend_recompute`, `domains_ai_compat_lane`.
- **Bewusst außerhalb compute-facing Integrationsarbeit:** `bench_compute_subcommand`, `runtime hooks / frame helpers`.
- **Kein Rückfall in Compute-Core-Arbeit:** Diese Linie ist ausschließlich Review/Priorisierung auf bestehenden finalen Contracts (`submit`, `status`, `status_evidence_export_surface`), nicht neue Core-Ausbauarbeit.

## 14) Nächste Richtungen nach Serie N (1-3), mit harter Priorisierung

Keine Wunschliste; nur technische Hebel auf Basis der Abschlussmatrix:

1. **Serie P (priorisiert): targeted rollout-Checks auf stabilisiertem `ops_compute_probe` + orchestrator outcome**
   - nächster direkte Integrationshebel auf bestehender Abschlussmatrix.
2. **Serie Q (nachrangig): erneuter broader adoption review nach Integrationsschritt**
   - erst danach belastbar, sonst droht Wiederholung derselben Review-Befunde ohne neuen Integrationsfortschritt.
3. **Serie O (geschlossen, nicht Ausbaupfad): maintenance-only Nachlaufkanon am Compute-Kern**
   - nur kleine maintenance-safe Korrekturen, keine Integrations- oder Capability-Erweiterung.

**Exakte Priorität zuerst: Serie P.**

## 15) Maintenance-only guardrail für den abgeschlossenen Compute-Kern (Serie O)

Serie N bleibt eine Review-/Priorisierungslinie für breitere Integration.
Die Maintenance-only Grenze für den abgeschlossenen Compute-Kern ist separat explizit festgezogen in:
- `docs/compute_core_maintenance_boundary_serie_o_v1.md`

Dabei gilt für Folgearbeit nach Serie N:
- `broader_review_candidate` bleibt Integrationsarbeit und ist **nicht automatisch** maintenance-only.
- Core-nahe Änderungen in `runtime/ucf-compute/*` müssen in den Minimal-Nachlaufkanon eingeordnet werden (`allowed_maintenance_safe_changes`, `discouraged_but_possible_with_care`, `not_in_maintenance_lane`; code-seitig gespiegelt durch `maintenance_safe_change`, `maintenance_safe_with_care`, `not_maintenance_only_requires_new_integration_or_buildout`).
- Alles außerhalb maintenance-only bleibt als neue Integration/Buildout zu behandeln.
