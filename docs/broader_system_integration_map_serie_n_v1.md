# Serie N: Broader UCF System Integration Map v1 (repo-basiert, schmal, hart priorisiert)

Status: repo-basierte, technisch harte Priorisierung breiterer UCF-Systemflächen gegen die **finale Compute-Linie**. Fokus bleibt Review/Priorisierung, nicht Ausbauprogramm.

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

## 1) Prompt-1 Kandidaten hart gegengeprüft (ohne neue hypothetische Kandidaten)

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

## 2) Schmale `candidate_priority_view` (minimal, nicht als Portfolio-Matrix)

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

## 3) Priorisierungskriterium explizit (gegen Nähe-Bias)

Priorisierung erfolgt **nicht** über historische Nähe, sondern nur wenn zusammen erfüllt:

1. Anschluss an finale Compute-Linie (`submit -> ... -> status/evidence export`).
2. Semantikpassung zu outward execution/status/evidence statt interner Hilfssignale.
3. Outward-facing Nutzbarkeit für reale UCF-Systemflächen.
4. Kein neuer Kernumbau erforderlich (schmale Integration genügt).

Fehlt einer dieser Punkte deutlich, wird ein Kandidat zurückgestuft.

## 4) Versteckte Legacy-Abhängigkeit explizit zurückgestuft

Bewusst zurückgestuft/ausgeschlossen für breitere Integration jetzt:

- `domains_ai_compat_lane` → `low_value_or_legacy_driven_candidate`
  - Haupthebel wäre Zugriff über legacy/compat Adapter statt canonical outward semantics.
- `replay_diff_backend_recompute` bleibt caveated
  - Nur als Replay-/Vergleichspfad sinnvoll; kein disguised outward Runtime-Contract.
- `bench_compute_subcommand`, `runtime hooks / frame helpers` → `not_worth_broader_integration_now`
  - rein interne/dev-nahe Pfade ohne semantisch sauberen Anschluss an finale Compute-Linie.

## 5) Nur 1-3 echte nächste breite Integrationsrichtungen

Keine Wunschliste; nur technisch sinnvolle nächste Richtungen:

1. **`runtime_orchestrator_env_bootstrap` schmal canonicalisieren** (`plausible_but_caveated_candidate` → Ziel: aligned)
   - Warum: größter breiter Runtime-Hebel bei gleichzeitig möglicher schrittweiser Anpassung ohne Kernaufbruch.
2. **`ops_compute_probe` als Referenzanker stabil halten** (`high_leverage_aligned_candidate`)
   - Warum: bereits sauberer outward Consumer; dient als Drift-/Semantikanker für weitere Integration.
3. **`replay_diff_backend_recompute` boundary-klar halten statt upliften** (`plausible_but_caveated_candidate`)
   - Warum: nützlich als technische Vergleichsfläche, aber bewusst kein outward Integrationsziel.

## 6) Doku-Rückbindung (keine zweite Wahrheitsquelle)

- Serie N priorisiert nur die bereits in `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP` sichtbaren Kandidaten.
- Serien K/M/L bleiben semantische Leitplanken; diese Datei ergänzt nur die **harte Priorisierungsschicht**.
- Keine zusätzliche Governance-/Portfolio-/Roadmap-Struktur.

## 7) Kleine Konsistenzchecks (nur nötig für final-line Anschlussfähigkeit)

Für priorisierte Kandidaten gilt als Mindestcheck:
- `high_leverage_aligned_candidate` muss canonical status+evidence Exportpfad enthalten.
- `plausible_but_caveated_candidate` darf nicht als bereits aligned zur finalen Linie markiert sein.
- `low_value_or_legacy_driven_candidate` / `not_worth_broader_integration_now` dürfen nicht als outward aligned auftauchen.

Diese Konsistenz ist über die bestehende Code-/Doku-Kopplung in `runtime/ucf-compute/src/reference_map.rs` Tests abgesichert.
