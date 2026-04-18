# Serie L: Final Technical Exit Dossier (Prompt 3) v1

Stand: Repo-Zustand am 2026-04-18.

Ziel: kompakte, technisch belastbare Exit-Zusammenfassung für den Real-Compute-Kern ohne zweite Wahrheitsquelle neben der finalen Referenzlinie.

Primäre Rückbindung (autoritative Quelle):
- `docs/final_reference_line_serie_j_v1.md`
- `runtime/ucf-compute/src/reference_map.rs` (`CANONICAL_FINAL_REFERENCE_LINE`, `CANONICAL_COMPUTE_REFERENCE_MAP`)
- `runtime/ucf-compute/src/contracts.rs` (`CROSS_CUTTING_PRODUCTION_INVARIANTS_V1`, `CANONICAL_RUNTIME_HANDOFF_SEMANTICS_V1`)

## 1) Canonical production line

Für den technischen Exit gilt exakt die kanonische Produktionslinie:

- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- Rollout/Replay/Diagnostics/Expert bleiben Erweiterungen auf derselben Kernsemantik, nicht zweite Produktionskerne.

Damit bleibt die outward Produktionsautorität auf einer Linie gebunden.

## 2) Stable core areas

Stabil und final exit-fähig sind:

1. **Execution Core** über `CanonicalComputeEntryPoint::submit` und `ComputePipelineBackend::compute_canonical`.
2. **Cross-cutting Kerninvarianten** (`blocked!=failed!=no_op`; getrennte `partial/stale/caveated/degraded`-Semantik).
3. **Canonical handoff semantics** über Execution/Diagnostics/Replay/Rollout/ExpertAction mit `complete|partial|caveated|blocked`.
4. **Outward status/evidence Exportlinie** als Adapter auf denselben Runtime-Kernzustand.

## 3) Constrained but accepted areas

Technisch constrained, aber für finalen Exit akzeptiert:

1. **Rollout/Replay strictness boundary** (`replay_preflight -> replay_with_entry`) bleibt fail-closed bei unvollständiger Grundlage (`insufficient`/`blocked`).
2. **Expert runtime control** (`run_operation_with_entry`, `replay_with_entry`) bleibt high-trust/internal und an shared core invariants gebunden.
3. **Outward caveated integration signals** bleiben read-only und transportieren Constraints explizit statt semantischer Verwässerung.

Diese Constraints sind akzeptierte Schutzkanten der produktiven Linie, keine Restdefekte.

## 4) Intentionally deferred areas

Bewusst nicht als final exit-authority akzeptiert:

1. **Compatibility/legacy lanes** (`build_backend(kind=stub|candle)`, worker/domain-compatibility lanes) als outward Produktionsautorität.
2. **Umdeutung interner/dev Lanes** zu generischen outward Contracts.
3. **Deep accelerator/fleet orchestration Plattformlogik** außerhalb der kanonischen Kernlinie.

Diese Bereiche bleiben technisch sichtbar, aber außerhalb der finalen outward Autoritätsgrenze.

## 5) Outward-facing integration stance

Outward Integrationen binden ausschließlich an:

- `CanonicalComputeEntryPoint::status_evidence_export_surface()`
- `RuntimeOpsSnapshot::integration_hook_view()` mit read-only Klassen (`read_only_integration_safe`, `caveated_conditional`)

`expert_only` und `internal_dev_test_only` bleiben nicht-outward. Dadurch gibt es keine zweite outward Semantikschicht neben der kanonischen Produktionslinie.

## 6) Serie-L Abschlussabgleich (Prompt 1 + Prompt 2)

Diese Datei ist die knappe finale Synthese aus:

- `docs/real_compute_exit_edge_review_serie_l_v1.md` (load-bearing edge review)
- `docs/real_compute_exit_boundary_serie_l_prompt2_v1.md` (accepted vs not-accepted boundary)

Alle drei Serie-L-Dokumente müssen dieselbe Schlussaussage halten:

- eine kanonische Produktionslinie,
- klar akzeptierte constrained Kanten,
- klar ausgeschlossene outward Authority für deferred/internal Lanes,
- outward Integration über read-only/caveated Export- und Hook-Semantik.

## 7) Kleine Konsistenzchecks (Exit)

Für Drift-Prävention in Serie L genügt der kleine Checkblock:

1. `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
2. `cargo fmt --all`
3. `cargo clippy --workspace --all-targets -- -D warnings`

Optionaler Voll-Check bei Übergabe:

4. `cargo test --workspace`
5. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`

Kein neuer Plattformaufbau, keine zweite Wahrheitsquelle, keine Governance-/Release-Nebenstruktur.
