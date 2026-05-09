# Blue-Brain Final Evidence Baseline Refresh — 2026-05-08

Stand: 2026-05-08 (UTC).

> Historical note (2026-05-09): This 2026-05-08 evidence note is retained as a historical SC1 baseline on HEAD `913f6ea50e47dcb4d980441d5fbd18d17e859f0b`. The current operative baseline successor is `docs/blue_brain_final_evidence_baseline_refresh_2026_05_09.md` on HEAD `efbeec23b752744dc9f87a2e2e3eeb9efe25104f`.


Diese Abschlussnotiz dokumentiert den historischen 2026-05-08 Evidence-/Baseline-Refresh für den damaligen Blue-Brain-Gesamtstand. Sie erzeugt **keine** neue Regionenfunktionalität, keinen weiteren Vertiefungskandidaten, keine Planner-/Agenten-/Policy-Logik und keine Compute-Core-Erweiterung.

## Zielzustand

- Maßgeblicher Baseline-Lauf: `out/blue_brain_audit_baseline_2026-05-08/`
- Referenzierter HEAD: `913f6ea50e47dcb4d980441d5fbd18d17e859f0b`
- Historischer Reproduzierbarkeitsstatus: **clean reproducible baseline**; current successor ist die 2026-05-09 clean maintenance-ready baseline
- Danach genügt normaler Maintenance-/Bugfix-/Cleanup-Modus; es gibt keinen aktiven neuen Ausbauhebel.

## Aktualisierte Reports und Logs

Der historische Refresh hat die damaligen Root-Reports und die gebündelte Baseline unter `out/blue_brain_audit_baseline_2026-05-08/` erzeugt bzw. synchronisiert:

- `out/docs_lint_report.json`
- `out/gate_report.json`
- `out/blue_brain_audit_baseline_2026-05-08/head_status.log`
- `out/blue_brain_audit_baseline_2026-05-08/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint_root.log`
- `out/blue_brain_audit_baseline_2026-05-08/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-08/readiness_gate_root.log`
- `out/blue_brain_audit_baseline_2026-05-08/gate_report.json`
- `out/blue_brain_audit_baseline_2026-05-08/cargo_fmt_check.log`
- `out/blue_brain_audit_baseline_2026-05-08/cargo_clippy_workspace.log`

Im historischen Lauf trugen `out/gate_report.json` und `out/blue_brain_audit_baseline_2026-05-08/gate_report.json` `code_version_tag = 913f6ea50e47dcb4d980441d5fbd18d17e859f0b`.

## Tatsächlich ausgeführte Kommandos

1. `git rev-parse HEAD`
2. `git status --short --branch`
3. `cargo test --workspace`
4. `cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`
5. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-08/gate_report.json`
6. `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
7. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
8. `cargo fmt --all -- --check`
9. `cargo clippy --workspace --all-targets -- -D warnings`

Alle oben genannten Kommandos liefen erfolgreich durch. Die Readiness-Gate-Testprofile enthalten weiterhin dokumentierte `SKIP`-Checks für testprofilbedingte oder fixturebedingte Nicht-Erzwingung; der Gesamtreport steht auf `PASS` und ist kein Caveat für diese Baseline.

## Cargo-Warnungsstatus

Die frühere Cargo-Warnung zum unsupported Root-Manifesteintrag `workspace.features` ist im frischen 2026-05-08-Lauf nicht mehr aufgetreten. Sie existiert nur noch in historischen Logs unter `out/blue_brain_audit_baseline_2026-05-02/` und darf nicht als aktueller Baseline-Befund gelesen werden.

## Historische Baselines

- `out/blue_brain_audit_baseline_2026-05-02/` bleibt historische Vergleichs-/Auditspur mit älteren Cargo-Warnungen und älterem `code_version_tag`.
- `out/blue_brain_audit_baseline_2026-05-04/` bleibt historische Übergangsspur mit älterem `code_version_tag`.
- Aktuelle operative Evidence ist nicht mehr der 2026-05-08-Lauf, sondern ausschließlich der 2026-05-09-Lauf plus die Root-Reports `out/docs_lint_report.json` und `out/gate_report.json`.

## Abschlussentscheidung

Historisch hob dieser Refresh den Blue-Brain-Gesamtstand auf eine **clean reproducible baseline** auf dem dokumentierten damaligen HEAD; current authority ist der 2026-05-09 clean maintenance-ready successor. Normale Maintenance-/Bugfix-/Cleanup-Arbeit genügt; es wird kein neuer Ausbau- oder Vertiefungspfad eröffnet.
