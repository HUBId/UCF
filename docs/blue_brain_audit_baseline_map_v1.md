# Blue-Brain Audit Baseline Map v1

Stand: 2026-05-02 (UTC).

Ziel dieser Referenz ist eine **kanonische, reproduzierbare Audit-Baseline** für den aktuellen Blue-Brain-/Zwei-Regionen-Stand, ohne neue Feature-Arbeit.

## Audit-Zustände (kanonisch)

- **clean reproducible baseline**
  - Arbeitsbaum ist sauber (`git status --short` leer).
  - Kanonische Checks laufen frisch durch.
  - Reports liegen unter `out/blue_brain_audit_baseline_2026-05-02/`.

- **accepted tracked audit artifact**
  - Versionierte Audit-Referenzen in `docs/` (diese Datei).
  - Versionierte Baseline-Reports unter `out/blue_brain_audit_baseline_2026-05-02/`.

- **ignored/generated artifact**
  - Laufzeit-/lokale Ephemera gemäß `.gitignore` (z. B. `target/`, `.ucf/`, runtime-spezifische `out/` unter Subprojekten).

- **unresolved workspace noise**
  - Unversionierte oder geänderte Dateien ohne klare Audit-Einordnung.
  - Bei Auftreten: vor Audit-Aussage bereinigen/klassifizieren.

- **non-canonical leftover artifact**
  - Verstreute Einzeloutputs außerhalb der Baseline-Struktur unter `out/`.
  - Nicht als Audit-Grundlage verwenden; entweder entfernen oder in Baseline-Pfad überführen.

## Kanonische Prüfschritte

Aus `AGENTS.md` (Repo-root) werden für diese Baseline folgende Schritte verwendet:

1. `cargo test --workspace`
2. `cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-02/docs_lint_report.json`
3. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-02/gate_report.json`

Zusätzliche Repo-/PR-Hygiene für diesen Pass:

4. `cargo fmt --all -- --check`
5. `cargo clippy --workspace --all-targets -- -D warnings`

## Reproduzierbare Artifact-Lage

Alle Audit-Baseline-Ergebnisse liegen gebündelt unter:

- `out/blue_brain_audit_baseline_2026-05-02/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-02/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-02/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-02/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-02/gate_report.json`

## Audit-Claim-Grenze

Diese Baseline erlaubt Aussagen zu:
- reproduzierbarer Ausführbarkeit der kanonischen Repo-Checks,
- aktuellem, sauberen Workspace-Status,
- konsistenter Ablage der verwendeten Audit-Reports.

Sie macht **keine** zusätzliche Aussage über nicht ausgeführte Matrix-/Umgebungsvarianten.
