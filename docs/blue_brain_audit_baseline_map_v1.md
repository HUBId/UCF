# Blue-Brain Audit Baseline Map v1

Stand: 2026-05-08 (UTC).

Ziel dieser Referenz ist eine **kanonische, reproduzierbare Audit-Baseline** für den aktuellen Blue-Brain-Stand nach BR6, IR1, MD2, MD3 und SC1, ohne neue Feature-Arbeit.

## Authority scope of this baseline

Diese Baseline belegt die aktuelle maintenance-facing Evidenzlage für:

- sechs bounded anatomische Regionen: Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus;
- IR1 bounded inter-region semantics as read/diagnostic/advisory/reference classes only;
- MD2 exactly one maintenance-hardened first model-deepening pair (`Amygdala ↔ Thalamus`);
- MD3 exactly one bounded second model-deepening pair (`Amygdala ↔ Basal Ganglia`);
- SC1 maintenance-ready-with-caveats closure.

Sie belegt **keine** neue Region, keine weitere Modellvertiefung, keine inter-region platform, keine globale Modell-/Neurodynamikplattform, keine Planner-/Agenten-/Policy-/Retry-Logik und keine Compute-Core-Erweiterung.

## Audit-Zustände (kanonisch)

- **clean reproducible baseline**
  - Kanonische Checks laufen frisch durch.
  - Reports liegen unter `out/blue_brain_audit_baseline_2026-05-08/`.
  - Root-Reports liegen zusätzlich unter `out/docs_lint_report.json` und `out/gate_report.json`.

- **accepted tracked audit artifact**
  - Versionierte Audit-Referenzen in `docs/` (diese Datei und SC1 Prompt 2).
  - Versionierte Baseline-Reports unter `out/blue_brain_audit_baseline_2026-05-08/`.

- **historical baseline trace**
  - Ältere Baselines wie `out/blue_brain_audit_baseline_2026-05-02/` und `out/blue_brain_audit_baseline_2026-05-04/`.
  - Nur Vergleichs-/Auditspur; nicht die aktuelle post-BR6/IR1/MD2/MD3/SC1 Evidenz.

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
2. `cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`
3. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-08/gate_report.json`

Zusätzliche Repo-/PR-Hygiene für diesen Pass:

4. `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
5. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
6. `cargo fmt --all -- --check`
7. `cargo clippy --workspace --all-targets -- -D warnings`

## Reproduzierbare Artifact-Lage

Alle aktuellen Audit-Baseline-Ergebnisse liegen gebündelt unter:

- `out/blue_brain_audit_baseline_2026-05-08/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-08/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-08/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-08/gate_report.json`
- `out/docs_lint_report.json`
- `out/gate_report.json`

## Historical baseline treatment

- `out/blue_brain_audit_baseline_2026-05-02/` bleibt eine historische BB29/pre-BR6 Vergleichsspur.
- `out/blue_brain_audit_baseline_2026-05-04/` bleibt eine historische Übergangsspur.
- Historische Baselines dürfen nicht als aktuelle operative Regions-/Relations-/Modelllage gelesen werden.
- Bei Widerspruch zwischen historischen Baselines und der Authority Map gilt `docs/blue_brain_authority_chain_status_map.md`.

## Audit-Claim-Grenze

Diese Baseline erlaubt Aussagen zu:

- reproduzierbarer Ausführbarkeit der kanonischen Repo-Checks im aktuellen maintenance-facing Blue-Brain-Stand,
- konsistenter Ablage der verwendeten Audit-Reports,
- sauberer Trennung von aktuellen Reports, historischen Baselines und non-canonical leftover artifacts,
- beseitigter Cargo-Maintenance-Noise bezüglich des früheren unsupported `workspace.features` Root-Manifesteintrags.

Sie macht **keine** zusätzliche Aussage über nicht ausgeführte Matrix-/Umgebungsvarianten und erzeugt keine operative Autorität neben der Authority Chain.
