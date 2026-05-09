# Blue-Brain Audit Baseline Map v1

Stand: 2026-05-09 (UTC).

Ziel dieser Referenz ist eine **kanonische, reproduzierbare Audit-Baseline** für den aktuellen Blue-Brain-Stand nach BR6, IR1, MD2, MD3 und SC1, ohne neue Feature-Arbeit. Maßgeblicher HEAD für diesen Refresh ist `9f263aac7e146bf58c65c8f17e467ec710486100`.

## Authority scope of this baseline

Diese Baseline belegt die aktuelle maintenance-facing Evidenzlage für:

- sechs bounded anatomische Regionen: Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum und Hypothalamus;
- IR1 bounded inter-region semantics as read/diagnostic/advisory/reference classes only;
- MD2 exactly one maintenance-hardened first model-deepening pair (`Amygdala ↔ Thalamus`);
- MD3 exactly one bounded second model-deepening pair (`Amygdala ↔ Basal Ganglia`);
- SC1 maintenance-ready closure, jetzt als clean maintenance-ready baseline belegt.

Sie belegt **keine** neue Region, keine weitere Modellvertiefung, keine inter-region platform, keine globale Modell-/Neurodynamikplattform, keine Planner-/Agenten-/Policy-/Retry-Logik und keine Compute-Core-Erweiterung.

## Audit-Zustände (kanonisch)

- **clean maintenance-ready baseline**
  - Kanonische Checks laufen frisch durch.
  - Reports liegen unter `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/` und referenzieren HEAD `9f263aac7e146bf58c65c8f17e467ec710486100`.
  - Root-Reports liegen zusätzlich unter `out/docs_lint_report.json` und `out/gate_report.json`.
  - `cargo_fmt_check.log` ist maintenance-facing selbsterklärend: Command, HEAD und PASS/OK-Marker stehen im Log.

- **accepted tracked audit artifact**
  - Versionierte Audit-Referenzen in `docs/` (diese Datei, SC1 Prompt 2, die historische finale 2026-05-08 Evidence-Abschlussnotiz und der 2026-05-09 Evidence-Successor).
  - Versionierte Baseline-Reports unter `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/`.

- **historical baseline trace**
  - Ältere Baselines wie `out/blue_brain_audit_baseline_2026-05-02/`, `out/blue_brain_audit_baseline_2026-05-04/`, `out/blue_brain_audit_baseline_2026-05-08/`, die unqualifizierte Same-Day-Vorgänger-Baseline `out/blue_brain_audit_baseline_2026-05-09/` und frühere HEAD-qualifizierte Same-Day-Läufe wie `out/blue_brain_audit_baseline_2026-05-09_head_13615edd74/`.
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
2. `cargo run -p ucf-ops -- docs lint --strict --out ./out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/docs_lint_report.json`
3. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/gate_report.json`

Zusätzliche Repo-/PR-Hygiene für diesen Pass:

4. `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
5. `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
6. `cargo fmt --all -- --check`
7. `cargo clippy --workspace --all-targets -- -D warnings`

## Reproduzierbare Artifact-Lage

Alle aktuellen Audit-Baseline-Ergebnisse liegen gebündelt unter:

- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/head_status.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/docs_lint_root.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/readiness_gate_root.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/gate_report.json`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/cargo_fmt_check.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/cargo_clippy_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/consistency_checks.log`
- `out/docs_lint_report.json`
- `out/gate_report.json`

Die beiden aktuellen Gate-Reports (`out/gate_report.json` und `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/gate_report.json`) tragen `code_version_tag = 9f263aac7e146bf58c65c8f17e467ec710486100`.

## Historical baseline treatment

- `out/blue_brain_audit_baseline_2026-05-02/` bleibt eine historische BB29/pre-BR6 Vergleichsspur; die dortige `workspace.features`-Cargo-Warnung ist historisch.
- `out/blue_brain_audit_baseline_2026-05-04/` bleibt eine historische Übergangsspur mit älterem `code_version_tag`.
- `out/blue_brain_audit_baseline_2026-05-08/` bleibt die unmittelbar vorherige SC1-Evidence-Spur auf HEAD `913f6ea50e47dcb4d980441d5fbd18d17e859f0b`; sie ist nicht mehr current operative evidence.
- `out/blue_brain_audit_baseline_2026-05-09/` bleibt eine unqualifizierte Same-Day-Vorgänger-Spur.
- `out/blue_brain_audit_baseline_2026-05-09_head_13615edd74/` bleibt eine frühere HEAD-qualifizierte Same-Day-Evidence; maßgeblich ist `out/blue_brain_audit_baseline_2026-05-09_head_9f263aac7e/`.
- Historische Baselines dürfen nicht als aktuelle operative Regions-/Relations-/Modelllage gelesen werden.
- Bei Widerspruch zwischen historischen Baselines und der Authority Map gilt `docs/blue_brain_authority_chain_status_map.md` plus diese aktuelle Baseline-Map.

## Audit-Claim-Grenze

Diese Baseline erlaubt Aussagen zu:

- reproduzierbarer Ausführbarkeit der kanonischen Repo-Checks im aktuellen maintenance-facing Blue-Brain-Stand,
- konsistenter Ablage der verwendeten Audit-Reports,
- sauberer Trennung von aktuellen Reports, historischen Baselines und non-canonical leftover artifacts,
- aktualisierter `code_version_tag`-Evidence auf den aktuellen HEAD,
- selbsterklärender fmt-Evidence,
- geklärter Trennung zwischen Architecture-Lane und tatsächlich aktiv implementiertem Relationsstatus in den current-authority-nahen Relationsdokumenten.

Sie macht **keine** zusätzliche Aussage über nicht ausgeführte Matrix-/Umgebungsvarianten und erzeugt keine operative Autorität neben der Authority Chain.

## Abschlussnotiz

Die aktuelle Evidence-/Baseline-Abschlussnotiz liegt unter `docs/blue_brain_final_evidence_baseline_refresh_2026_05_09.md` und bestätigt den Status **clean maintenance-ready baseline** auf HEAD `9f263aac7e146bf58c65c8f17e467ec710486100`.
