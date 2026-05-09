# Blue-Brain Audit Anchor Summary — 2026-05-09

- audit_target_head: `c500cc14ae5c3ba2347933ba959ef0a6629ceee3`
- audit_target_short: `c500cc14ae`
- baseline_dir: `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/`
- status: clean maintenance-ready baseline for this commit-bound anchor

## Anchor rule

All reports, logs and baseline references in this bundle apply exactly to audit target head `c500cc14ae5c3ba2347933ba959ef0a6629ceee3`. Later commits or merges are not automatically part of this baseline; they require a deliberate new refresh. Later commits do not retroactively invalidate this completed anchor.

## Commands executed

- `git rev-parse HEAD`
- `git status --short --branch`
- `cargo test --workspace`
- `cargo run -p ucf-ops -- docs lint --strict --out out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/docs_lint_report.json`
- `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
- `cargo run -p ucf-ops -- readiness-gate --profile test --out out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/gate_report.json`
- `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`

## Reports/logs refreshed

- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/head_status.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/cargo_test_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/docs_lint.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/docs_lint_report.json`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/docs_lint_root.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/readiness_gate.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/readiness_gate_root.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/gate_report.json`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/cargo_fmt_check.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/cargo_clippy_workspace.log`
- `out/blue_brain_audit_baseline_2026-05-09_head_c500cc14ae/consistency_checks.log`
- `out/docs_lint_report.json`
- `out/gate_report.json`

The gate report code_version_tag is `c500cc14ae5c3ba2347933ba959ef0a6629ceee3`.
