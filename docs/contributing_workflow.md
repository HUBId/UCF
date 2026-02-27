# Contributing Workflow (Prompt Series + Gates)

This document defines the minimal, deterministic workflow for contributors and Codex.

## 1) Pick and prepare work

1. Identify prompt scope from:
   - `docs/prompt_series_index.md`
   - `docs/prompt_rulebook.md`
2. If using queue automation, use:
   ```bash
   python scripts/prompt_runner.py next
   ```
3. Create branch using policy in `docs/branch_policy.md`.

## 2) Author the next prompt (when needed)

1. Use template:
   - `docs/codex_prompt_template.txt`
2. Optional render helper:
   ```bash
   python scripts/prompt_runner.py render --id <id> --template docs/codex_prompt_template.txt
   ```
3. Update prompt tracking:
   ```bash
   python scripts/prompt_runner.py self-check
   ```

## 3) Implement change + impact-aware checks

Use change-impact planning when behavior/code changes:

```bash
cargo run -p ucf-ops -- change-impact --base HEAD~1 --head HEAD --out ./out/change_impact_plan.md
```

Minimal check plan by change type:

- Docs-only:
  - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
- Runtime or policy changes:
  - `cargo test --workspace`
  - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
  - `cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json`
  - `cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test`
- Tooling/ops changes:
  - docs lint + readiness gate at minimum.

Formatting/lint baseline:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

## 4) Prepare signoff artifacts

Use deterministic `./out/` paths and attach to PR:
- `./out/docs_lint_report.json`
- `./out/gate_report.json`
- `./out/adversarial_report.json`
- `./out/change_impact_plan.md` (if generated)

If spec-affecting changes exist, regenerate and commit snapshot:

```bash
cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
```

For release signoff validation (when applicable):

```bash
cargo run -p ucf-ops -- release signoff --validate --out ./out/<run_id> --emit release/v0_signoff_result.json --checklist release/v0_signoff_checklist.toml
```

## 5) Finalize prompt run status

After merge-ready completion:

```bash
python scripts/prompt_runner.py done <id>
```

If blocked/failing:

```bash
python scripts/prompt_runner.py fail <id> --reason "<reason>"
```
