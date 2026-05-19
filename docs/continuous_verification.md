# Continuous Verification (Nightly Offline Gates)

The nightly pipeline runs scheduled, deterministic, offline-first verification across docs, policy snapshot, safety gates, adversarial checks, goldens, and drift.

## What nightly runs

Primary workflow: `.github/workflows/nightly_verify.yml`.

Linux nightly executes:
- `cargo test --workspace --all-targets`
- `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
- `cargo run -p ucf-ops -- spec snapshot ...` and `diff -u docs/spec_snapshot.md ./out/spec_snapshot_nightly.md`
- `cargo run -p ucf-ops -- goldens verify --all --os linux --report-out ./out/goldens_report.json`
- `cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json`
- `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate_nightly --workspace-test-report ./out/workspace_test_report.json`
- `cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json`
- `cargo run -p ucf-ops -- drift report ... --out ./out/drift_report.json`
- `cargo run -p ucf-ops -- nightly summarize --out ./out/nightly_summary.json`

Windows nightly executes a bounded subset (tests + docs lint + one golden).

## How to interpret `nightly_summary.json`

`ucf-ops nightly summarize` emits:
- overall status (`PASS` / `FAIL`),
- failed components,
- deterministic component ordering,
- triage hints (copy/paste remediation commands),
- `golden_refresh_suggested`.

`golden_refresh_suggested=true` only when all failing golden scenarios are conservative refresh candidates.

## Golden refresh candidate heuristics

A golden verify failure is marked as candidate refresh **only** when:
- fixed-point scalar summaries are unchanged,
- sampled digest structure (`tick`, `window`, count) is unchanged,
- mismatch is restricted to digest-prefix style fields.

If scalar or structure changes are detected, it is treated as regression (`refresh_candidate=false`).

The pipeline never auto-updates golden fixtures. It only reports suggestions.

## Failure triage

1. Inspect `out/nightly_summary.json` first.
2. Run listed remediation command(s) locally.
3. For golden candidates, perform explicit review before updating:
   - `cargo run -p ucf-ops -- goldens update --scenario <id> --os linux`
4. Re-run nightly-equivalent local steps and verify clean status.

## Run nightly steps locally (manual)

```bash
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out ./out/spec_snapshot_nightly.md
diff -u docs/spec_snapshot.md ./out/spec_snapshot_nightly.md
cargo run -p ucf-ops -- goldens verify --all --os linux --report-out ./out/goldens_report.json
cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate_nightly_local --workspace-test-report ./out/workspace_test_report.json
cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json
run_id=$(jq -r '.run_id' ./out/gate_report.json)
cargo run -p ucf-ops -- drift report --run "$run_id" --windows 4 --out ./out/drift_report.json
cargo run -p ucf-ops -- nightly summarize --docs ./out/docs_lint_report.json --gate ./out/gate_report.json --adversarial ./out/adversarial_report.json --goldens ./out/goldens_report.json --drift ./out/drift_report.json --out ./out/nightly_summary.json
```
