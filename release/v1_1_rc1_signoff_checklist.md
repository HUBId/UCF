# v1.1-rc1 Sign-Off Checklist

1. [ ] `cargo fmt --all -- --check`
2. [ ] `cargo clippy --workspace --all-targets -- -D warnings`
3. [ ] `cargo test --workspace --all-targets`
4. [ ] Stage model candidates (`ucf-ops models stage`) for changed slots.
5. [ ] Run probes (`ucf-ops models probe --out ./out/v1_1/probe_report.json`).
6. [ ] Produce world shadow evidence (`ucf-ops world shadow-report --run <id> --windows <n> --out ./out/world_shadow_report.json`) when WorldVljepa is active/promoted.
7. [ ] Promote models with provenance (`ucf-ops models promote ...`).
8. [ ] Run readiness gate (`cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/v1_1/readiness_gate.json --workdir ./.ucf_gate_v1_1`).
9. [ ] Run adversarial suite (`cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/v1_1/adversarial_report.json`).
10. [ ] Run bench (`cargo run -p ucf-ops -- bench --ticks 2048 --out ./out/v1_1/bench_report.json`) with opt kernels / GPU mode if enabled.
11. [ ] Data governance snapshot + compaction check completed.
12. [ ] Policy graph digest pinned from `policy validate` output.
13. [ ] Validate sign-off (`cargo run -p ucf-ops -- release signoff --validate --checklist release/v1_1_rc1_signoff_checklist.toml --out ./out/v1_1 --emit release/v1_1_rc1_signoff_result.json`).
