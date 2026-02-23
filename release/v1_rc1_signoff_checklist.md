# v1.0-rc1 Sign-Off Checklist

1. [ ] `cargo fmt --all -- --check`
2. [ ] `cargo clippy --workspace --all-targets -- -D warnings`
3. [ ] `cargo test --workspace --all-targets`
4. [ ] `cargo run -p ucf-ops -- determinism scan`
5. [ ] `cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1`
6. [ ] `cargo run -p ucf-ops -- models verify --manifest models/manifest.toml`
7. [ ] `cargo run -p ucf-ops -- models probe --manifest models/manifest.toml --out ./out/rc1/probe_report.json`
8. [ ] `cargo run -p ucf-ops -- release rc1-gate --out ./out/rc1_gate.json --load-smoke`
9. [ ] `scripts/load_rc1.sh ./out/rc1`
10. [ ] `SOAK_MINUTES=30 scripts/soak_rc1.sh ./out/rc1`
11. [ ] `cargo run -p ucf-ops -- diagnostics collect --run <run_id> --out ./out/diag_<run_id>.zip`
12. [ ] `cargo run -p ucf-ops -- release signoff --validate --checklist release/v1_rc1_signoff_checklist.toml --out ./out/rc1 --emit release/v1_rc1_signoff_result.json`

Thresholds:
- fallbacks <= 5 per 1000 ticks
- p95 <= 50ms for critical stages
- memory <= 3GB on target
