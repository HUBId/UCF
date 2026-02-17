# v0 Sign-Off Checklist (Taggable)

Run every command offline from repo root. Replace `<run_id>` with a unique directory (example: `v0-local-001`).

1. [ ] `cargo test --workspace`  
   Purpose: workspace verification baseline.  
   Artifact: none.  
   Pass criteria: exit code `0`.
2. [ ] `cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/<run_id>`  
   Purpose: deterministic bringup and metadata generation.  
   Artifact: `./out/<run_id>/run_metadata.json`.  
   Pass criteria: artifact exists; digest prefixes are stable per repeated run.
3. [ ] `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/<run_id>/gate_report.json`  
   Purpose: readiness gate.  
   Artifact: `gate_report.json`.  
   Pass criteria: report `status=pass`.
4. [ ] `cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/<run_id>/adversarial_report.json`  
   Purpose: adversarial regression.  
   Artifact: `adversarial_report.json`.  
   Pass criteria: `pass=true`.
5. [ ] `cargo run -p ucf-ops -- bench --scenario fixtures/e2e_scenario_a.json --ticks 256 --out ./out/<run_id>/bench_report.json`  
   Purpose: deterministic performance envelope.  
   Artifact: `bench_report.json`.  
   Pass criteria: no memory cap violation (`memory.cap_exceeded=false`).
6. [ ] `cargo run -p ucf-ops -- models verify --manifest models/manifest.toml`  
   Purpose: model slot hash-lock verification.  
   Artifact: none.  
   Pass criteria: every slot verified or explicitly disabled.
7. [ ] `cargo run -p ucf-ops -- models probe --manifest models/manifest.toml --out ./out/<run_id>/probe_report.json` *(optional)*  
   Purpose: deterministic probe run for present model files.  
   Artifact: `probe_report.json`.  
   Pass criteria: `summary.pass=true` when real/fixture weights are present.
8. [ ] `cargo run -p ucf-ops -- ess snapshot --out ./out/<run_id>/snapshot.snap`  
   Purpose: reproducible ESS snapshot.  
   Artifact: `snapshot.snap`.  
   Pass criteria: snapshot digest + manifest digest emitted.
9. [ ] `cargo run -p ucf-ops -- security verify-chain --from 1 --to 32`  
   Purpose: audit/capability chain integrity check.  
   Artifact: none.  
   Pass criteria: exit code `0`.
10. [ ] `cargo run -p ucf-ops -- out manifest --dir ./out/<run_id>`  
    Purpose: emit artifact hash manifest.  
    Artifact: `manifest.json`.  
    Pass criteria: manifest contains all generated outputs + sha256.
11. [ ] `cargo run -p ucf-ops -- release signoff --validate --out ./out/<run_id> --emit release/v0_signoff_result.json`  
    Purpose: machine-validated tag gate.  
    Artifact: `release/v0_signoff_result.json`.  
    Pass criteria: `pass=true`.

## Tag gate
After all checks pass, create a release tag against the exact commit/hash set represented by `run_metadata.json`, `manifest.json`, and `release/v0_signoff_result.json`.
