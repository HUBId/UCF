# Incident Playbook

1. Inspect logs in `.ucf/logs/` and recent out artifacts.
2. Run diagnostics:
   ```bash
   cargo run -p ucf-ops -- diag
   ```
3. Run readiness and policy sanity checks:
   ```bash
   cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/incident_gate.json
   cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1
   ```
4. Collect diagnostics bundle (redaction-safe):
   ```bash
   cargo run -p ucf-ops -- diagnostics collect --run <run_id> --out ./out/diag_<run_id>.zip
   ```
5. Replay bug report if needed:
   ```bash
   cargo run -p ucf-ops -- export-bugreport --last 50
   cargo run -p ucf-ops -- replay-bugreport ./.ucf/reports/bugreport_<timestamp> --mode compute
   ```
6. Attach `diag_<run_id>.zip` + replay report + readiness report to incident ticket.
