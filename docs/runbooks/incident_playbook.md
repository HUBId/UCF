# Incident Playbook

1. Inspect logs in `.ucf/logs/`.
2. Run diagnostics:
   ```bash
   cargo run -p ucf-ops -- diag
   ```
3. Verify audit/ESS health from diagnostics output.
4. Export bounded bug report slice:
   ```bash
   cargo run -p ucf-ops -- export-bugreport --last 50
   cargo run -p ucf-ops -- verify-bugreport ./.ucf/reports/bugreport_<timestamp>
   ```
5. Replay in no-action mode:
   ```bash
   cargo run -p ucf-ops -- replay-bugreport ./.ucf/reports/bugreport_<timestamp> --mode compute
   ```
6. Attach bugreport directory and replay report to incident ticket.
