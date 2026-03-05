# Diagnostics Bundle (rc1)

Collect a replayable, redaction-safe archive:

```bash
cargo run -p ucf-ops -- diagnostics collect --run <run_id> --out ./out/diag_<run_id>.zip
```

Bundle content (best effort):
- `run_metadata.json`
- `metrics_summary.json`
- `gate_report.json`
- `adversarial_report.json`
- `bench_report.json`
- explain-tick snapshots from `.ucf/explain_tick/`

## Redaction policy
- No raw output payloads are included by default.
- Keys named `text` / `payload` are rewritten as `text_redacted` / `payload_redacted` during collection.
- Artifact is suitable for replay and incident escalation.

## Backtrace handling

- Default deployment posture: set `RUST_BACKTRACE=0`.
- Diagnostics collection supports explicit opt-in:
  - `ucf-ops diagnostics collect --run <id> --out <zip> --include_backtrace`
- Backtrace/path material is path-redacted before writing into the diagnostics archive.
- Backtrace content is not emitted into normal gateway/runtime safe error responses.
