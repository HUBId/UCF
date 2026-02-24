# Model Incident Runbook

## Immediate containment

1. Pin to known-good promoted hash:
   - `export UCF_MODEL_PIN_<SLOT>=<sha256>`
2. Restart runtime and verify model provenance records.

## Rollback

1. Execute rollback to known-good promoted hash:
   - `cargo run -p ucf-ops -- models rollback --slot <slot> --to <sha256>`
2. Validate active hash:
   - `cargo run -p ucf-ops -- models list --slot <slot>`

## Diagnostics bundle

1. Collect:
   - `cargo run -p ucf-ops -- diagnostics collect --out ./out/incident_bundle`
2. Attach:
   - `./out/probe_report.json`
   - `./out/gate_report.json`
   - `models/MANIFEST.toml`
