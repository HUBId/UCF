# Runbooks Index

- Bringup: `docs/runbooks/bringup.md`
- Diagnostics: `docs/runbooks/diagnostics.md`
- Failure modes: `docs/runbooks/failure_modes.md`
- Incident playbook: `docs/runbooks/incident_playbook.md`

## v0 Sign-Off + Tagging
1. Execute the checklist sequence in `release/v0_signoff_checklist.md`.
2. Validate artifacts with:
   - `cargo run -p ucf-ops -- out manifest --dir ./out/<run_id>`
   - `cargo run -p ucf-ops -- release signoff --validate --out ./out/<run_id> --emit release/v0_signoff_result.json`
3. Confirm `release/v0_signoff_result.json` reports `pass=true`.
4. Tag the commit only after the signoff result and manifest are committed or archived.
