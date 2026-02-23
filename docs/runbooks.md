# Runbooks Index

- Bringup: `docs/runbooks/bringup.md`
- Diagnostics: `docs/runbooks/diagnostics.md`
- Diagnostics bundle: `docs/runbooks/diagnostics_bundle.md`
- Failure modes: `docs/runbooks/failure_modes.md`
- Incident playbook: `docs/runbooks/incident_playbook.md`

## v1.0-rc1 Sign-Off + Tagging
1. Run rc1 gate:
   - `cargo run -p ucf-ops -- release rc1-gate --out ./out/rc1_gate.json --load-smoke`
2. Run load and soak suites:
   - `scripts/load_rc1.sh`
   - `SOAK_MINUTES=30 scripts/soak_rc1.sh`
3. Validate machine checklist:
   - `cargo run -p ucf-ops -- release signoff --validate --checklist release/v1_rc1_signoff_checklist.toml --out ./out/rc1 --emit release/v1_rc1_signoff_result.json`
4. Store `./out/rc1*`, checklist result, and manifests.
5. Tag only after PASS artifacts are archived: `git tag -a v1.0-rc1 <commit>`.


## Strict mode audits
- Determinism: `cargo run -p ucf-ops -- determinism scan`
- Hidden paths: `cargo run -p ucf-ops -- audit scan`
- In CI both scans are blocking and must return zero violations.
