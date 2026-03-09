# v2 Signoff Gate (Minimal Real-Backend Scaffolding)

## Purpose
`ucf-ops v2 gate` is the bounded signoff gate for the **v2 minimal phase**.

PASS means the repository proves, in offline and hardware-neutral mode, that:
- minimal real-backend scaffolding is wired for the supported slots,
- tiny real fixtures probe successfully,
- shadow runs have no decision impact,
- shadow-ready evidence is present,
- drift budget and alerts rules are present,
- strict checks relevant to v2 pass.

## Explicit non-goals
PASS does **not** certify that:
- real backends are approved for Active mode,
- large models are supported,
- GPU or remote/online compute is validated.

## Command
```bash
cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json
```

## Exit codes
- `0` — overall PASS
- `2` — overall FAIL

## Report schema
`V2GateReportV1`
- `schema_version`
- `overall_status` (`PASS` | `FAIL`)
- `checks` (fixed order)

Per check:
- `name`
- `status` (`PASS` | `FAIL` | `SKIP`)
- `evidence_digest_prefixes`
- `remediation_hint_code`
- `notes`

## Fixed-order checks
1. `v0_gate_pass`
2. `v1_gate_pass`
3. `models_manifest_verify`
4. `world_tiny_fixture_probe_pass`
5. `second_slot_tiny_fixture_probe_pass`
6. `world_shadow_no_impact`
7. `second_slot_shadow_no_impact`
8. `world_shadow_ready`
9. `second_slot_shadow_ready`
10. `drift_budget_present`
11. `alerts_rules_present`
12. `strict_check_v2`
13. `world_parity_report_present`
14. `burn_world_probe_pass` (optional, `SKIP` when backend-burn is not enabled)
15. `burn_world_shadow_compare_present` (optional, `SKIP` when backend-burn is not enabled)

## PASS/FAIL/SKIP interpretation
- `PASS`: check verified.
- `FAIL`: required evidence missing, invalid, or failed.
- `SKIP`: allowed only for explicitly optional burn-path checks.

## Supported-slot scope for this phase
v2 minimal signoff scope is **exactly two slots**:
- `world_jepa`
- and exactly one second slot declared in `docs/series_state_snapshot.md` (`sae` or `ssm`).
