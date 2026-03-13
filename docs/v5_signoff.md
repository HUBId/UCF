# v5 Signoff Gate (Governance/Export/Review Hardening)

## Purpose

`ucf-ops v5 gate` is the v5 closure gate for governance and consistency hardening.
It validates coherent, deterministic review/export surfaces for the bounded supported real-slot scope:

- `world_jepa`
- plus exactly one configured second slot (`sae` or `ssm`) from `docs/series_state_snapshot.md`

The gate remains offline, hardware-neutral, deterministic, and conservative.
It is not a runtime capability expansion gate.

## Command

```bash
cargo run -p ucf-ops -- v5 gate --out ./out/v5_gate_report.json
```

## Exit codes

- `0`: overall `PASS`
- `2`: overall `FAIL`

## Report schema

The command emits `V5GateReportV1`:

- `schema_version`
- `overall_status` (`PASS | FAIL`)
- `checks` (fixed ordering)

Each check contains:

- `name`
- `status` (`PASS | FAIL | SKIP`)
- `evidence_digest_prefixes` (bounded map)
- `remediation_hint_code`
- `notes` (bounded code)

## Required checks

1. `v0_gate_pass`
2. `v1_gate_pass`
3. `v2_gate_pass`
4. `v3_gate_pass`
5. `v4_gate_pass`
6. `supported_set_review_present`
7. `supported_set_review_consistent`
8. `active_review_snapshot_present`
9. `active_review_snapshot_consistent`
10. `backend_resolution_present`
11. `backend_resolution_consistent`
12. `enriched_repro_export_smoke_pass`
13. `enriched_bugkit_export_smoke_pass`
14. `remediation_consistency_pass`
15. `operator_review_packet_present`
16. `operator_review_packet_consistent`
17. `artifact_schema_snapshot_checks_pass`
18. `portability_docs_checks_pass`

## Optional checks

19. `optional_backend_resolution_consistent`
20. `chosen_slot_burn_optional_path_consistent`

`chosen_slot_burn_optional_path_consistent` is `SKIP` when the chosen-slot Burn path is formally unsupported/unconfigured.

## PASS / FAIL / SKIP interpretation

- **PASS (overall):** all required governance/export/review consistency surfaces are coherent; checks are `PASS` or optional `SKIP`.
- **FAIL (overall):** at least one required evidence/consistency surface is missing, stale, or inconsistent.
- **SKIP (check-level only):** optional unsupported/unconfigured path.

Fail-closed normalization:

- missing required report/artifact => `FAIL`
- stale artifact schema snapshot => `FAIL`
- remediation consistency drift => `FAIL`
- review packet disagreement with active-review/signoff/gates => `FAIL`
- optional unsupported Burn path => `SKIP`

## What PASS certifies

At v5, `PASS` certifies:

- supported-set review is explicit and consistent
- active-review snapshot exists and aligns with current governance evidence
- backend resolution for the chosen second slot is explicit and consistent
- enriched repro/bugkit export smokes verify expected evidence context behavior
- remediation consistency remains stable
- operator review packet is derivable and aligned with signoff/evidence/gates
- artifact schema and portability/docs hygiene checks pass

## What PASS does NOT certify

A v5 `PASS` does **not** certify:

- broader runtime capability
- additional slots/backends becoming production-ready
- automatic activation of any slot/backend
- GPU, remote compute, training, or large-model readiness

v5 is strictly governance/export/review hardening.

## Post-v5 continuation note

After v5 gate PASS, continue at Prompt 230 via `docs/next_10_prompts.md`.
