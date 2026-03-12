# v4 Signoff Gate (Consistency/Governance)

## Purpose

`ucf-ops v4 gate` is the final v4 consistency/signoff hardening gate.
It validates that governance and evidence surfaces remain coherent for the bounded supported real-slot scope:

- `world_jepa`
- plus exactly one configured second slot (`sae` or `ssm`) from `docs/series_state_snapshot.md`

The gate is intentionally offline, deterministic, and hardware-neutral.
It is not a runtime capability expansion gate.

## Command

```bash
cargo run -p ucf-ops -- v4 gate --out ./out/v4_gate_report.json
```

## Exit codes

- `0`: overall `PASS`
- `2`: overall `FAIL`

## Report schema

The command emits `V4GateReportV1`:

- `schema_version`
- `overall_status` (`PASS | FAIL`)
- `checks` (fixed ordering)

Each check carries:

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
5. `supported_slot_set_consistent`
6. `backend_evidence_snapshot_present`
7. `backend_evidence_snapshot_schema_stable`
8. `operator_report_present`
9. `operator_signoff_present`
10. `operator_signoff_consistent_with_evidence`
11. `remediation_registry_present`
12. `remediation_registry_consistent_across_reports`
13. `strict_evidence_present`
14. `strict_operator_alignment_ok`
15. `artifact_schema_snapshot_checks_pass`
16. `portability_docs_checks_pass`

## Optional checks

17. `optional_backend_states_consistent`
18. `burn_parity_optional_path_consistent`

`burn_parity_optional_path_consistent` is `SKIP` when optional parity path is not configured/present.

## PASS / FAIL / SKIP interpretation

- **PASS (overall):** All required surfaces are coherent and every check is `PASS` or `SKIP`.
- **FAIL (overall):** Any required consistency surface is missing/stale/mismatched.
- **SKIP (check-level only):** Optional unsupported/unconfigured path.

Normalization is fail-closed for required evidence:

- missing required artifacts/reports -> `FAIL`
- stale artifact schema snapshots -> `FAIL`
- remediation registry mismatch across reports -> `FAIL`
- strict/operator/signoff mismatch -> `FAIL`

## What PASS certifies

At v4, `PASS` certifies **consistency only**:

- supported real-slot set remains coherent across consumers
- backend evidence snapshot exists and remains schema-stable
- operator report/signoff are derivable and mutually consistent with strict/evidence
- remediation registry use is unified across sampled reports
- portability/docs/schema checks are current

## What PASS does NOT certify

A v4 `PASS` does **not** imply:

- broader runtime capability
- additional slots/backends becoming production-ready
- automatic active rollout approval
- GPU/remote compute/training readiness

v4 remains a conservative consistency/signoff hardening phase.

## Post-v4 continuation note

After `ucf-ops v4 gate` reports PASS, continue execution at **Prompt 220** via `docs/next_10_prompts.md`.
