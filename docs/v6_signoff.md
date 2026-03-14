# v6 Signoff Gate (Governance / Interop / Workflow Hardening)

## Purpose

`ucf-ops v6 gate` is the v6 closure gate for governance, export normalization, interop consistency, and operator workflow chain alignment.

The gate is explicitly:

- offline-first
- hardware-neutral
- deterministic
- fail-closed

It validates consistency over the **applied** supported scope (`SupportedRealSlotSetV2` + `AppliedSupportedSetContextV1`) and does not expand runtime capability.

## Command

```bash
cargo run -p ucf-ops -- v6 gate --out ./out/v6_gate_report.json
```

## Exit codes

- `0`: overall `PASS`
- `2`: overall `FAIL`

## Report schema

The command emits `V6GateReportV1`:

- `schema_version`
- `overall_status` (`PASS | FAIL`)
- `checks` (fixed ordering)

Each check includes:

- `name`
- `status` (`PASS | FAIL | SKIP`)
- `evidence_digest_prefixes` (bounded map)
- `remediation_hint_code`
- `notes` (bounded code)

### Fixed check order

1. `v0_gate_pass`
2. `v1_gate_pass`
3. `v2_gate_pass`
4. `v3_gate_pass`
5. `v4_gate_pass`
6. `v5_gate_pass`
7. `governance_primary_surfaces_pass`
8. `applied_supported_scope_present`
9. `applied_supported_scope_consistent`
10. `export_normalization_pass`
11. `interop_consistency_pass`
12. `operator_workflow_chain_present`
13. `operator_workflow_chain_consistent`
14. `artifact_schema_snapshot_checks_pass`
15. `portability_docs_checks_pass`
16. `optional_backend_path_consistent`
17. `legacy_artifact_translation_ok`

`legacy_artifact_translation_ok` is `SKIP` when no legacy translation path is present.

## PASS / FAIL / SKIP interpretation

- **PASS (overall):** all required checks are `PASS`; optional checks are `PASS` or `SKIP`.
- **FAIL (overall):** at least one required surface is missing or inconsistent.
- **SKIP (check-level):** optional unsupported/unconfigured path only.

## What PASS certifies

A v6 PASS certifies that:

- applied supported scope is authoritative and consumed consistently
- governance primary surfaces validate
- export normalization is canonical
- interop consistency matrix passes
- operator workflow chain is present and aligned with governance/scope/export/interop surfaces
- artifact schema snapshot checks pass
- portability/docs checks pass

## What PASS does NOT certify

A v6 PASS does **not** certify:

- broader runtime capability
- additional slots/backends becoming production-ready
- automatic activation of any slot/backend
- GPU, remote compute, training, or large-model readiness

v6 remains a governance/interop/workflow hardening phase.

After v6 gate PASS, continue at Prompt 240 via `docs/next_10_prompts.md`.
