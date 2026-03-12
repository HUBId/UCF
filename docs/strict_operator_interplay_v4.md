# Strict/Operator Interplay v4

This change hardens consistency between strict mode, consolidated operator report, and operator signoff.

## Unified strict evidence surface

Both `operator report` and `operator signoff` now consume `StrictEvidenceSnapshotV1` as the single strict evidence input surface.

Snapshot fields:
- `schema_version`
- `strict_mode_enabled`
- `strict_status` (`PASS|FAIL|MISSING|SKIP`)
- `strict_report_digest_prefix`
- `policy_graph_digest_prefix`
- `manifest_digest_prefix`
- `supported_slot_set_digest_prefix`
- `primary_denial_code`
- `remediation_codes` (bounded)
- `failing_check_ids` (bounded)
- `snapshot_digest`

## Consistency guarantee

For the same strict input:
- operator report strict section and operator signoff use the same primary reason family
- remediation codes come from the same strict mapping path
- missing strict evidence is explicit (`MISSING`) and fail-closed when required

No runtime control semantics were changed; this is read-only/report consistency hardening.

## Commands

```bash
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json
cargo run -p ucf-ops -- strict explain --out ./out/strict_explain.json
```

To verify agreement, compare:
- `strict_explain.json.snapshot.primary_denial_code`
- `operator_report.json.sections.strict_section.primary_denial_code`
- `operator_signoff.json.reasons[0]` (when strict blocks)
