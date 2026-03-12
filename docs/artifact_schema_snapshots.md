# Artifact Schema Snapshots

This repository enforces deterministic shape snapshots for governance/export artifacts that must remain stable across v3/v4/v5 hardening.

## Covered artifacts

Snapshots are generated under `docs/artifact_schema_snapshots/`:

- `active_review_snapshot_v1.json`
- `backend_resolution_v1.json`
- `backend_evidence_snapshot_v1.json`
- `repro_pack_manifest_v1.json`
- `bugkit_manifest_v1.json`
- `remediation_consistency_check_v1.json`
- `operator_report_v1.json`
- `operator_signoff_v1.json`
- `strict_failure_report_v3.json`
- `v3_gate_report_v1.json`
- `v4_gate_report_v1.json`
- `readiness_gate_report_v1.json`
- `index.json` (covered artifact index)

## Regeneration

```bash
cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots
```

The generator is deterministic:

- fixed artifact emission order
- stable sorted field/type maps
- stable sorted enum variant lists
- no runtime timestamps

## Drift check

```bash
cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
```

`drift_kind` is conservative:

- `ADDITIVE`: optional field additions or enum variant additions
- `BREAKING`: field removal, required-field regression, or incompatible type change
- `UNKNOWN`: any unclassified drift (treated as non-additive for review)

## CI/docs-lint enforcement

`ucf-ops docs lint --strict` runs artifact snapshot drift detection and fails when snapshots differ from regenerated output.

When a schema change is intentional:

1. regenerate snapshots,
2. review the diff,
3. commit snapshots together with rationale/docs updates.

For v5 export/report artifacts, snapshots freeze manifest/report *shape only* (fields, optionality, and enum variants), not runtime values or file contents. This explicitly guards enriched export evidence references (`backend_evidence_snapshot`, `active_review_snapshot`, `operator_signoff`, `backend_resolution`), digest/link fields, and bounded reason/remediation metadata from silent schema drift.
