# Artifact Schema Snapshots

This repository enforces deterministic shape snapshots for operator/signoff artifacts that must remain stable across v3/v4 hardening.

## Covered artifacts

Snapshots are generated under `docs/artifact_schema_snapshots/`:

- `backend_evidence_snapshot_v1.json`
- `operator_report_v1.json`
- `operator_signoff_v1.json`
- `strict_failure_report_v3.json`
- `v3_gate_report_v1.json`
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
