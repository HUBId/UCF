# Artifact Schema Snapshots

This repository enforces deterministic shape snapshots for governance/review/export/interop artifacts that must remain stable across v3/v4/v5/v6/v7 hardening.

## Covered artifacts

Snapshots are generated under `docs/artifact_schema_snapshots/`:

- `active_review_snapshot_v1.json`
- `backend_resolution_v1.json`
- `backend_evidence_snapshot_v1.json`
- `governance_primary_surfaces_v1.json`
- `supported_real_slot_set_v2.json`
- `applied_scope_authority_v1.json` (v7)
- `applied_supported_set_context_v1.json`
- `bundle_roundtrip_consistency_v1.json` (v7)
- `repro_pack_manifest_v1.json`
- `bugkit_manifest_v1.json`
- `canonical_bundle_consumption_context_v1.json` (v7)
- `canonical_export_artifact_ref_v1.json`
- `canonical_export_context_v1.json`
- `remediation_consistency_check_v1.json`
- `cross_surface_context_matrix_v1.json`
- `cross_surface_condition_observation_v1.json` (v7)
- `interop_consistency_matrix_report_v1.json`
- `operator_report_v1.json`
- `operator_signoff_v1.json` (v7 additive shape drift tracked here)
- `operator_review_packet_v1.json` (v7 additive shape drift tracked here)
- `strict_failure_report_v3.json`
- `v3_gate_report_v1.json`
- `v4_gate_report_v1.json`
- `v5_gate_report_v1.json`
- `readiness_gate_report_v1.json`
- `reviewability_reduction_v1.json` (v7)
- `slot_reviewability_truth_v1.json` (v7)
- `supported_scope_reevaluation_v1.json` (v7)
- `index.json` (covered artifact index)

These v7 artifacts are treated as frozen cross-surface contract points: applied-scope authority, supported-scope reevaluation, per-slot reviewability truth/reduction, canonical bundle consumption/roundtrip consistency, and remediation interop observations.

## Regeneration

```bash
cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots
```

The generator is deterministic:

- fixed artifact emission order
- stable sorted field/type maps
- stable sorted enum variant lists
- no runtime timestamps or nondeterministic runtime values

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

For governance/review/export/interop artifacts, snapshots freeze contract shape only (fields, optionality, and enum variants), not runtime values. This explicitly guards shared cross-surface contracts from silent schema drift.
