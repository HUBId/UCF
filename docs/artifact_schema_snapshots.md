# Artifact Schema Snapshots

This repository enforces deterministic shape snapshots for governance/review/export/interop artifacts that must remain stable across v3/v4/v5/v6/v7/v8 hardening.

## Covered artifacts

Snapshots are generated under `docs/artifact_schema_snapshots/`:

- `active_review_snapshot_v1.json`
- `applied_scope_authority_v1.json` (v7)
- `applied_supported_set_context_v1.json`
- `backend_evidence_snapshot_v1.json`
- `backend_resolution_v1.json`
- `bugkit_manifest_v1.json`
- `bundle_roundtrip_consistency_v1.json` (v7/v8 additive drift tracked)
- `canonical_bundle_consumption_context_v1.json` (v7)
- `canonical_bundle_spine_v1.json` (v8)
- `canonical_export_artifact_ref_v1.json`
- `canonical_export_context_v1.json`
- `canonical_governance_entry_v1.json` (v8)
- `canonical_readiness_spine_v1.json` (v8)
- `cross_surface_condition_observation_v1.json` (v7)
- `cross_surface_context_matrix_v1.json` (v6/v8 additive drift tracked)
- `governance_primary_surfaces_v1.json`
- `interop_consistency_matrix_report_v1.json`
- `operator_report_v1.json`
- `operator_review_packet_v1.json` (v7/v8 additive shape drift tracked)
- `operator_signoff_v1.json` (v7/v8 additive shape drift tracked)
- `operator_workflow_chain_v1.json` (v8 additive shape drift tracked)
- `readiness_gate_report_v1.json`
- `remediation_consistency_check_v1.json`
- `repro_pack_manifest_v1.json`
- `reviewability_reduction_v1.json` (v7)
- `slot_reviewability_truth_v1.json` (v7)
- `spine_condition_observation_v1.json` (v8)
- `strict_failure_report_v3.json`
- `supported_real_slot_set_v2.json`
- `supported_scope_execution_v3.json` (v8)
- `supported_scope_reevaluation_v1.json` (v7)
- `v3_gate_report_v1.json`
- `v4_gate_report_v1.json`
- `v5_gate_report_v1.json`
- `v7_gate_report_v1.json` (v8 lane coverage)
- `index.json` (covered artifact index)

v8 contract points now frozen here include canonical governance entry, supported scope execution, canonical readiness spine, canonical bundle spine, and spine-level condition observation; related additive updates in active-review/signoff/review-packet/workflow and interop/roundtrip schemas are also tracked.

## Regeneration

```bash
cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots
```

The generator is deterministic:

- fixed artifact emission order (sorted by artifact id)
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
3. run the check command,
4. commit snapshots together with rationale/docs updates.

For governance/scope/readiness/bundle/interop artifacts, snapshots freeze contract shape only (fields, optionality, and enum variants), not runtime values. This explicitly guards shared cross-surface contracts from silent schema drift.
