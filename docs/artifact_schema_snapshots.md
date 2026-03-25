# Artifact Schema Snapshots

This repository enforces deterministic shape snapshots for governance/review/export/interop artifacts that must remain stable across v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13 hardening.

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
- `canonical_governance_entry_authority_v2.json` (v9 final authority freeze)
- `canonical_bundle_authority_v2.json` (v9 final authority freeze)
- `canonical_primary_semantics_authority_v1.json` (v9 final authority freeze)
- `canonical_readiness_authority_v2.json` (v9 final authority freeze)
- `canonical_readiness_spine_v1.json` (v8)
- `canonical_roundtrip_chain_v1.json` (v9)
- `canonical_continuity_authority_v1.json` (v9 continuity authority freeze)
- `cross_surface_condition_observation_v1.json` (v7)
- `cross_surface_context_matrix_v1.json` (v6/v8 additive drift tracked)
- `final_bundle_consumer_authority_v1.json` (v10 final consumer-authority freeze)
- `final_bundle_residual_sweep_v1.json` (v11 final residual-sweep freeze)
- `final_continuity_authority_v2.json` (v10 sole top-level continuity authority)
- `final_governance_consumer_authority_v1.json` (v10 final consumer-authority freeze)
- `final_governance_residual_sweep_v1.json` (v11 final residual-sweep freeze)
- `final_primary_semantics_consumer_authority_v1.json` (v10 final consumer-authority freeze)
- `final_primary_semantics_residual_sweep_v1.json` (v11 final residual-sweep freeze)
- `final_readiness_consumer_authority_v1.json` (v10 final consumer-authority freeze)
- `final_readiness_residual_sweep_v1.json` (v11 final residual-sweep freeze)
- `governance_primary_surfaces_v1.json`
- `interop_consistency_matrix_report_v1.json`
- `operator_report_v1.json`
- `operator_review_packet_v1.json` (v7/v8 additive shape drift tracked)
- `operator_signoff_v1.json` (v7/v8 additive shape drift tracked)
- `operator_workflow_chain_v1.json` (v8 additive shape drift tracked)
- `readiness_gate_report_v1.json`
- `remediation_consistency_check_v1.json`
- `repro_pack_manifest_v1.json`
- `residual_free_bundle_absolute_sweep_v1.json` (v13 absolute residual-free contract)
- `residual_free_bundle_consumer_authority_v1.json` (v12 residual-free consumer authority)
- `residual_free_continuity_authority_v1.json`
- `residual_free_governance_absolute_sweep_v1.json` (v13 absolute residual-free contract)
- `residual_free_governance_consumer_authority_v1.json` (v12 residual-free consumer authority)
- `residual_free_primary_semantics_absolute_sweep_v1.json` (v13 absolute residual-free contract)
- `residual_free_primary_semantics_consumer_authority_v1.json` (v12 residual-free consumer authority)
- `residual_free_readiness_absolute_sweep_v1.json` (v13 absolute residual-free contract)
- `residual_free_readiness_consumer_authority_v1.json` (v12 residual-free consumer authority)
- `reviewability_reduction_v1.json` (v7)
- `slot_reviewability_truth_v1.json` (v7)
- `spine_condition_observation_v1.json` (v8)
- `strict_failure_report_v3.json`
- `supported_real_slot_set_v2.json`
- `supported_scope_execution_v3.json` (v8)
- `supported_scope_execution_v4.json` (v9 canonical supported-scope execution)
- `supported_scope_execution_v5.json` (v10 final supported-scope execution)
- `supported_scope_execution_v6.json` (v11 supported-scope execution freeze)
- `supported_scope_execution_v7.json` (v12 residual-free supported-scope execution)
- `supported_scope_execution_v8.json` (v13 absolute residual-free supported-scope execution)
- `supported_scope_reevaluation_v1.json` (v7)
- `v3_gate_report_v1.json`
- `v4_gate_report_v1.json`
- `v5_gate_report_v1.json`
- `v7_gate_report_v1.json` (v8 lane coverage)
- `v8_gate_report_v1.json`
- `v9_gate_report_v1.json`
- `v10_gate_report_v1.json` (v10 consolidated gate)
- `index.json` (covered artifact index)

v11 contract points now frozen here include:

- `canonical_governance_entry_authority_v2`
- `supported_scope_execution_v6`
- `canonical_readiness_authority_v2`
- `canonical_bundle_authority_v2`
- `canonical_primary_semantics_authority_v1`
- `final_governance_consumer_authority_v1`
- `final_readiness_consumer_authority_v1`
- `final_bundle_consumer_authority_v1`
- `final_primary_semantics_consumer_authority_v1`
- `final_governance_residual_sweep_v1`
- `final_readiness_residual_sweep_v1`
- `final_bundle_residual_sweep_v1`
- `final_primary_semantics_residual_sweep_v1`

Related additive updates in canonical governance/readiness/bundle/primary-semantics authority families, operator signoff/review/workflow, and v9/v10/v11-adjacent gate/interop families remain tracked through the same snapshot lane.

v12 contract points now frozen here include:

- `residual_free_governance_consumer_authority_v1`
- `supported_scope_execution_v7`
- `residual_free_readiness_consumer_authority_v1`
- `residual_free_bundle_consumer_authority_v1`
- `residual_free_primary_semantics_consumer_authority_v1`
- additive updates to `final_governance_residual_sweep_v1`
- additive updates to `final_readiness_residual_sweep_v1`
- additive updates to `final_bundle_residual_sweep_v1`
- additive updates to `final_primary_semantics_residual_sweep_v1`
- additive updates to `operator_signoff_v1`, `operator_review_packet_v1`, and `operator_workflow_chain_v1`
- additive updates to consolidated gate report families covered by the same lane

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

For governance/scope/readiness/bundle/primary-semantics/interop artifacts, snapshots freeze contract shape only (fields, optionality, and enum variants), not runtime values. This explicitly guards shared cross-surface contracts from silent schema drift.

For v12 residual-free final-input surfaces, this freeze is now first-class for cross-surface contracts and CI review:

- residual-free governance consumer authority,
- supported-scope execution v7,
- residual-free readiness consumer authority,
- residual-free bundle consumer authority,
- residual-free primary-semantics consumer authority.

For v13 absolute residual-free surfaces, this freeze is now first-class for cross-surface contracts and CI review:

- residual-free governance absolute sweep,
- supported-scope execution v8,
- residual-free readiness absolute sweep,
- residual-free bundle absolute sweep,
- residual-free primary-semantics absolute sweep.
