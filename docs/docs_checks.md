# Docs-as-Checks (CI blocking)

`ucf-ops docs lint` validates documentation artifacts that are part of the runtime specification.

## Command

```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
```

Modes:
- `--strict`: fail on all lint failures (CI default).
- `--warn`: downgrade module-map mismatch warnings to non-blocking.

## Checks

1. **Spec snapshot up-to-date**
   - Regenerates snapshot in a temp file and compares against `docs/spec_snapshot.md`.
   - Fails when committed snapshot is stale.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
     git add docs/spec_snapshot.md
     ```

2. **Policy packs validate (base + overlay)**
   - Runs policy merge/validation for `policies/packs/base_v1` + `policies/packs/overlays/test`.
   - Fails on merge errors, schema issues, or unknown keys.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
     ```

3. **Prompt index integrity**
   - Parses prompt IDs from `docs/prompt_series_index.md`.
   - Enforces unique and strictly increasing IDs.
   - Validates `PROMPT <N> —` heading format when prompt headings are present.

4. **Module map best-effort cargo metadata consistency**
   - Parses crate-like entries in `docs/module_map.md`.
   - Compares them with local `cargo metadata --no-deps --format-version 1` package names.
   - `--strict`: mismatch fails.
   - `--warn`: mismatch warns and continues.

5. **Hardware-neutral docs guardrail**
   - Scans `docs/prompt_series_index.md`, `docs/prompt_rulebook.md`, and `docs/deploy_portable.md` for obvious hardware-specific terms.
   - Fails when forbidden terms appear in core docs outside clearly marked history sections.
   - Allows deploy/history mentions as warnings.
   - Remediation:
     ```bash
     # Replace hardware/vendor wording with DeviceProfile + explicit budgets
     cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
     ```


6. **Artifact schema snapshots up-to-date**
   - Regenerates shape snapshots for covered v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/v19 governance/scope/readiness/review/export/interop artifacts and compares them with committed files in `docs/artifact_schema_snapshots/`.
   - v12 residual-free final-input contract points (`residual_free_*_consumer_authority_v1` plus `supported_scope_execution_v7`) are enforced in the same lane.
   - v13 absolute residual-free contract points (`residual_free_*_absolute_sweep_v1` plus `supported_scope_execution_v8`) are enforced in the same lane.
   - v14 terminal absolute residual-free contract points (`absolute_final_*_terminal_sweep_v1` plus `supported_scope_execution_v9`) are enforced in the same lane.
   - v15 ultimate terminal absolute residual-free contract points (`terminal_*_ultimate_sweep_v1` plus `supported_scope_execution_v10`) are enforced in the same lane.
   - v16 convergence + current scope-execution contract points (`governance_convergence_sweep_v1`, `supported_scope_execution_v11`, `readiness_convergence_sweep_v1`, `bundle_convergence_sweep_v1`, `primary_semantics_convergence_sweep_v1`) are enforced in the same lane.
   - v17 stabilization + current supported-scope execution v12 contract points (`governance_stabilization_sweep_v1`, `supported_scope_execution_v12`, `readiness_stabilization_sweep_v1`, `bundle_stabilization_sweep_v1`, `primary_semantics_stabilization_sweep_v1`) are enforced in the same lane.
   - v18 final-consolidation + current supported-scope execution v13 contract points (`governance_final_consolidation_sweep_v1`, `supported_scope_execution_v13`, `readiness_final_consolidation_sweep_v1`, `bundle_final_consolidation_sweep_v1`, `primary_semantics_final_consolidation_sweep_v1`) are enforced in the same lane.
   - v19 closure + current supported-scope execution v14 contract points (`governance_closure_sweep_v1`, `supported_scope_execution_v14`, `readiness_closure_sweep_v1`, `bundle_closure_sweep_v1`, `primary_semantics_closure_sweep_v1`) are enforced in the same lane.
   - Classifies drift conservatively as `ADDITIVE`, `BREAKING`, or `UNKNOWN`; strict lint fails on drift.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots
     cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
     git add docs/artifact_schema_snapshots
     ```

7. **Remediation registry doc up-to-date**
   - Regenerates `docs/remediation_codes_v1.md` from the canonical remediation registry and compares against the committed file.
   - Fails when generated output differs (stale registry docs are blocking).
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md
     git add docs/remediation_codes_v1.md
     ```

8. **v4 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/backend_evidence_snapshot_v4.md`
     - `docs/operator_signoff_v4.md`
     - `docs/remediation_codes_v1.md`
     - `docs/artifact_schema_snapshots.md`
   - Also requires Prompt 216 tracking in `docs/series_state_snapshot.md`.

## Report output

When `--out` is provided, lint writes deterministic JSON with per-check status and remediation hints.


9. **v5 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/active_review_snapshot_v5.md`
     - `docs/sae_burn_resolution_v5.md`
     - `docs/repro_pack.md`
     - `docs/bug_report_kit.md`
     - `docs/remediation_consistency_v5.md`
     - `docs/artifact_schema_snapshots.md`


10. **v6 docs linkage consistency**
   - Note: portability smokes run `models supported-set-review` before `models supported-set-apply` so apply remains deterministic and offline.
   - Requires presence and portability/docs linkage for:
     - `docs/governance_primary_surfaces_v6.md`
     - `docs/supported_set_apply_v6.md`
     - `docs/applied_supported_scope_v6.md`
     - `docs/export_normalization_v6.md`
     - `docs/interop_consistency_v6.md`
     - `docs/artifact_schema_snapshots.md`


11. **v7 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/applied_scope_authority_v7.md`
     - `docs/supported_scope_reevaluation_v7.md`
     - `docs/reviewability_truth_v7.md`
     - `docs/export_roundtrip_v7.md`
     - `docs/remediation_interop_consistency_v7.md`
     - `docs/artifact_schema_snapshots.md`

12. **v8 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/canonical_governance_entry_v8.md`
     - `docs/supported_scope_execution_v8.md`
     - `docs/readiness_spine_v8.md`
     - `docs/bundle_spine_v8.md`
     - `docs/remediation_spine_consistency_v8.md`
     - `docs/artifact_schema_snapshots.md`

13. **v9 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/canonical_governance_entry_sweep_v9.md`
     - `docs/supported_scope_execution_v9.md`
     - `docs/canonical_readiness_sweep_v9.md`
     - `docs/canonical_bundle_sweep_v9.md`
     - `docs/primary_semantics_sweep_v9.md`
     - `docs/artifact_schema_snapshots.md`

14. **v10 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/final_governance_consumer_sweep_v10.md`
     - `docs/supported_scope_execution_v10.md`
     - `docs/final_readiness_consumer_sweep_v10.md`
     - `docs/final_bundle_consumer_sweep_v10.md`
     - `docs/final_primary_semantics_sweep_v10.md`
     - `docs/final_continuity_sweep_v10.md`
     - `docs/artifact_schema_snapshots.md`

15. **v11 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_residual_sweep_v11.md`
     - `docs/supported_scope_execution_v11.md`
     - `docs/readiness_residual_sweep_v11.md`
     - `docs/bundle_residual_sweep_v11.md`
     - `docs/primary_semantics_residual_sweep_v11.md`
     - `docs/artifact_schema_snapshots.md`
16. **v12 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/residual_free_governance_sweep_v12.md`
     - `docs/supported_scope_execution_v12.md`
     - `docs/residual_free_readiness_sweep_v12.md`
     - `docs/residual_free_bundle_sweep_v12.md`
     - `docs/residual_free_primary_semantics_sweep_v12.md`
     - `docs/artifact_schema_snapshots.md`
17. **v13 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_absolute_sweep_v13.md`
     - `docs/supported_scope_execution_v13.md`
     - `docs/readiness_absolute_sweep_v13.md`
     - `docs/bundle_absolute_sweep_v13.md`
     - `docs/primary_semantics_absolute_sweep_v13.md`
     - `docs/artifact_schema_snapshots.md`
18. **v14 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_terminal_sweep_v14.md`
     - `docs/supported_scope_execution_v14.md`
     - `docs/readiness_terminal_sweep_v14.md`
     - `docs/bundle_terminal_sweep_v14.md`
     - `docs/primary_semantics_terminal_sweep_v14.md`
     - `docs/artifact_schema_snapshots.md`
19. **v15 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_ultimate_sweep_v15.md`
     - `docs/supported_scope_execution_v15.md`
     - `docs/readiness_ultimate_sweep_v15.md`
     - `docs/bundle_ultimate_sweep_v15.md`
     - `docs/primary_semantics_ultimate_sweep_v15.md`
     - `docs/ultimate_terminal_absolute_final_input_continuity_v15.md`
     - `docs/artifact_schema_snapshots.md`
20. **v16 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_convergence_sweep_v16.md`
     - `docs/supported_scope_execution_v16.md`
     - `docs/readiness_convergence_sweep_v16.md`
     - `docs/bundle_convergence_sweep_v16.md`
     - `docs/primary_semantics_convergence_sweep_v16.md`
     - `docs/artifact_schema_snapshots.md`

21. **v17 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_stabilization_sweep_v17.md`
     - `docs/supported_scope_execution_v17.md`
     - `docs/readiness_stabilization_sweep_v17.md`
     - `docs/bundle_stabilization_sweep_v17.md`
     - `docs/primary_semantics_stabilization_sweep_v17.md`
     - `docs/canonical_stabilization_continuity_v17.md`
     - `docs/artifact_schema_snapshots.md`

22. **v18 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/governance_final_consolidation_sweep_v18.md`
     - `docs/supported_scope_execution_v18.md`
     - `docs/readiness_final_consolidation_sweep_v18.md`
     - `docs/bundle_final_consolidation_sweep_v18.md`
     - `docs/primary_semantics_final_consolidation_sweep_v18.md`
     - `docs/canonical_final_consolidation_continuity_v18.md`
     - `docs/artifact_schema_snapshots.md`
