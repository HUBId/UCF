# Portability Gate v10 Refresh (Linux + Windows)

`Portability Gate` blocks merges when core runtime/ops checks are not cross-platform safe.

## What is checked

1. **Cross-platform CI matrix (blocking)**
   - Linux lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json`
     - `cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json`
     - `cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json`
     - `cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json`
     - `cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json`
     - `cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json --workdir .`
     - `cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir .`
     - `cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json`
     - `cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir .`
     - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
     - `cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json`
     - `cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json`
     - `cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir .`
     - `cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir .`
     - `cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir .`
     - `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`
     - `cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json`
     - `cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json`
     - `cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json`
     - `cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json`
     - `cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json`
     - `cargo run -p ucf-ops -- docs remediation-codes --out ./out/remediation_codes_v1.generated.md` (must match `docs/remediation_codes_v1.md`)
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json`
     - `cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json`
     - `cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json`
     - `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json`
     - `cargo run -p ucf-ops -- operator report --out ./out/operator_report.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
     - `cargo run -p ucf-ops -- portability report --out ./out/portability_report.json`
   - Windows lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json`
     - `cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json`
     - `cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json`
     - `cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json`
     - `cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json --workdir .`
     - `cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir .`
     - `cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json`
     - `cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir .`
     - `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
     - `cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json`
     - `cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json`
     - `cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir .`
     - `cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir .`
     - `cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir .`
     - `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`
     - `cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json`
     - `cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json`
     - `cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json`
     - `cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json`
     - `cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json`
     - `cargo run -p ucf-ops -- docs remediation-codes --out ./out/remediation_codes_v1.generated.md` (must match `docs/remediation_codes_v1.md`)
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json`
     - `cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json`
     - `cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json`
     - `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json`
     - `cargo run -p ucf-ops -- operator report --out ./out/operator_report.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
     - `cargo run -p ucf-ops -- portability report --out ./out/portability_report.json`

2. **v10 generation smoke checks (blocking unless explicitly optional)**
   - `governance-entry-check` must pass and preserve canonical governance entry usage across consumers.
   - `governance-entry-sweep` must pass and prove deterministic canonical entry authority across final governance surfaces.
   - `models supported-scope-execute` must produce deterministic bounded execution decisions.
   - `models supported-scope-execute-v4` must emit deterministic `REAFFIRM_FREEZE` / `EXECUTE_EXPAND_BY_ONE` execution decisions.
   - `final-governance-consumer-sweep` must pass and prove deterministic final governance consumer authority coverage.
   - `models supported-scope-execute-v5` must emit deterministic `REAFFIRM_FREEZE` / `EXECUTE_EXPAND_BY_ONE` decisions against final governance consumer authority.
   - `final-readiness-consumer-sweep` must pass and emit deterministic mismatch categories for canonical final readiness consumers.
   - `final-bundle-consumer-sweep` must pass and prove canonical bundle-input authority consumption across canonical consumers.
   - `readiness-spine-check` must emit deterministic mismatch categories and remediation codes.
   - `readiness-spine-sweep` must pass and prove deterministic final readiness authority coverage.
   - `exports bundle-spine-check` must reconstruct canonical bundle spine deterministically from bounded fixture bundles.
   - `exports bundle-spine-sweep` must reconstruct canonical bundle authority deterministically across repro/bugkit/export surfaces.
   - `primary-semantics-sweep` must prove canonical primary blocking/remediation consistency with deterministic mismatch categories.
   - `final-primary-semantics-sweep` must prove universal consumer enforcement of the same canonical primary authority inputs across canonical surfaces.
   - `remediation-spine-check` must map canonical conditions/remediations consistently across scope/governance/readiness/bundle surfaces.
   - `models active-review-snapshot` and `models backend-resolution` must run in bounded offline mode (optional backend-resolution paths must SKIP cleanly).
   - Enriched export smokes (`repro pack` + `repro verify`, `bugkit build`) must generate deterministic manifests with bounded fixtures and no payload/weight inclusion by default.
   - `remediation-consistency-check` must pass and emit deterministic mismatch categories.
   - `models evidence-snapshot` must run in bounded offline mode and write deterministic JSON shape output.
   - `operator signoff` must produce deterministic signoff output and actionable remediation codes.
   - `docs remediation-codes` must match committed `docs/remediation_codes_v1.md`.
   - `spec artifact-schemas-check` must pass with no drift in committed schema snapshots.

3. **Path/hardware/network guardrails (blocking)**
   - `audit path-scan`: no hard-coded runtime OS service paths.
   - `audit hardware-scan`: no hardware/vendor assumptions in guarded runtime/docs scope.
   - `audit net-deps` (Linux lane): hidden network dependency drift is blocked.

4. **Docs consistency (via `docs lint --strict`)**
   - Enforces v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 docs consistency and linkage.
   - Enforces remediation registry doc freshness.
   - Enforces artifact schema snapshot freshness and deterministic drift reporting.

5. **Consolidated portability summary (`portability report`)**
   - Orchestrates checks and writes `./out/portability_report.json`.
   - Emits explicit `PASS|FAIL|SKIP` per section.

## v6 docs covered by portability/docs gates

- `docs/governance_primary_surfaces_v6.md`
- `docs/supported_set_apply_v6.md`
- `docs/applied_supported_scope_v6.md`
- `docs/export_normalization_v6.md`
- `docs/interop_consistency_v6.md`
- `docs/artifact_schema_snapshots.md`

## v8 docs covered by portability/docs gates

- `docs/canonical_governance_entry_v8.md`
- `docs/supported_scope_execution_v8.md`
- `docs/readiness_spine_v8.md`
- `docs/bundle_spine_v8.md`
- `docs/remediation_spine_consistency_v8.md`
- `docs/artifact_schema_snapshots.md`


## v10 docs covered by portability/docs gates

- `docs/final_governance_consumer_sweep_v10.md`
- `docs/supported_scope_execution_v10.md`
- `docs/final_readiness_consumer_sweep_v10.md`
- `docs/final_bundle_consumer_sweep_v10.md`
- `docs/final_primary_semantics_sweep_v10.md`
- `docs/artifact_schema_snapshots.md`

## v9 docs covered by portability/docs gates

- `docs/canonical_governance_entry_sweep_v9.md`
- `docs/supported_scope_execution_v9.md`
- `docs/canonical_readiness_sweep_v9.md`
- `docs/canonical_bundle_sweep_v9.md`
- `docs/primary_semantics_sweep_v9.md`
- `docs/artifact_schema_snapshots.md`

## v5 docs covered by portability/docs gates

- `docs/active_review_snapshot_v5.md`
- `docs/sae_burn_resolution_v5.md`
- `docs/repro_pack.md`
- `docs/bug_report_kit.md`
- `docs/remediation_consistency_v5.md`
- `docs/artifact_schema_snapshots.md`

## v4 docs covered by portability/docs gates

- `docs/backend_evidence_snapshot_v4.md`
- `docs/operator_signoff_v4.md`
- `docs/remediation_codes_v1.md`
- `docs/artifact_schema_snapshots.md`

## v3 docs retained in portability linkage checks

- `docs/models_eligibility_v3.md`
- `docs/strict_mode_v3.md`
- `docs/operator_report_v3.md`

## Determinism across OS

- The gate enforces deterministic behavior **within each OS lane**.
- Reports are produced in stable ordering for scan outputs and summary sections.
- Schema/remediation drift failures include artifact-level detail and regeneration commands.
- Optional backend/report paths may report `SKIP` (never panic) with stable skip reasons.

## Interpreting FAIL vs SKIP

- **FAIL**: deterministic portability/docs/schema/final-sweep invariants regressed and must be fixed before merge.
- **SKIP**: bounded optional backend/report path is unavailable in the current environment; this is expected and non-panicking.
- Required v10 final sweeps are blocking in normal bounded smoke contexts; `governance-entry-sweep`/`readiness-spine-sweep`/`final-governance-consumer-sweep`/`final-readiness-consumer-sweep` may emit `SKIP` only when optional applied-scope prerequisites are unavailable (`APPLIED_SCOPE_*` guardrails), and `bundle-spine-sweep`/`final-bundle-consumer-sweep` may emit `SKIP` when optional canonical export refs are unavailable (`CANONICAL_EXPORT_REFS_REQUIRED`-class guardrails), never via panic.

## Local run instructions

### Linux/macOS shell

```bash
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json
cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir .
cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir .
cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json
cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json
cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir .
cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir .
cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json
cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json || echo "backend_resolution=skip optional_second_slot_path_unavailable"
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e/v0_flow_a.json --ticks 16 --workdir ./.ucf_portability_smoke --out ./out/portability_smoke_bringup
run_json=$(ls -1 ./.ucf_portability_smoke/ess/runs/*.json | sort | tail -n1)
run_id=$(basename "$run_json" .json)
cargo run -p ucf-ops -- repro pack --run "$run_id" --out ./out/repro_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json
cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json
cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json
cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir .
cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json
cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json
cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json
mkdir -p ./.ucf_portability_smoke/out/$run_id
cp ./out/portability_smoke_bringup/run_metadata.json ./.ucf_portability_smoke/out/$run_id/run_metadata.json
cp ./out/portability_smoke_bringup/metrics_summary.json ./.ucf_portability_smoke/out/$run_id/metrics_summary.json
cargo run -p ucf-ops -- bugkit build --run "$run_id" --out ./out/bugkit_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json
cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json
cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json
cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json
cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json
cargo run -p ucf-ops -- docs remediation-codes --out ./out/remediation_codes_v1.generated.md
diff -u docs/remediation_codes_v1.md ./out/remediation_codes_v1.generated.md
cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json
cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json
cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
cargo run -p ucf-ops -- portability check --out ./out/portability.json
cargo run -p ucf-ops -- portability report --out ./out/portability_report.json
```


### Windows PowerShell

```powershell
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json
cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir .
cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json
cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json
if ($LASTEXITCODE -ne 0) { Write-Host "backend_resolution=skip optional_second_slot_path_unavailable"; $global:LASTEXITCODE = 0 }
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e/v0_flow_a.json --ticks 16 --workdir ./.ucf_portability_smoke --out ./out/portability_smoke_bringup
$run_json = Get-ChildItem ".\.ucf_portability_smoke\ess\runs\*.json" | Sort-Object Name | Select-Object -Last 1
if (-not $run_json) { throw "no run metadata json found under ./.ucf_portability_smoke/ess/runs" }
$run_id = [System.IO.Path]::GetFileNameWithoutExtension($run_json.Name)
cargo run -p ucf-ops -- repro pack --run $run_id --out ./out/repro_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json
cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json
cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json
cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
cargo run -p ucf-ops -- models supported-scope-execute-v5 --out ./out/supported_scope_execute_v5.json --workdir .
cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json
cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json
cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json
New-Item -ItemType Directory -Force -Path ".\.ucf_portability_smoke\out\$run_id" | Out-Null
Copy-Item "./out/portability_smoke_bringup/run_metadata.json" ".\.ucf_portability_smoke\out\$run_id\run_metadata.json"
Copy-Item "./out/portability_smoke_bringup/metrics_summary.json" ".\.ucf_portability_smoke\out\$run_id\metrics_summary.json"
cargo run -p ucf-ops -- bugkit build --run $run_id --out ./out/bugkit_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json
cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json
cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json
cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json
cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json
cargo run -p ucf-ops -- docs remediation-codes --out ./out/remediation_codes_v1.generated.md
git diff --no-index --exit-code docs/remediation_codes_v1.md ./out/remediation_codes_v1.generated.md
cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json
cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json
cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
cargo run -p ucf-ops -- portability check --out ./out/portability.json
cargo run -p ucf-ops -- portability report --out ./out/portability_report.json
```


## Common failures and remediation

- **`spec artifact-schemas-check` failed**
  - Regenerate snapshots and commit updates:
    - `cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots`

- **`models evidence-snapshot` failed**
  - Validate fixture/backend evidence paths and rerun:
    - `cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json`

- **`operator signoff` failed**
  - Recreate supporting bounded reports (`v0/v1/v2`, strict, operator report, eligibility) and rerun:
    - `cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json`

- **remediation codes doc check failed**
  - Regenerate and commit:
    - `cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md`

## FAIL vs SKIP semantics

- **FAIL**: blocking regression; command executed but returned non-pass status or errored.
- **SKIP**: optional backend/report path unavailable; command section is non-blocking and must emit explicit skip reason.
- **SKIP**: bounded readiness context unavailable (for example readiness spine emits only bounded-context drift categories in smoke mode); command section is non-blocking and must emit explicit skip reason.
- **SKIP**: bounded remediation context unavailable (for example remediation spine reports only `MISSING_SURFACE` / `UNKNOWN_CONDITION_MAPPING` categories in smoke mode); command section is non-blocking and must emit explicit skip reason.
- Required docs/path/hardware/schema checks are expected to `PASS` on supported Linux/Windows setups.


- **`exports normalize-check` failed**
  - Re-run bounded export normalization smoke and inspect mismatch categories:
    - `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`

- **`interop consistency-matrix` failed**
  - Regenerate bounded interoperability evidence and rerun matrix:
    - `cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json`


## v7 docs coverage

- `docs/applied_scope_authority_v7.md`
- `docs/supported_scope_reevaluation_v7.md`
- `docs/reviewability_truth_v7.md`
- `docs/export_roundtrip_v7.md`
- `docs/remediation_interop_consistency_v7.md`
- `docs/artifact_schema_snapshots.md`

## v8 docs coverage

- `docs/canonical_governance_entry_v8.md`
- `docs/supported_scope_execution_v8.md`
- `docs/readiness_spine_v8.md`
- `docs/bundle_spine_v8.md`
- `docs/remediation_spine_consistency_v8.md`
- `docs/artifact_schema_snapshots.md`
