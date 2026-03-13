# Portability Gate v5 Refresh (Linux + Windows)

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

2. **v5 generation smoke checks (blocking unless explicitly optional)**
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
   - Enforces v3 + v4 + v5 docs consistency and linkage.
   - Enforces remediation registry doc freshness.
   - Enforces artifact schema snapshot freshness and deterministic drift reporting.

5. **Consolidated portability summary (`portability report`)**
   - Orchestrates checks and writes `./out/portability_report.json`.
   - Emits explicit `PASS|FAIL|SKIP` per section.

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

## Local run instructions

### Linux/macOS shell

```bash
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json || echo "backend_resolution=skip optional_second_slot_path_unavailable"
cargo run -p ucf-ops -- bringup --demo --ticks 16 --workdir ./.ucf_portability_smoke
run_id=$(python -c 'import json;print(json.load(open("./.ucf_portability_smoke/ess/run_metadata_record.json", "r", encoding="utf-8"))["run_id"])')
cargo run -p ucf-ops -- repro pack --run "$run_id" --out ./out/repro_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- bugkit build --run "$run_id" --out ./out/bugkit_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json
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
cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json
cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json
if ($LASTEXITCODE -ne 0) { Write-Host "backend_resolution=skip optional_second_slot_path_unavailable"; $global:LASTEXITCODE = 0 }
cargo run -p ucf-ops -- bringup --demo --ticks 16 --workdir ./.ucf_portability_smoke
$run_record = Get-Content ".\.ucf_portability_smoke\ess\run_metadata_record.json" | ConvertFrom-Json
if (-not $run_record.run_id) { throw "run_metadata_record.json missing run_id" }
$run_id = $run_record.run_id
cargo run -p ucf-ops -- repro pack --run $run_id --out ./out/repro_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- repro verify --pack ./out/repro_portability.zip --out ./out/repro_verify_portability.json --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- bugkit build --run $run_id --out ./out/bugkit_portability.zip --workdir ./.ucf_portability_smoke
cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json
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
- Required docs/path/hardware/schema checks are expected to `PASS` on supported Linux/Windows setups.
