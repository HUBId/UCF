# UCF UNIFIED COGNITIVE FABRIC

## Canonical feature matrix (production)
Supported, CI-enforced lanes:
- `default (toy)`: `cargo test --workspace --all-targets`
- `candle-cpu`: `--features "compute-candle,llm-candle,lfm-candle"`
- `burn-cpu`: `--features "compute-burn,backend-burn,llm-burn,lfm-burn"`
- `stage-isolation`: `--features "sandbox-wasm,stage-isolation"`
- `ebm-train` (tools-only): `cargo test -p ucf-ebm-train --features "ebm-train"`

See `docs/feature_matrix.md` for details.

## Post-rc1 hardening commands
- Determinism scan: `cargo run -p ucf-ops -- determinism scan`
- Hidden path audit scan: `cargo run -p ucf-ops -- audit scan`
- Hardware assumptions scan: `cargo run -p ucf-ops -- audit hardware-scan`
- Spec snapshot: `cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md`
- Docs lint (CI-blocking): `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`

## Portability/docs gate (Linux + Windows, v9 refresh)
- `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
- `cargo run -p ucf-ops -- audit path-scan`
- `cargo run -p ucf-ops -- audit hardware-scan`
- `cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json` (Linux lane)
- `cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json`
- `cargo run -p ucf-ops -- governance-entry-check --out ./out/governance_entry_check.json`
- `cargo run -p ucf-ops -- governance-entry-sweep --out ./out/governance_entry_sweep.json`
- `cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json`
- `cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v4 --out ./out/supported_scope_execute_v4.json --workdir .`
- `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
- `cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json`
- `cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json`
- `cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json`
- `cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json`
- `cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json`
- `cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json`
- `cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json`
- `cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json`
- `cargo run -p ucf-ops -- governance-surfaces-check --out ./out/governance_surfaces_check.json`
- `cargo run -p ucf-ops -- models supported-set-review --out ./out/supported_set_review.json --workdir .`
- `cargo run -p ucf-ops -- models supported-set-apply --out ./out/supported_set_apply.json --workdir .`
- `cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json --workdir .`
- `cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json`
- `cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json`
- `cargo run -p ucf-ops -- models active-review-snapshot --out ./out/active_review_snapshot.json`
- `cargo run -p ucf-ops -- models backend-resolution --slot sae --out ./out/backend_resolution_sae.json` *(optional path may SKIP if second-slot differs)*
- `cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency_portability.json`
- `cargo run -p ucf-ops -- models evidence-snapshot --out ./out/backend_evidence_snapshot.json`
- `cargo run -p ucf-ops -- operator signoff --out ./out/operator_signoff.json`
- `cargo run -p ucf-ops -- docs remediation-codes --out ./out/remediation_codes_v1.generated.md`
- `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
- `cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json`
- `cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json`
- `cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json`
- `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json`
- `cargo run -p ucf-ops -- operator report --out ./out/operator_report.json`
- `cargo run -p ucf-ops -- portability report --out ./out/portability_report.json`

## Architecture
- See `docs/architecture/COHERENCE_LOOP.md`.
- See `docs/architecture/DELTA_ONN_SNN.md` (ONN/SNN delta spec).
- See `docs/architecture/MODULE_COMMITS.md`.

## Demo
- Run the deterministic checkpoint demo:
  - `cargo run -q -p ucf-demo -- --cycles 12 --seed 42`
- The demo prints one coherence summary line per cycle (gamma/plv/lock/surprise/learning/delta/NSR).
- Full operator notes: `docs/ops/DEMO.md`.


## Real Compute Onboarding v0 Quick Start
- One-command bringup:
  - `cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/<run_id>`
- Readiness gate:
  - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/<run_id>/gate_report.json`
- Full signoff bundle:
  - `release/v0_signoff_checklist.md` (human)
  - `release/v0_signoff_checklist.toml` (machine)
  - `docs/freeze_v0_index.md`
  - `docs/v0_scope.md`
  - `docs/artifact_convention_v0.md`


## Minimal Viable Governance v2 Profiles
- Profile ladder via `UCF_PROFILE=dev|test|prod` with config files in `configs/`.
- Device budget profile via `UCF_DEVICE_PROFILE=small|medium|large` (resource budgets only).
- Start commands:
  - `UCF_PROFILE=dev cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/dev`
  - `UCF_PROFILE=test cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/test`
  - `UCF_PROFILE=prod cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/prod`
- Config contract v1: `docs/config_contract_v1.md`.
- Validate config schema: `cargo run -p ucf-ops -- config validate --in configs/test.toml`.
- Unknown config keys fail fast during TOML parse/load (`ConfigV1` with strict schema).
- Migrate legacy config: `cargo run -p ucf-ops -- config migrate --in old.toml --out new.toml --diff ./out/config_diff.txt`.
- Export policy key registry: `cargo run -p ucf-ops -- policy keys --out docs/policy_key_registry.md`.
- Migration guide: `docs/config_migration_v2.md`.


## Portable bringup (bundle-first)
- Build bundle:
  - `python deploy/scripts/build_bundle.py --target ./bundles/releases/ucf_v1 --profile prod`
- Switch to bundle root and run local gates:
  - `./bundles/releases/ucf_v1/bin/ucf-ops readiness-gate --bundle ./bundles/releases/ucf_v1 --profile test --out ./bundles/releases/ucf_v1/out/gate.json`
  - `./bundles/releases/ucf_v1/bin/ucf-ops docs lint --bundle ./bundles/releases/ucf_v1 --strict --out ./bundles/releases/ucf_v1/out/docs_lint.json`
- Upgrade/rollback via bundle switching:
  - `./deploy/scripts/upgrade_bundle.sh upgrade <bundle_id>`
  - `./deploy/scripts/upgrade_bundle.sh rollback <bundle_id>`
- Full guide: `docs/deploy_portable.md`.

## Prompt Runner
- Queue file (machine): `docs/prompt_queue.toml`
- Queue file (human): `docs/prompt_queue.md`
- Usage guide: `docs/prompt_runner.md`

Typical flow:
1. Add/edit prompt entries in `docs/prompt_queue.toml` (or use `python scripts/prompt_runner.py add ...`).
2. Pull next prompt (and create deterministic logs):
   - `python scripts/prompt_runner.py next`
3. After manual execution, mark result:
   - success: `python scripts/prompt_runner.py done <id>`
   - failure: `python scripts/prompt_runner.py fail <id> --reason "..."`
4. Validate queue integrity:
   - `python scripts/prompt_runner.py self-check`

Safety defaults:
- Mutating commands refuse dirty working trees unless `--allow-dirty` is passed.
- Logs are written to `./out/prompt_runs/<id>/`.
- Offline guard is best-effort and warning-only (proxy/env + command hints).

## Prompt series workflow
- Prompt index (1–128): `docs/prompt_series_index.md`
- Module impact map: `docs/module_map.md`
- Next-prompt authoring rules: `docs/prompt_rulebook.md`
- Canonical copy/paste template: `docs/codex_prompt_template.txt`
- Filled example: `docs/codex_prompt_template_example.txt`
- Current series state snapshot: `docs/series_state_snapshot.md`
- Contributor workflow: `docs/contributing_workflow.md`
- Canonical branch policy: `docs/branch_policy.md`
- Codex repo instructions: `AGENTS.md`

Quick workflow:
1. Start at the next monotonic prompt ID and follow `docs/prompt_rulebook.md`.
2. Use `docs/codex_prompt_template.txt` for new prompts and place task-specific content between `START_TASK_SPECIFIC` / `END_TASK_SPECIFIC`.
3. Optional helper: `python scripts/prompt_runner.py render --id <id> --template docs/codex_prompt_template.txt`.
4. Implement changes and update index/module map entries for the new prompt.
5. Run readiness/signoff checks as applicable:
   - v1.0-rc1: `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/<run_id>/gate_report.json`
   - v1.1-rc1: `cargo run -p ucf-ops -- readiness-gate --profile v1_1_rc1 --out ./out/<run_id>/v1_1_gate_report.json`
6. Keep release signoff artifacts aligned (`release/*.md`, `release/*.toml`).


See `docs/docs_checks.md` for remediation guidance when docs lint fails.

Operator/reviewer end-state reference:
- `docs/end_state.md`


## Stable Core API (`ucf-sdk`)
- Crate: `ucf-sdk`
- Stable boundary types:
  - `ControlFrameV1`
  - `DecisionEventV1`
  - `EssSummaryQueryV1`
  - `EssSummaryResponseV1`
  - `Digest32`, `UQ0_16` (stable re-exports)
- Deterministic encoding helpers are available on all boundary structs via `encode_deterministic()`.
- Internal engine modules are intentionally not re-exported.

Local API compatibility checks:
- `python scripts/sdk_api_snapshot.py generate`
- `python scripts/sdk_api_snapshot.py check --baseline-ref HEAD^`

See `docs/sdk_versioning.md` for semver and deprecation policy.
