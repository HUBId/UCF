# UCF UNIFIED COGNITIVE FABRIC

## Canonical feature matrix (production)
Supported, CI-enforced lanes:
- `default (toy)`: `cargo test --workspace --all-targets`
- `candle-cpu`: `--features "compute-candle,llm-candle,lfm-candle"`
- `burn-cpu`: `--features "compute-burn,backend-burn,llm-burn,lfm-burn"`
- `stage-isolation`: `--features "sandbox-wasm,stage-isolation"`
- `ebm-train` (tools-only): `cargo test -p ucf-ebm-train --features "ebm-train"`

See `docs/feature_matrix.md` for details.

## Consolidation claim boundary

Current consolidation claims are bounded: Micro/Meso explicit append/readback, Macro candidate, and a local consolidation-level finalization boundary are covered by the consolidation roadmap/E2E tests; Replay/Sleep/Geist/ISM, identity anchoring, Gateway writes, capabilities, real compute, and production consolidation readiness remain deferred. See `docs/roadmap/full_consolidation_roadmap_boundary_audit.md` and `docs/roadmap/consolidation_record_authority_schema_alignment.md`.

Current replay claims are also bounded: deterministic ReplayToken intent/reference, ReplaySchedule planned ordering, verify-only ReplayAudit, local-only ReplayAppliedBoundary, bounded Replay E2E determinism, and Prompt 65 Evidence/Archive append/readback as audit/provenance persistence only. There is no runtime replay apply, scheduler/queue/worker, Sleep/Geist/ISM/identity runtime integration, Gateway-visible replay, production readiness, or second event log. See `docs/roadmap/replay_scheduler_roadmap_boundary_audit.md`, `docs/roadmap/replay_record_authority_schema_alignment.md`, and `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`.

Current Sleep claims are bounded to a deterministic `SleepPlanCandidate` builder from bounded Replay metadata, verify-only `SleepPlanAudit`, local-only `SleepAppliedBoundary`, bounded Sleep E2E determinism, and Prompt 66 Evidence/Archive append/readback as audit/provenance persistence only. There is no Sleep runtime, Sleep Cycle Coordinator activation, coordinator trigger/report/WAL/journal, SleepCompleted, memory stabilization, Geist/ISM or identity runtime integration, Gateway-visible Sleep, production Sleep readiness, or second event log. See `docs/roadmap/sleep_integration_roadmap_boundary_audit.md`, `docs/roadmap/sleep_record_authority_schema_alignment.md`, and `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`.

Current Geist/ISM claims are bounded to `GeistProjectionCandidate` candidate-only, `GeistProjectionAudit` verify-only, `ISMCandidateBoundary` local read-model/candidate-only, bounded Geist/ISM E2E determinism from Sleep-derived input, and Prompt 67 Evidence/Archive append/readback as audit/provenance persistence only. Prompt 68 adds deterministic cross-layer Replay → Sleep → Geist/ISM readback E2E. There is no Geist runtime, `GeistApplied`, ISM write/upsert, `IdentityAnchor`, `IdentityFinalization`, memory stabilization, Policy mutation, Gateway/action authority, production Geist/ISM readiness, or second event log. See `docs/roadmap/geist_ism_roadmap_boundary_audit.md`, `docs/roadmap/geist_ism_record_authority_schema_alignment.md`, and `docs/roadmap/evidence_archive_append_contracts_roadmap_boundary_audit.md`.

## Post-rc1 hardening commands
- Determinism scan: `cargo run -p ucf-ops -- determinism scan`
- Hidden path audit scan: `cargo run -p ucf-ops -- audit scan`
- Hardware assumptions scan: `cargo run -p ucf-ops -- audit hardware-scan`
- Spec snapshot: `cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md`
- Docs lint (CI-blocking): `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`

## Portability/docs gate (Linux + Windows, v18 refresh)
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
- `cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json`
- `cargo run -p ucf-ops -- governance-residual-sweep --out ./out/governance_residual_sweep.json`
- `cargo run -p ucf-ops -- residual-free-governance-sweep --out ./out/residual_free_governance_sweep.json`
- `cargo run -p ucf-ops -- governance-absolute-sweep --out ./out/governance_absolute_sweep.json`
- `cargo run -p ucf-ops -- governance-terminal-sweep --out ./out/governance_terminal_sweep.json`
- `cargo run -p ucf-ops -- governance-ultimate-sweep --out ./out/governance_ultimate_sweep.json`
- `cargo run -p ucf-ops -- governance-convergence-sweep --out ./out/governance_convergence_sweep.json`
- `cargo run -p ucf-ops -- governance-stabilization-sweep --out ./out/governance_stabilization_sweep.json`
- `cargo run -p ucf-ops -- governance-final-consolidation-sweep --out ./out/governance_final_consolidation_sweep.json`
- `cargo run -p ucf-ops -- models supported-scope-execute-v7 --out ./out/supported_scope_execute_v7.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v8 --out ./out/supported_scope_execute_v8.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v9 --out ./out/supported_scope_execute_v9.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v10 --out ./out/supported_scope_execute_v10.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v12 --out ./out/supported_scope_execute_v12.json --workdir .`
- `cargo run -p ucf-ops -- models supported-scope-execute-v13 --out ./out/supported_scope_execute_v13.json --workdir .`
- `cargo run -p ucf-ops -- governance-closure-sweep --out ./out/governance_closure_sweep.json`
- `cargo run -p ucf-ops -- models supported-scope-execute-v14 --out ./out/supported_scope_execute_v14.json --workdir .`
- `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json`
- `cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json`
- `cargo run -p ucf-ops -- final-readiness-consumer-sweep --out ./out/final_readiness_consumer_sweep.json`
- `cargo run -p ucf-ops -- readiness-residual-sweep --out ./out/readiness_residual_sweep.json`
- `cargo run -p ucf-ops -- residual-free-readiness-sweep --out ./out/residual_free_readiness_sweep.json`
- `cargo run -p ucf-ops -- readiness-absolute-sweep --out ./out/readiness_absolute_sweep.json`
- `cargo run -p ucf-ops -- readiness-terminal-sweep --out ./out/readiness_terminal_sweep.json`
- `cargo run -p ucf-ops -- readiness-ultimate-sweep --out ./out/readiness_ultimate_sweep.json`
- `cargo run -p ucf-ops -- readiness-convergence-sweep --out ./out/readiness_convergence_sweep.json`
- `cargo run -p ucf-ops -- readiness-stabilization-sweep --out ./out/readiness_stabilization_sweep.json`
- `cargo run -p ucf-ops -- readiness-final-consolidation-sweep --out ./out/readiness_final_consolidation_sweep.json`
- `cargo run -p ucf-ops -- readiness-closure-sweep --out ./out/readiness_closure_sweep.json`
- `cargo run -p ucf-ops -- operator review-truth-check --out ./out/review_truth_check.json`
- `cargo run -p ucf-ops -- exports roundtrip-check --in ./out/repro_portability.zip --out ./out/export_roundtrip_check.json`
- `cargo run -p ucf-ops -- exports bundle-spine-check --in ./out/repro_portability.zip --out ./out/bundle_spine_check.json`
- `cargo run -p ucf-ops -- exports bundle-spine-sweep --out ./out/bundle_spine_sweep.json`
- `cargo run -p ucf-ops -- final-bundle-consumer-sweep --out ./out/final_bundle_consumer_sweep.json`
- `cargo run -p ucf-ops -- bundle-residual-sweep --out ./out/bundle_residual_sweep.json`
- `cargo run -p ucf-ops -- residual-free-bundle-sweep --out ./out/residual_free_bundle_sweep.json`
- `cargo run -p ucf-ops -- bundle-absolute-sweep --out ./out/bundle_absolute_sweep.json`
- `cargo run -p ucf-ops -- bundle-terminal-sweep --out ./out/bundle_terminal_sweep.json`
- `cargo run -p ucf-ops -- bundle-ultimate-sweep --out ./out/bundle_ultimate_sweep.json`
- `cargo run -p ucf-ops -- bundle-convergence-sweep --out ./out/bundle_convergence_sweep.json`
- `cargo run -p ucf-ops -- bundle-stabilization-sweep --out ./out/bundle_stabilization_sweep.json`
- `cargo run -p ucf-ops -- bundle-final-consolidation-sweep --out ./out/bundle_final_consolidation_sweep.json`
- `cargo run -p ucf-ops -- bundle-closure-sweep --out ./out/bundle_closure_sweep.json`
- `cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json`
- `cargo run -p ucf-ops -- remediation-spine-check --out ./out/remediation_spine_check.json`
- `cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json`
- `cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-residual-sweep --out ./out/primary_semantics_residual_sweep.json`
- `cargo run -p ucf-ops -- residual-free-primary-semantics-sweep --out ./out/residual_free_primary_semantics_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-absolute-sweep --out ./out/primary_semantics_absolute_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-terminal-sweep --out ./out/primary_semantics_terminal_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-ultimate-sweep --out ./out/primary_semantics_ultimate_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-convergence-sweep --out ./out/primary_semantics_convergence_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-stabilization-sweep --out ./out/primary_semantics_stabilization_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-final-consolidation-sweep --out ./out/primary_semantics_final_consolidation_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-closure-sweep --out ./out/primary_semantics_closure_sweep.json`
- `cargo run -p ucf-ops -- governance-seal-sweep --out ./out/governance_seal_sweep.json`
- `cargo run -p ucf-ops -- models supported-scope-execute-v15 --out ./out/supported_scope_execute_v15.json --workdir .`
- `cargo run -p ucf-ops -- readiness-seal-sweep --out ./out/readiness_seal_sweep.json`
- `cargo run -p ucf-ops -- bundle-seal-sweep --out ./out/bundle_seal_sweep.json`
- `cargo run -p ucf-ops -- primary-semantics-seal-sweep --out ./out/primary_semantics_seal_sweep.json`
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
- `cargo run -p ucf-ops -- portability report --out ./out/portability_report_v20.json`

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

Current compute lanes are bounded by the docs-only compute matrix: `docs/roadmap/compute_feature_ci_matrix.md` and `docs/roadmap/real_compute_lane_inventory.md`. Stub fixture and toy golden lanes are not real inference; optional-real and remote/external lanes are compile-only unless a future local artifact-backed runtime fixture proves otherwise. Production compute claims are forbidden for current lanes.

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

- `docs/residual_free_continuity_v11.md`


## Readiness claim boundary

- This repository may produce passing **test-profile** readiness artifacts without claiming prod readiness.
- `SKIP`, `TIMEOUT`, missing evidence, or stale `out/*.json` reports are not `PASS` and not current truth.
- Prod-readiness claims require fresh split evidence (`workspace-test-check` + `readiness-gate --profile prod`) and explicit blocker closure.
