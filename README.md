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
- Spec snapshot: `cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md`
- Docs lint (CI-blocking): `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`

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
- Start commands:
  - `UCF_PROFILE=dev cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/dev`
  - `UCF_PROFILE=test cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/test`
  - `UCF_PROFILE=prod cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out/prod`
- Unknown config keys fail fast during TOML parse/load.
- Migration guide: `docs/config_migration_v2.md`.

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
