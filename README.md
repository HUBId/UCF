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
