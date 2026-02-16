# Real Compute Onboarding v0 E2E

This document describes the deterministic end-to-end test coverage for the runtime compute onboarding flow.

## Wiring map

The integration path covered by `runtime/ucf-runtime/tests/e2e_real_compute_onboarding.rs` is:

1. **Entry / ingestion**
   - A deterministic stream of `ControlFrame` values is generated from JSON fixtures in `fixtures/`.
2. **Compute chain**
   - `context_digest` is derived from each control frame.
   - Compute backend executes world model (JEPA stub), SAE extraction, and SSM update.
   - Runtime attaches risk/confidence and evidence-chain digests into `ComputeSignalsSummary`.
3. **Decision chain**
   - Candidate generation emits `CandidateSetRecord`.
   - Selection is deterministic and tied to the compute summary + policy gate.
4. **Output chain**
   - LLM stub produces bounded deterministic output for allowed classes.
   - `OutputRecord` is appended and linked by `decision_id` and `evidence_chain_digest`.
5. **ESS append chain**
   - For each tick: `ControlIn -> DecisionOut -> CandidateSet -> Output -> Nsr`.
   - Records stay bounded (digest/scalar centric) and preserve deterministic ordering.
6. **Optional hooks**
   - Consolidation + geist hooks are enabled in E2E test mode.
   - Hooks are exercised without tool activation and audited via runtime counters.

## Fixtures

- `fixtures/e2e_scenario_a.json`: baseline, 32 ticks, low-volatility signal.
- `fixtures/e2e_scenario_b.json`: stress, 32 ticks, high-volatility signal.

Both fixtures are deterministic, compact, and offline-friendly.

## What the E2E tests assert

- Scalar bounds for surprise/pressure/risk/confidence (`[0,1]`).
- Presence of load-bearing digests (`compute_chain_digest`, NSR digest, output digest).
- Stress trend checks:
  - average pressure in scenario B > scenario A,
  - average risk in scenario B >= scenario A,
  - at least one budget-degraded tick in scenario B.
- ESS linking invariants:
  - candidate/output records carry valid decision/evidence references,
  - per-correlation record ordering is stable,
  - no tool/sandbox execution path is used (deny-by-default).
- Optional hook path executes without panics.
- Golden digest prefixes (ticks `0,1,2,15,31`) detect drift.

## Running

```bash
cargo test -p ucf-runtime --test e2e_real_compute_onboarding
```

Or full workspace regression:

```bash
cargo test --workspace
```

## Extending scenarios

- Add another fixture JSON under `fixtures/`.
- Reuse the same schema (`scenario`, `ticks`, `channel`, `intent_summary`, `signal_values`).
- Add a `run_scenario(..., budget_profile)` invocation and trend/golden checks for representative checkpoints.
