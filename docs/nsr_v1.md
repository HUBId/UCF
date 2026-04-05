# NSR v1 — bounded Datalog-lite risk

NSR v1 computes a deterministic, offline, bounded risk score (`nsr_risk_q`) from context and candidate metadata.

## DSL (minimal)
- Facts/predicates:
  - `candidate_kind(tool|json|text)`
  - `tool_class(<u8>)`
  - `context_risk_level(low|med|high)`
  - `emergency(active)`
  - `budget_low(true)`
- Rule form:
  - `head :- body1, body2, not body3.`
- Risk terms:
  - `risk_term(term_id, weight_q)` where `term_id: u16`, `weight_q: u16 (UQ0_16)`.

## Boundedness
- max input facts: 128
- max derived facts: 256
- max reasons: 8
- max eval steps: 2048

Budget overflow returns conservative output (`BudgetExceeded`, risk=1.0).

## Determinism
- Stable rule order.
- Canonical fact keys with deterministic forward chaining.
- Deterministic reason sorting (contribution desc, then term id).

## Fail-safe
If rule parsing fails:
- `status = ParseErrorFallback`
- Tool candidate -> `risk=1.0`
- Non-tool candidate -> elevated conservative risk.

## Fusion with EBM and Governor
- EBM consumes NSR via `NsrRiskAmplifier` term in `ebm_constraints.toml`.
- Governor adds `nsr_penalty_q` only above threshold.
- Tightening-only: NSR can only increase score/tier strictness, never decrease.

## Explain Tick
Explain output is digest-only:
- `nsr_risk_q`
- `nsr_status`
- reason ids (`nsr_reasons`)
- `nsr_rules_digest_prefix`

No raw payloads or full proof traces are exposed.

## Runtime readiness (canonical pipeline attachment)

NSR readiness is tracked with the canonical stage ladder:

- `scaffolded`
- `contract-ready`
- `artifact-ready`
- `runtime-path-ready`
- `production-blocked`

Current repo status (`runtime/ucf-compute`):

- `contract-ready`: yes (`NsrRequest` / `NsrResult` / `NsrFailureKind` in `contracts.rs`)
- `artifact-ready`: partial (`ebm_reasoner` slot provenance wired; no dedicated NSR slot family yet)
- `runtime-path-ready`: partial (`UCF_NSR_MODE=best_effort|required` runs NSR as post-inference hook)
- `production-blocked`: yes

Primary blockers to broader production usage:

1. Intent projection into NSR is minimal (no full action/tool plan contract in `ucf-compute` path yet).
2. NSR artifact contract currently reuses `ebm_reasoner` seam; dedicated NSR slot contract is not finalized.
3. No backend-specific NSR execution matrix across Burn/Candle runtime packs beyond the shared post-hook.
4. Required-mode rollout policy/thresholds are not calibrated for production traffic.
