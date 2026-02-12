# NSR v0 (Policy Ecology Reasoner)

NSR v0 provides a deterministic, offline, bounded risk signal for decision/tool intents.

## What it does
- Builds bounded facts from decision intent and runtime context.
- Runs a deterministic Datalog-lite forward-chaining engine.
- Produces `NsrAssessment`:
  - `nsr_risk` (0..1)
  - `nsr_confidence` (0..1)
  - bounded `reasons`
  - `facts_digest` and `digest`
  - `ruleset_id`, `schema_version`, `engine_id`
  - `policy_hint` (`Block`, `SafeOnly`, `Normal`)

## What it does not do
- No direct action execution.
- No external SMT/Datalog backend in v0.
- No network dependency.

## Rule format
Rules are bounded and deterministic:
- `Rule { head, body, weight }`
- body predicates are `HasFact(Fact)` or `Not(Fact)`.
- Rule iteration order is stable (index order).

## Determinism and boundedness
- Ordered fact set (`BTreeSet`).
- Canonical fact/reason hashing before digesting.
- Budget caps for rules, facts, inference steps, and reasons.
- Failure mode supports:
  - fail-open default assessment (`risk=1.0`, `confidence=0.0`), or
  - fail-fast error via config.

## ToolGate interaction
- NSR is advisory signal provider + gating hint.
- ToolGate remains hard enforcement point.
- Runtime integration maps `PolicyHint::Block` to decision gating metadata (`nsr_block`) and persistence in ESS.

## Extensibility
`NsrEngine` trait supports backend swapping:
- `NsrDatalogLiteEngine` (implemented)
- `NsrSmtEngine` stub (`BackendDisabled`)
